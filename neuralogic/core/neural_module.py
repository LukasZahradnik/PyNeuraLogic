from __future__ import annotations

from typing import TYPE_CHECKING, Collection

import jpype

from neuralogic.core.builder.dataset import BuiltDataset, GroundedDataset
from neuralogic.core.builder.static_graph import StaticGraphDataset

if TYPE_CHECKING:
    from neuralogic.core.builder import DatasetBuilder
    from neuralogic.core.settings.settings_proxy import SettingsProxy
from neuralogic.core.constructs.java_objects import ValueFactory
from neuralogic.dataset import Dataset
from neuralogic.dataset.base import BaseDataset
from neuralogic.setup import initialize, is_initialized
from neuralogic.utils.visualize import draw_model

Value = list | float


class NeuralModule:
    """
    NeuralModule is the base class for all neural models.
    It provides methods for grounding, building, training, and testing.
    """

    def __init__(self):
        """Initializes the neural module."""
        if not is_initialized():
            initialize()

        self._need_sync = False
        self._value_factory = ValueFactory()

        self._parsed_model = None
        self._dataset_builder: DatasetBuilder | None = None
        self._settings: SettingsProxy | None = None

        self._neural_model = None
        self._strategy = None
        self._trainer = None

        self._invalidation = None
        self._evaluation = None
        self._backpropagation = None

        self._weight_updater = None
        self._tensor_parameters = None
        self._torch_module = None

    def ground(
        self,
        dataset: BaseDataset,
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
        progress: bool = False,
    ) -> GroundedDataset:
        """Grounds the provided dataset using the model's settings.

        Parameters
        ----------
        dataset : BaseDataset
            The dataset to ground.
        batch_size : int
            The batch size for grounding. Default: 1.
        learnable_facts : bool
            Whether facts are learnable. Default: False.
        progress : bool
            Whether to show progress. Default: False.

        Returns
        -------
        GroundedDataset
            The grounded dataset.
        """
        if self._dataset_builder is None or self._settings is None:
            raise ValueError("model is not built")

        return self._dataset_builder.ground_dataset(
            dataset,
            self._settings,
            batch_size=batch_size,
            learnable_facts=learnable_facts,
            progress=progress,
        )

    def build_static_dataset(
        self,
        dataset: Dataset,
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
        progress: bool = False,
    ) -> StaticGraphDataset:
        """Build a static-graph dataset from a logic ``Dataset``.

        Only the first sample is grounded and neuralized.  The resulting
        ``StaticGraphDataset`` reuses the same neural graph for every sample,
        updating fact values via ``set_fact_value`` before each learning step.

        Parameters
        ----------
        dataset : Dataset
            The dataset with one or more samples sharing identical structure.
        batch_size : int
            The batch size. Default: 1.
        learnable_facts : bool
            Whether facts are learnable. Default: False.
        progress : bool
            Whether to show progress. Default: False.

        Returns
        -------
        StaticGraphDataset
        """
        if self._dataset_builder is None or self._settings is None:
            raise ValueError("model is not built")

        from neuralogic.core.builder.static_graph import build_static_graph_dataset

        return build_static_graph_dataset(
            self._dataset_builder,
            dataset,
            self._settings,
            batch_size=batch_size,
            learnable_facts=learnable_facts,
            progress=progress,
        )

    def build_dataset(
        self,
        dataset: BaseDataset | GroundedDataset,
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
        progress: bool = False,
    ) -> BuiltDataset:
        """Builds (ground and neuralize) the provided dataset.

        Parameters
        ----------
        dataset : Union[BaseDataset, GroundedDataset]
            The dataset to build.
        batch_size : int
            The batch size. Default: 1.
        learnable_facts : bool
            Whether facts are learnable. Default: False.
        progress : bool
            Whether to show progress. Default: False.

        Returns
        -------
        BuiltDataset
            The built dataset.
        """
        if self._dataset_builder is None or self._settings is None:
            raise ValueError("model is not built")

        return self._dataset_builder.build_dataset(
            dataset,
            self._settings,
            batch_size=batch_size,
            learnable_facts=learnable_facts,
            progress=progress,
        )

    def __call__(self, dataset=None):
        if isinstance(dataset, StaticGraphDataset):
            return self._test_static_graph(dataset)

        samples, _ = self._dataset_to_samples(dataset)
        sample_collection = samples if isinstance(samples, Collection) else [samples]

        # Invalidate each sample immediately before evaluating it, not all of them up front. Samples built
        # from one example share neurons, and evaluating one leaves its values on them - so an invalidation
        # that happened before the first sample ran does not clear what the first sample wrote, and every
        # later sample accumulates onto it. The backend's own loops (PythonTrainingStrategy) already pair
        # the two, which is why test() is unaffected.
        results = []
        for sample in sample_collection:
            self._trainer.invalidateSample(self._invalidation, sample._java_sample)
            results.append(
                self._value_factory.from_java(
                    self._trainer.evaluateSample(self._evaluation, sample._java_sample).getOutput(),
                )
            )

        if self._torch_module is None:
            return results

        return self._torch_module.forward(self, samples, results)

    def forward(self, dataset):
        return self(dataset)

    def train(self, dataset, epochs: int = 1) -> Value:
        """Trains the model on the provided dataset.

        Parameters
        ----------
        dataset : Any
            The dataset to train on. Can be a Dataset, GroundedDataset, BuiltDataset,
            StaticGraphDataset, or a list of samples.
        epochs : int
            The number of epochs to train. Default: 1.

        Returns
        -------
        Union[Tuple[Value, Value, Value], List[Tuple[Value, Value, Value]]]
            The training results (target, output, error).
        """
        if isinstance(dataset, StaticGraphDataset):
            return self._train_static_graph(dataset, epochs)

        samples, batch_size = self._dataset_to_samples(dataset)

        if not isinstance(samples, Collection):
            result = self._strategy.learnSample(samples._java_sample)
            res = (
                ValueFactory.from_java(result.getTarget()),
                ValueFactory.from_java(result.getOutput()),
                ValueFactory.from_java(result.errorValue()),
            )
        else:
            sample_array = jpype.java.util.ArrayList([sample._java_sample for sample in samples])
            results = self._strategy.learnSamples(sample_array, epochs, batch_size)
            res = [
                (
                    ValueFactory.from_java(result.getTarget()),
                    ValueFactory.from_java(result.getOutput()),
                    ValueFactory.from_java(result.errorValue()),
                )
                for result in results
            ]

        self._update_tensor_parameters()
        return res

    def test(self, dataset) -> Value:
        """Tests the model on the provided dataset.

        Parameters
        ----------
        dataset : Any
            The dataset to test on.

        Returns
        -------
        Union[Value, List[Value]]
            The test results (outputs).
        """
        if isinstance(dataset, StaticGraphDataset):
            return self._test_static_graph(dataset)

        samples, batch_size = self._dataset_to_samples(dataset)

        if not isinstance(samples, Collection):
            return ValueFactory.from_java(self._strategy.evaluateSample(samples._java_sample))

        sample_array = jpype.java.util.ArrayList([sample._java_sample for sample in samples])
        results = self._strategy.evaluateSamples(sample_array, batch_size)

        return [ValueFactory.from_java(result) for result in results]

    def loss(self, dataset) -> float:
        """The dataset's loss, reduced the way the error function says.

        This is the single number torch's criterion hands back, and the quantity the optimizer is descending:
        the per-query errors summed and then divided by the batch's total element count under
        ``reduction="mean"``, or left undivided under ``"sum"``.

        It is deliberately separate from :meth:`validate`, whose per-query values are *not* reduced across the
        batch - each is summed over its own components, which is torch's ``reduction="none"``. Both are
        useful, and conflating them is what let a reported number drift away from the one being minimised
        once before.

        Parameters
        ----------
        dataset : Any
            The dataset to take the loss of.

        Returns
        -------
        float
            The reduced loss.
        """
        samples, batch_size = self._dataset_to_samples(dataset)
        sample_list = samples if isinstance(samples, Collection) else [samples]
        sample_array = jpype.java.util.ArrayList([sample._java_sample for sample in sample_list])

        return float(ValueFactory.from_java(self._strategy.reducedError(sample_array, batch_size)))

    def validate(self, dataset) -> Value:
        """Evaluates the model on the provided dataset and reports the error, without training on it.

        Same shape of result as :meth:`train`, so a validation loss can be computed the same way, but nothing
        is backpropagated and the optimizer is not stepped.

        Parameters
        ----------
        dataset : Any
            The dataset to validate on.

        Returns
        -------
        Union[Tuple[Value, Value, Value], List[Tuple[Value, Value, Value]]]
            The validation results (target, output, error).
        """
        samples, batch_size = self._dataset_to_samples(dataset)

        if not isinstance(samples, Collection):
            result = self._strategy.validateSample(samples._java_sample)
            return (
                ValueFactory.from_java(result.getTarget()),
                ValueFactory.from_java(result.getOutput()),
                ValueFactory.from_java(result.errorValue()),
            )

        sample_array = jpype.java.util.ArrayList([sample._java_sample for sample in samples])
        results = self._strategy.validateSamples(sample_array, batch_size)

        return [
            (
                ValueFactory.from_java(result.getTarget()),
                ValueFactory.from_java(result.getOutput()),
                ValueFactory.from_java(result.errorValue()),
            )
            for result in results
        ]
    def _train_static_graph(self, dataset: StaticGraphDataset, epochs: int = 1) -> list:
        """Train on a StaticGraphDataset by iterating over fact mappings.

        For each sample's fact mapping, updates the shared neural sample's
        fact values via ``set_fact_value``, then calls ``learnSample``.

        Parameters
        ----------
        dataset : StaticGraphDataset
            The static graph dataset.
        epochs : int
            Number of epochs. Default: 1.

        Returns
        -------
        list
            List of (target, output, error) tuples for every sample in every epoch.
        """
        results = []
        static_sample = dataset.static_sample

        for _ in range(epochs):
            for index, mapping in enumerate(dataset.fact_mappings):
                for fact, value in mapping:
                    _, java_value = self._value_factory.get_value(value)
                    static_sample.set_fact_value(fact, java_value)
                # The shared sample carries the first sample's target, so this sample's has to be put in
                # place as well - otherwise every one of them is fitted to that first label.
                target = dataset.targets[index]
                if target is not None:
                    _, java_target = self._value_factory.get_value(target)
                    static_sample._java_sample.target = java_target
                result = self._strategy.learnSample(static_sample._java_sample)
                results.append((
                    ValueFactory.from_java(result.getTarget()),
                    ValueFactory.from_java(result.getOutput()),
                    ValueFactory.from_java(result.errorValue()),
                ))

        self._update_tensor_parameters()
        return results

    def _test_static_graph(self, dataset: StaticGraphDataset) -> list:
        """Evaluate on a StaticGraphDataset.

        For each sample's fact mapping, updates fact values on the shared
        neural sample, then evaluates without updating weights.

        Parameters
        ----------
        dataset : StaticGraphDataset
            The static graph dataset.

        Returns
        -------
        list
            Model outputs for every sample.
        """
        results = []
        static_sample = dataset.static_sample

        for mapping in dataset.fact_mappings:
            for fact, value in mapping:
                _, java_value = self._value_factory.get_value(value)
                static_sample.set_fact_value(fact, java_value)
            result = ValueFactory.from_java(
                self._strategy.evaluateSample(static_sample._java_sample)
            )
            results.append(result)

        return results

    def reset_parameters(self):
        self._strategy.resetParameters()

    def parameters(self) -> dict:
        """Returns the model parameters.

        Returns
        -------
        dict
            The model parameters.
        """
        return self.state_dict()

    def state_dict(self) -> dict:
        """Returns the state dictionary of the model.

        Returns
        -------
        dict
            The state dictionary (weights and weight names).
        """
        weights = self._neural_model.getAllWeights()
        weights_dict = {}
        weight_names = {}

        for weight in weights:
            if weight.isLearnable():
                weights_dict[weight.index] = ValueFactory.from_java(weight.value)
                weight_names[weight.index] = str(weight.name)
        return {
            "weights": weights_dict,
            "weight_names": weight_names,
        }

    def fixed_state_dict(self) -> dict:
        """Returns the weights training does not touch, in the same shape as :meth:`state_dict`.

        A fixed weight is a modelling choice rather than a parameter - an initial hidden state, say - so it is
        deliberately absent from :meth:`state_dict`, which describes what training changes. Reading one is
        still a reasonable thing to want, and without this the only way was through the private Java model.

        The library's own internal constants sit at negative indices, the logical `ONE` among them. Those are
        not modelling choices and are left out.

        Returns
        -------
        dict
            The fixed weights and their names.
        """
        weights_dict = {}
        weight_names = {}

        for weight in self._neural_model.getAllWeights():
            if not weight.isLearnable() and weight.index >= 0:
                weights_dict[weight.index] = ValueFactory.from_java(weight.value)
                weight_names[weight.index] = str(weight.name)
        return {
            "weights": weights_dict,
            "weight_names": weight_names,
        }

    def load_fixed_state_dict(self, state_dict: dict):
        """Sets the weights training does not touch, from a :meth:`fixed_state_dict` shaped dictionary.

        Setting one is initialization, not learning, so this is separate from :meth:`load_state_dict` - which
        skips fixed weights precisely because training must not move them.

        Parameters
        ----------
        state_dict : dict
            Fixed weights to set, keyed by weight index. Indices absent from the model are an error, and a
            learnable index is refused rather than quietly written.
        """
        by_index = {weight.index: weight for weight in self._neural_model.getAllWeights()}
        weight_dict = state_dict["weights"]

        for index, value in weight_dict.items():
            if index < 0:
                raise ValueError(f"weight {index} is one of the library's internal constants and is not settable")
            weight = by_index.get(index)
            if weight is None:
                raise ValueError(f"there is no weight with index {index} in this model")
            if weight.isLearnable():
                raise ValueError(f"weight {index} is learnable - use load_state_dict for those")
            self._set_weight_value(weight.value, value)

        if self._torch_module is not None:
            self._torch_module.update_tensor_parameters(self._tensor_parameters)

    @staticmethod
    def _set_weight_value(weight_value, value):
        if isinstance(value, (float, int)):
            weight_value.set(0, float(value))
            return
        if isinstance(value[0], (float, int)):
            for i, val in enumerate(value):
                weight_value.set(i, float(val))
            return

        cols = len(value[0])
        for i, values in enumerate(value):
            for j, val in enumerate(values):
                weight_value.set(i * cols + j, float(val))

    def tensor_parameters(self):
        if self._torch_module is None:
            raise NotImplementedError(
                "tensor_parameters() requires the PyTorch backend. Call model.build(settings, torch=True) to enable it."
            )

        self._tensor_parameters = self._torch_module.tensor_parameters(
            self._tensor_parameters,
            self._weight_updater,
            self._value_factory,
            self._neural_model,
        )

        return list(self._tensor_parameters)

    def _update_tensor_parameters(self):
        if self._torch_module is not None:
            self._torch_module.update_tensor_parameters(self._tensor_parameters)

    def load_state_dict(self, state_dict: dict):
        self._sync_model(state_dict, self._neural_model.getAllWeights())

        if self._torch_module is not None:
            self._torch_module.update_tensor_parameters(self._tensor_parameters)

    def draw(
        self,
        filename: str | None = None,
        show=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: str | None = None,
        *args,
        **kwargs,
    ):
        if self._dataset_builder is None or self._settings is None:
            raise ValueError("model is not built")
        return draw_model(self, filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)

    def _initialize_neural_module(self, dataset_builder: DatasetBuilder, settings: SettingsProxy, model, torch: bool):
        self._dataset_builder = dataset_builder
        self._settings = settings
        self._neural_model = model

        if torch:
            try:
                import torch
            except ImportError:
                raise ImportError("torch is not installed in the environment")

            from neuralogic.core.torch.neural_module import TorchNeuralModule

            self._torch_module = TorchNeuralModule()

        optimizer = self._settings.optimizer.initialize()
        lr_decay = self._settings.optimizer.get_lr_decay()

        python_strategy = jpype.JClass(
            "cz.cvut.fel.ida.neural.networks.computation.training.strategies.PythonTrainingStrategy"
        )

        self._strategy = python_strategy(settings.settings, model, optimizer, lr_decay)
        self._trainer = self._strategy.getTrainer()

        self._invalidation = self._trainer.getInvalidation()
        self._evaluation = self._trainer.getEvaluation()
        self._backpropagation = self._trainer.getBackpropagation()
        self._weight_updater = self._backpropagation.weightUpdater

        self.reset_parameters()

    def _dataset_to_samples(self, dataset):
        if isinstance(dataset, StaticGraphDataset):
            return dataset.static_sample, dataset._batch_size

        if isinstance(dataset, Dataset):
            dataset = self.build_dataset(dataset)
            return dataset._samples, dataset._batch_size

        if isinstance(dataset, GroundedDataset):
            dataset = dataset.neuralize()
            return dataset._samples, dataset._batch_size

        if isinstance(dataset, BuiltDataset):
            return dataset._samples, dataset._batch_size
        return dataset, 1

    def _sync_model(self, state_dict: dict | None = None, weights=None):
        state_dict = self.state_dict() if state_dict is None else state_dict
        weights = self._parsed_model.getAllWeights() if weights is None else weights
        weight_dict = state_dict["weights"]

        for weight in weights:
            if not weight.isLearnable():
                continue
            self._set_weight_value(weight.value, weight_dict[weight.index])

    def _backprop(self, sample, gradient):
        _, gradient_value = self._value_factory.get_value(gradient)

        weight_updater = self._backpropagation.backpropagate(sample._java_sample, gradient_value)
        state_index = self._backpropagation.backproper

        return state_index, weight_updater
