from __future__ import annotations

from typing import TYPE_CHECKING, Collection

import jpype

from neuralogic.core.builder.dataset import BuiltDataset, GroundedDataset

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
        samples, _ = self._dataset_to_samples(dataset)
        sample_collection = samples if isinstance(samples, Collection) else [samples]

        for sample in sample_collection:
            self._trainer.invalidateSample(self._invalidation, sample._java_sample)

        results = [
            self._value_factory.from_java(
                self._trainer.evaluateSample(self._evaluation, sample._java_sample).getOutput(),
            )
            for sample in sample_collection
        ]

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
            The dataset to train on. Can be a Dataset, GroundedDataset, BuiltDataset, or a list of samples.
        epochs : int
            The number of epochs to train. Default: 1.

        Returns
        -------
        Union[Tuple[Value, Value, Value], List[Tuple[Value, Value, Value]]]
            The training results (target, output, error).
        """
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
        samples, batch_size = self._dataset_to_samples(dataset)

        if not isinstance(samples, Collection):
            return ValueFactory.from_java(self._strategy.evaluateSample(samples._java_sample))

        sample_array = jpype.java.util.ArrayList([sample._java_sample for sample in samples])
        results = self._strategy.evaluateSamples(sample_array, batch_size)

        return [ValueFactory.from_java(result) for result in results]

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
            if weight.isLearnable:
                weights_dict[weight.index] = ValueFactory.from_java(weight.value)
                weight_names[weight.index] = str(weight.name)
        return {
            "weights": weights_dict,
            "weight_names": weight_names,
        }

    def tensor_parameters(self):
        if self._torch_module is None:
            raise NotImplementedError(
                "tensor_parameters() requires the PyTorch backend. "
                "Call model.build(settings, torch=True) to enable it."
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
            if not weight.isLearnable:
                continue
            weight_value = weight.value

            index = weight.index
            value = weight_dict[index]

            if isinstance(value, (float, int)):
                weight_value.set(0, float(value))
                continue

            if isinstance(value[0], (float, int)):
                for i, val in enumerate(value):
                    weight_value.set(i, float(val))
                continue

            cols = len(value[0])

            for i, values in enumerate(value):
                for j, val in enumerate(values):
                    weight_value.set(i * cols + j, float(val))

    def _backprop(self, sample, gradient):
        _, gradient_value = self._value_factory.get_value(gradient)

        weight_updater = self._backpropagation.backpropagate(sample._java_sample, gradient_value)
        state_index = self._backpropagation.backproper

        return state_index, weight_updater
