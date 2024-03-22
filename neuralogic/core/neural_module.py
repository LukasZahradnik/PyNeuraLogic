from typing import Optional, Union, Callable, Dict, Any, Set, Collection
import json

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core.constructs.java_objects import ValueFactory
from neuralogic.core.constructs.relation import BaseRelation
from neuralogic.core.builder import DatasetBuilder
from neuralogic.core.builder.components import BuiltDataset, GroundedDataset
from neuralogic.core.result import Results, Result
from neuralogic.core.settings.settings_proxy import SettingsProxy
from neuralogic.dataset import Dataset
from neuralogic.dataset.base import BaseDataset

from neuralogic.utils.visualize import draw_model


class NeuralModule:
    def __init__(self):
        if not is_initialized():
            initialize()

        self.need_sync = False

        self.number_format = jpype.JClass("cz.cvut.fel.ida.setup.Settings").superDetailedNumberFormat
        self.value_factory = ValueFactory()

        @jpype.JImplements(
            jpype.JClass("cz.cvut.fel.ida.neural.networks.computation.iteration.actions.PythonHookHandler")
        )
        class HookHandler:
            def __init__(self, module: "NeuralModule"):
                self.module = module

            @jpype.JOverride
            def handleHook(self, hook, value):
                self.module._run_hook(hook, json.loads(value))

        self.hooks: Dict[str, Set[Callable]] = {}
        self.hooks_set = False

        self.hook_handler = HookHandler(self)

        self.parsed_template = None
        self.dataset_builder: Optional[DatasetBuilder] = None
        self.settings: Optional[SettingsProxy] = None

        self.neural_model = None
        self.strategy = None
        self.trainer = None

        self.invalidation = None
        self.evaluation = None
        self.backpropagation = None

    def _initialize_neural_module(self, dataset_builder: DatasetBuilder, settings: SettingsProxy, model):
        self.parsed_template = dataset_builder.parsed_template
        self.dataset_builder = dataset_builder
        self.settings = settings
        self.neural_model = model

        optimizer = self.settings.optimizer.initialize()
        lr_decay = self.settings.optimizer.get_lr_decay()

        python_strategy = jpype.JClass(
            "cz.cvut.fel.ida.neural.networks.computation.training.strategies.PythonTrainingStrategy"
        )

        self.strategy = python_strategy(settings.settings, model, optimizer, lr_decay)
        self.trainer = self.strategy.getTrainer()

        self.invalidation = self.trainer.getInvalidation()
        self.evaluation = self.trainer.getEvaluation()
        self.backpropagation = self.trainer.getBackpropagation()

        self.reset_parameters()

    def ground(
        self,
        dataset: BaseDataset,
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
    ) -> GroundedDataset:
        return self.dataset_builder.ground_dataset(
            dataset,
            self.settings,
            batch_size=batch_size,
            learnable_facts=learnable_facts,
        )

    def build_dataset(
        self,
        dataset: Union[BaseDataset, GroundedDataset],
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
        progress: bool = False,
    ) -> BuiltDataset:
        return self.dataset_builder.build_dataset(
            dataset,
            self.settings,
            batch_size=batch_size,
            learnable_facts=learnable_facts,
            progress=progress,
        )

    def train(self, dataset, epochs: int = 1):
        return self._train_test(dataset, True, epochs)

    def test(self, dataset, epochs: int = 1):
        return self._train_test(dataset, False, epochs)

    def _train_test(self, dataset, train: bool, epochs: int = 1):
        self.hooks_set = len(self.hooks) != 0
        samples, batch_size = self._dataset_to_samples(dataset)

        if self.hooks_set:
            self.strategy.setHooks(set(self.hooks.keys()), self.hook_handler)

        if not isinstance(samples, Collection):
            if train:
                result = self.strategy.learnSample(samples.java_sample)
                return json.loads(str(result)), 1
            return json.loads(str(self.strategy.evaluateSample(samples.java_sample)))

        sample_array = jpype.java.util.ArrayList([sample.java_sample for sample in samples])

        if train:
            results = self.strategy.learnSamples(sample_array, epochs, batch_size)

            return json.loads(str(results)), len(samples)

        results = self.strategy.evaluateSamples(sample_array, batch_size)
        return json.loads(str(results))

    def __call__(self, dataset=None) -> Union[Results, Result]:
        samples, batch_size = self._dataset_to_samples(dataset)

        if isinstance(samples, Collection):
            results = []

            for sample in samples:
                self.trainer.invalidateSample(self.invalidation, sample.java_sample)

                results.append(
                    Result(
                        self.trainer.evaluateSample(self.evaluation, sample.java_sample),
                        sample.java_sample,
                        self,
                        self.number_format,
                    )
                )
            return Results(results)

        sample = samples
        self.trainer.invalidateSample(self.invalidation, sample.java_sample)
        result = self.trainer.evaluateSample(self.evaluation, sample.java_sample)

        return Result(result, sample.java_sample, self, self.number_format)

    def forward(self, dataset) -> Union[Results, Result]:
        return self(dataset)

    def backprop(self, sample, gradient):
        trainer = self.strategy.getTrainer()
        _, gradient_value = self.value_factory.get_value(gradient)

        backpropagation = trainer.getBackpropagation()
        weight_updater = backpropagation.backpropagate(sample.java_sample, gradient_value)
        state_index = backpropagation.backproper

        return state_index, weight_updater

    def reset_parameters(self):
        self.strategy.resetParameters()

    def parameters(self) -> Dict:
        return self.state_dict()

    def state_dict(self) -> Dict:
        weights = self.neural_model.getAllWeights()
        weights_dict = {}
        weight_names = {}

        for weight in weights:
            if weight.isLearnable:
                weights_dict[weight.index] = ValueFactory.from_java(weight.value)
                weight_names[weight.index] = weight.name
        return {
            "weights": weights_dict,
            "weight_names": weight_names,
        }

    def load_state_dict(self, state_dict: Dict):
        self.sync_template(state_dict, self.neural_model.getAllWeights())

    def draw(
        self,
        filename: Optional[str] = None,
        show=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        return draw_model(self, filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)

    def set_hooks(self, hooks):
        self.hooks_set = len(hooks) != 0
        self.hooks = hooks

    def add_hook(self, relation: Union[BaseRelation, str], callback: Callable[[Any], None]) -> None:
        """Hooks the callable to be called with the relation's value as an argument when the value of
        the relation is being calculated.

        :param relation:
        :param callback:
        :return:
        """
        name = str(relation)

        if isinstance(relation, BaseRelation):
            name = name[:-1]

        if name not in self.hooks:
            self.hooks[name] = {callback}
        else:
            self.hooks[name].add(callback)

    def remove_hook(self, relation: Union[BaseRelation, str], callback):
        """Removes the callable from the relation's hooks

        :param relation:
        :param callback:
        :return:
        """
        name = str(relation)

        if isinstance(relation, BaseRelation):
            name = name[:-1]

        if name not in self.hooks:
            return
        self.hooks[name].discard(callback)

    def _run_hook(self, hook: str, value):
        for callback in self.hooks[hook]:
            callback(value)

    def _dataset_to_samples(self, dataset):
        if isinstance(dataset, Dataset):
            dataset = self.build_dataset(dataset)
            return dataset.samples, dataset.batch_size

        if isinstance(dataset, BuiltDataset):
            return dataset.samples, dataset.batch_size
        return dataset, 1

    def sync_template(self, state_dict: Optional[Dict] = None, weights=None):
        state_dict = self.state_dict() if state_dict is None else state_dict
        weights = self.parsed_template.getAllWeights() if weights is None else weights
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
