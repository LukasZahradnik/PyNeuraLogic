import json
from typing import Dict, Sized

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.core import BuiltDataset
from neuralogic.core.constructs.java_objects import ValueFactory
from neuralogic.nn.base import AbstractNeuraLogic
from neuralogic.core.settings import SettingsProxy


class NeuraLogic(AbstractNeuraLogic):
    def __init__(self, model, dataset_builder, template, settings: SettingsProxy):
        super().__init__(dataset_builder, template, settings)

        if not is_initialized():
            initialize()

        python_strategy = jpype.JClass(
            "cz.cvut.fel.ida.neural.networks.computation.training.strategies.PythonTrainingStrategy"
        )

        self.do_train = True
        self.need_sync = False

        self.value_factory = ValueFactory()

        optimizer = self.settings.optimizer.initialize()
        lr_decay = self.settings.optimizer.get_lr_decay()

        self.neural_model = model
        self.strategy = python_strategy(settings.settings, model, optimizer, lr_decay)

        self.samples_len = 0
        self.number_format = self.settings.settings_class.superDetailedNumberFormat

        @jpype.JImplements(
            jpype.JClass("cz.cvut.fel.ida.neural.networks.computation.iteration.actions.PythonHookHandler")
        )
        class HookHandler:
            def __init__(self, module: "NeuraLogic"):
                self.module = module

            @jpype.JOverride
            def handleHook(self, hook, value):
                self.module.run_hook(hook, json.loads(value))

        self.hook_handler = HookHandler(self)
        self.reset_parameters()

    def reset_parameters(self):
        self.strategy.resetParameters()

    def train(self):
        self.do_train = True

    def test(self):
        self.do_train = False

    def set_training_samples(self, samples):
        self.samples_len = len(samples)
        self.strategy.setSamples(jpype.java.util.ArrayList(samples))

    def __call__(self, dataset=None, train: bool = None, epochs: int = 1):
        self.hooks_set = len(self.hooks) != 0

        if isinstance(dataset, BuiltDataset):
            samples = dataset.samples
            batch_size = dataset.batch_size
        else:
            samples = dataset
            batch_size = 1

        if self.hooks_set:
            self.strategy.setHooks(set(self.hooks.keys()), self.hook_handler)

        if train is not None:
            self.do_train = train

        if samples is None:
            results = self.strategy.learnSamples(epochs, batch_size)
            deserialized_results = json.loads(str(results))

            return deserialized_results, self.samples_len

        if not isinstance(samples, Sized):
            if self.do_train:
                result = self.strategy.learnSample(samples.java_sample)
                return json.loads(str(result)), 1
            return json.loads(str(self.strategy.evaluateSample(samples.java_sample)))

        sample_array = jpype.java.util.ArrayList([sample.java_sample for sample in samples])

        if self.do_train:
            results = self.strategy.learnSamples(sample_array, epochs, batch_size)

            return json.loads(str(results)), len(samples)

        results = self.strategy.evaluateSamples(sample_array, batch_size)
        return json.loads(str(results))

    def backprop(self, sample, gradient):
        trainer = self.strategy.getTrainer()
        _, gradient_value = self.value_factory.get_value(gradient)

        backpropagation = trainer.getBackpropagation()
        weight_updater = backpropagation.backpropagate(sample.java_sample, gradient_value)
        state_index = backpropagation.backproper

        return state_index, weight_updater

    def state_dict(self) -> Dict:
        weights = self.neural_model.getAllWeights()
        weights_dict = {}
        weight_names = {}

        for weight in weights:
            if weight.isLearnable:
                weights_dict[weight.index] = ValueFactory.from_java(weight.value, SettingsProxy.number_format())
                weight_names[weight.index] = str(weight.name)
        return {
            "weights": weights_dict,
            "weight_names": weight_names,
        }

    def load_state_dict(self, state_dict: Dict):
        self.sync_template(state_dict, self.neural_model.getAllWeights())
