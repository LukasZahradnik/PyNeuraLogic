import json
from typing import Dict, Sized

import jpype

from neuralogic import is_initialized, initialize
from neuralogic.nn.base import AbstractNeuraLogic
from neuralogic.core.settings import SettingsProxy
from neuralogic.core.enums import Backend


class NeuraLogic(AbstractNeuraLogic):
    def __init__(self, model, dataset_builder, template, settings: SettingsProxy):
        super().__init__(Backend.JAVA, dataset_builder, template, settings)

        if not is_initialized():
            initialize()

        python_strategy = jpype.JClass(
            "cz.cvut.fel.ida.neural.networks.computation.training.strategies.PythonTrainingStrategy"
        )

        self.do_train = True
        self.need_sync = False

        self.neural_model = model
        self.strategy = python_strategy(settings.settings, model)
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

    def __call__(self, samples=None, train: bool = None, epochs: int = 1):
        self.hooks_set = len(self.hooks) != 0

        if self.hooks_set:
            self.strategy.setHooks(set(self.hooks.keys()), self.hook_handler)

        if train is not None:
            self.do_train = train

        if samples is None:
            results = self.strategy.learnSamples(epochs)
            deserialized_results = json.loads(str(results))

            return deserialized_results, self.samples_len

        if not isinstance(samples, Sized):
            if self.do_train:
                result = self.strategy.learnSample(samples.java_sample)
                return json.loads(str(result)), 1
            return json.loads(str(self.strategy.evaluateSample(samples.java_sample)))

        if self.do_train:
            results = self.strategy.learnSamples(
                jpype.java.util.ArrayList([sample.java_sample for sample in samples]), epochs
            )

            return json.loads(str(results)), len(samples)

        results = self.strategy.evaluateSamples(jpype.java.util.ArrayList([sample.java_sample for sample in samples]))
        return json.loads(str(results))

    def state_dict(self) -> Dict:
        weights = self.neural_model.getAllWeights()
        weights_dict = {}
        weight_names = {}

        for weight in weights:
            if weight.isLearnable:
                value = weight.value

                size = list(value.size())

                if len(size) == 0 or size[0] == 0:
                    weights_dict[weight.index] = value.get(0)
                elif len(size) == 1 or size[0] == 1 or size[1] == 1:
                    weights_dict[weight.index] = list(value.values)
                else:
                    weights_dict[weight.index] = [list(value) for value in value.values]
                weight_names[weight.index] = weight.name
        return {
            "weights": weights_dict,
            "weight_names": weight_names,
        }

    def load_state_dict(self, state_dict: Dict):
        self.sync_template(state_dict, self.neural_model.getAllWeights())
