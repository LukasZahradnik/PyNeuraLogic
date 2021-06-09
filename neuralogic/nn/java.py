import json
from collections import Sized
from typing import Optional, Dict
from py4j.java_gateway import get_field
from py4j.java_collections import SetConverter

from neuralogic import get_neuralogic, get_gateway
from neuralogic.nn.base import AbstractNeuraLogic
from neuralogic.core.settings import Settings


class Loss:
    def __init__(self, loss):
        self.loss = loss

    def backward(self):
        self.loss.backward()

    def value(self) -> float:
        return get_field(self.loss.getError(), "value")

    def output(self) -> float:
        return get_field(self.loss.getOutput(), "value")

    def target(self) -> float:
        return get_field(self.loss.getTarget(), "value")


class NeuraLogic(AbstractNeuraLogic):
    class HookHandler:
        def __init__(self, module: "NeuraLogic"):
            self.module = module

        def handleHook(self, hook, value):
            self.module.run_hook(hook, json.loads(value))

        class Java:
            implements = ["cz.cvut.fel.ida.neural.networks.computation.iteration.actions.PythonHookHandler"]

    def __init__(self, model, template, settings: Optional[Settings] = None):
        super().__init__(template)
        self.namespace = get_neuralogic().cz.cvut.fel.ida.neural.networks.computation.training.strategies
        self.do_train = True
        self.need_sync = False

        if settings is None:
            settings = Settings()

        self.settings = settings
        self.neural_model = model
        self.strategy = self.namespace.PythonTrainingStrategy(settings.settings, model)
        self.samples_len = 0

        self.hook_handler = NeuraLogic.HookHandler(self)
        self.reset_parameters()

    def reset_parameters(self):
        self.strategy.resetParameters()

    def train(self):
        self.do_train = True

    def test(self):
        self.do_train = False

    def set_training_samples(self, samples):
        self.samples_len = len(samples)
        self.strategy.setSamples(samples)

    def __call__(self, samples=None, train: bool = None, auto_backprop: bool = False, epochs: int = 1):
        self.hooks_set = len(self.hooks) != 0

        if self.hooks_set:
            self.strategy.setHooks(
                SetConverter().convert(set(self.hooks.keys()), get_gateway()._gateway_client), self.hook_handler
            )

        if train is not None:
            self.do_train = train

        if samples is None:
            return self.strategy.learnSamples(epochs), self.samples_len

        if not isinstance(samples, Sized):
            if self.do_train:
                if auto_backprop:
                    return self.strategy.learnSample(samples), 1
            result = self.strategy.evaluateSample(samples)
            return Loss(result)

        if self.do_train:
            return self.strategy.learnSamples(samples, epochs), len(samples)

        results = self.strategy.evaluateSamples(samples)
        return [(get_field(result.getTarget(), "value"), get_field(result.getOutput(), "value")) for result in results]

    def state_dict(self) -> Dict:
        weights = self.neural_model.getAllWeights()
        weights_dict = {}

        for weight in weights:
            if get_field(weight, "isLearnable"):
                value = get_field(weight, "value")

                size = list(value.size())

                if len(size) == 0 or size[0] == 0:
                    weights_dict[get_field(weight, "index")] = get_field(value, "value")
                elif len(size) == 1 or size[0] == 1 or size[1] == 1:
                    weights_dict[get_field(weight, "index")] = list(get_field(value, "values"))
                else:
                    weights_dict[get_field(weight, "index")] = json.loads(value.toString())
        return {
            "weights": weights_dict,
        }

    def load_state_dict(self, state_dict: Dict):
        self.sync_template(state_dict, self.neural_model.getAllWeights())
