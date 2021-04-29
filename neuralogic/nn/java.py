import json
from collections import Sized
from typing import Optional, Dict

from neuralogic import get_neuralogic
from py4j.java_gateway import get_field

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


class NeuraLogicLayer:
    def __init__(self, model, settings: Optional[Settings] = None):
        self.namespace = get_neuralogic().cz.cvut.fel.ida.neural.networks.computation.training.strategies
        self.do_train = True

        if settings is None:
            settings = Settings()

        self.settings = settings
        self.neural_model = model
        self.strategy = self.namespace.PythonTrainingStrategy(settings.settings, model)
        self.samples_len = 0
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
        weights = self.neural_model.getAllWeights()
        weight_dict = state_dict["weights"]

        for weight in weights:
            if not get_field(weight, "isLearnable"):
                continue
            weight_value = get_field(weight, "value")

            index = get_field(weight, "index")
            value = weight_dict[index]

            if isinstance(value, (float, int)):
                weight_value.set(0, float(value))
                continue

            if isinstance(value[0], (float, int)):
                for i, val in enumerate(value):
                    weight_value.set(i, float(val))
                continue

            rows = len(value)

            for i, values in enumerate(value):
                for j, val in enumerate(values):
                    weight_value.set(i * rows + j, float(val))
