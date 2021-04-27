from collections import Sized

from neuralogic import get_neuralogic
from py4j.java_gateway import get_field

from neuralogic.core.model import Model


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
    def __init__(self, model: Model):
        self.namespace = get_neuralogic().cz.cvut.fel.ida.neural.networks.computation.training.strategies
        self.do_train = True
        self.settings = model.settings

        self.neural_model = model.model
        self.strategy = self.namespace.PythonTrainingStrategy(model.settings.settings, model.model)

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
            results = self.strategy.learnSamples(epochs)
            return sum(get_field(result.errorValue(), "value") for result in results), self.samples_len

        if not isinstance(samples, Sized):
            if self.do_train:
                if auto_backprop:
                    result = self.strategy.learnSample(samples)
                    return get_field(result.errorValue(), "value"), 1
            result = self.strategy.evaluateSample(samples)
            return Loss(result)

        if self.do_train:
            results = self.strategy.learnSamples(samples, epochs)
            return sum(get_field(result.errorValue(), "value") for result in results), len(samples)

        results = self.strategy.evaluateSamples(samples)
        return [(get_field(result.getTarget(), "value"), get_field(result.getOutput(), "value")) for result in results]
