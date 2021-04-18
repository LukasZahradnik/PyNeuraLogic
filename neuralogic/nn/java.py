from collections import Iterable

from neuralogic import get_neuralogic, get_gateway
from py4j.java_gateway import get_field
from py4j.java_collections import ListConverter

from neuralogic.data import Dataset


class NeuraLogicLayer:
    def __init__(self, dataset: Dataset):
        self.namespace = get_neuralogic().cz.cvut.fel.ida.neural.networks.computation.training.strategies
        self.do_train = True
        self.settings = dataset.settings

        self.neural_model = dataset.neural_model
        self.strategy = self.namespace.IterativeTrainingStrategy(
            dataset.settings.settings, dataset.neural_model, ListConverter().convert([], get_gateway()._gateway_client)
        )
        self.trainer = get_field(self.strategy, "trainer")
        self.reset_parameters()

    def reset_parameters(self):
        self.trainer.restart(self.settings.settings)
        self.neural_model.resetWeights(get_field(self.strategy, "valueInitializer"))

    def train(self):
        self.do_train = True

    def test(self):
        self.do_train = False

    def __call__(self, samples, train: bool = None):
        if train is not None:
            self.do_train = train

        iterable = True

        if not isinstance(samples, Iterable):
            iterable = False
            samples = ListConverter().convert([samples], get_gateway()._gateway_client)

        if self.do_train:
            results = self.trainer.learnEpoch(self.neural_model, samples)
            return sum(get_field(result.errorValue(), "value") for result in results), len(samples)

        results = self.trainer.evaluate(samples)
        if iterable is False:
            return get_field(results[0].getTarget(), "value"), get_field(results[0].getOutput(), "value")
        return [(get_field(result.getTarget(), "value"), get_field(result.getOutput(), "value")) for result in results]
