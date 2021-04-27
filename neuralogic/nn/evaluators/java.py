from typing import Optional

from neuralogic.core.builder import Backend
from neuralogic.core import Problem
from neuralogic.core.model import Model
from neuralogic.nn.java import NeuraLogicLayer
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.core.settings import Settings
from neuralogic.utils.data import Dataset


class JavaEvaluator(AbstractEvaluator):
    def __init__(
        self, problem: Optional[Problem], model: Optional[Model], dataset: Optional[Dataset], settings: Settings
    ):
        super().__init__(problem, model, dataset, settings)

        if problem is not None:
            model, dataset = problem.build(Backend.JAVA)

            self.dataset = dataset
            self.model = model

        self.neuralogic_layer = NeuraLogicLayer(self.model)
        self.neuralogic_layer.set_training_samples(self.dataset.samples)

    def train(self, generator: bool = True, epochs: int = None):
        if epochs is None:
            epochs = self.settings.epochs

        def _train():
            for _ in range(epochs):
                result = self.neuralogic_layer(None, True)
                yield result

        if generator:
            return _train()

        return self.neuralogic_layer(None, True, epochs=epochs)

    def test(self, generator: bool = True):
        def _test():
            for sample in self.dataset.samples:
                result = self.neuralogic_layer(sample, False)
                yield result.target(), result.output()

        if generator:
            return _test()
        return self.neuralogic_layer(self.dataset.samples, False)
