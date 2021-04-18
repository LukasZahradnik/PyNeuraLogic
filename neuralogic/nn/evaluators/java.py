from typing import Optional

from neuralogic.builder import Backend
from neuralogic.data import Dataset
from neuralogic.model import Model
from neuralogic.nn.java import NeuraLogicLayer
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.settings import Settings


class JavaEvaluator(AbstractEvaluator):
    def __init__(self, model: Optional[Model], dataset: Optional[Dataset], settings: Settings):
        super().__init__(model, dataset, settings)

        if model is not None:
            self.dataset = model.build(Backend.JAVA)
        if dataset is not None:
            self.dataset = dataset
        self.neuralogic_layer = NeuraLogicLayer(self.dataset)

    def train(self, generator: bool = True):
        epochs = self.settings.epochs

        def _train():
            for _ in range(epochs):
                result = self.neuralogic_layer(self.dataset.samples, True)
                yield result

        if generator:
            return _train()

        stats = 0, 0
        for stats in _train():
            pass
        return stats

    def test(self, generator: bool = True):
        def _test():
            for sample in self.dataset.samples:
                result = self.neuralogic_layer(sample, False)
                yield result

        if generator:
            return _test()
        return list(_test())
