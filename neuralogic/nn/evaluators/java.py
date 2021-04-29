from typing import Optional, Dict

from neuralogic.core.builder import Backend
from neuralogic.core import Problem
from neuralogic.nn.java import NeuraLogicLayer
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.core.settings import Settings
from neuralogic.utils.data import Dataset


class JavaEvaluator(AbstractEvaluator):
    def __init__(
        self,
        problem: Optional[Problem],
        model: Optional[NeuraLogicLayer],
        dataset: Optional[Dataset],
        settings: Settings,
    ):
        super().__init__(problem, model, dataset, settings)

        if problem is not None:
            model, dataset = problem.build(Backend.JAVA)
            self.dataset = dataset

        self.neuralogic_layer: NeuraLogicLayer = model
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

    def state_dict(self) -> Dict:
        return self.neuralogic_layer.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.neuralogic_layer.load_state_dict(state_dict)
