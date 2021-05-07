from typing import Optional, Dict

from neuralogic.core.builder import Backend
from neuralogic.core import Problem
from neuralogic.nn.base import AbstractEvaluator
from neuralogic.core.settings import Settings


class JavaEvaluator(AbstractEvaluator):
    def __init__(
        self,
        problem: Optional[Problem],
        settings: Settings,
    ):
        super().__init__(Backend.JAVA, problem, settings)
        self.neuralogic_model.set_training_samples(self.dataset.samples)

    def train(self, generator: bool = True, epochs: int = None):
        if epochs is None:
            epochs = self.settings.epochs

        def _train():
            for _ in range(epochs):
                result = self.neuralogic_model(None, True)
                yield result

        if generator:
            return _train()

        return self.neuralogic_model(None, True, epochs=epochs)

    def test(self, generator: bool = True):
        def _test():
            for sample in self.dataset.samples:
                result = self.neuralogic_model(sample, False)
                yield result.target(), result.output()

        if generator:
            return _test()
        return self.neuralogic_model(self.dataset.samples, False)

    def state_dict(self) -> Dict:
        return self.neuralogic_model.state_dict()

    def load_state_dict(self, state_dict: Dict):
        self.neuralogic_model.load_state_dict(state_dict)
