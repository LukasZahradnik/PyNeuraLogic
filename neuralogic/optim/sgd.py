import jpype

from neuralogic.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr: float = 0.1):
        self._lr = lr

    @property
    def lr(self) -> float:
        return self._lr

    def get(self):
        sgd_class = jpype.JClass("cz.cvut.fel.ida.neural.networks.computation.training.optimizers.SGD")
        lr = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")(self._lr)

        return sgd_class(lr)

    def __str__(self) -> str:
        return f"SGD({self.lr=})".replace("self.", "")
