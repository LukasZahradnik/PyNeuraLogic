from typing import Tuple

import jpype

from neuralogic.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-08):
        self._lr = lr
        self._betas = betas
        self._eps = eps

    @property
    def lr(self) -> float:
        return self._lr

    @property
    def betas(self) -> Tuple[float, float]:
        return self._betas

    @property
    def eps(self) -> float:
        return self._eps

    def get(self):
        adam_class = jpype.JClass("cz.cvut.fel.ida.neural.networks.computation.training.optimizers.Adam")
        lr = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")(self._lr)

        return adam_class(lr, self._betas[0], self._betas[1], self._eps)

    def is_default(self) -> bool:
        return self._betas == (0.9, 0.999) and self._eps == 1e-08

    def __str__(self) -> str:
        return f"Adam(lr={self.lr}, betas={self.betas}, eps={self.eps})"
