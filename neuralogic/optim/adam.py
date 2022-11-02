from typing import Tuple, Optional

import jpype

from neuralogic.optim.lr_scheduler import LRDecay
from neuralogic.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        lr_decay: Optional[LRDecay] = None,
    ):
        super().__init__(lr, lr_decay)
        self._betas = betas
        self._eps = eps

    @property
    def betas(self) -> Tuple[float, float]:
        return self._betas

    @property
    def eps(self) -> float:
        return self._eps

    def initialize(self):
        if self._optimizer:
            return self._optimizer

        adam_class = jpype.JClass("cz.cvut.fel.ida.neural.networks.computation.training.optimizers.Adam")
        self._lr_object = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")(self._lr)
        self._optimizer = adam_class(self._lr_object, self._betas[0], self._betas[1], self._eps)

        return self._optimizer

    def __str__(self) -> str:
        return f"Adam(lr={self.lr}, betas={self.betas}, eps={self.eps}, lr_decay={self._lr_decay})"
