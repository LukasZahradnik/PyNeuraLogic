from typing import Optional

import jpype

from neuralogic.optim.lr_scheduler import LRDecay
from neuralogic.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr: float = 0.1, lr_decay: Optional[LRDecay] = None):
        super().__init__(lr, lr_decay)

    def initialize(self):
        if self._optimizer:
            return self._optimizer

        sgd_class = jpype.JClass("cz.cvut.fel.ida.neural.networks.computation.training.optimizers.SGD")
        self._lr_object = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")(self._lr)
        self._optimizer = sgd_class(self._lr_object)

        return self._optimizer

    def __str__(self) -> str:
        return f"SGD(lr={self.lr}, lr_decay={self._lr_decay})"
