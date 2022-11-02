from typing import Optional

from neuralogic.optim.lr_scheduler import LRDecay


class Optimizer:
    def __init__(self, lr: float, lr_decay: Optional[LRDecay] = None):
        if lr_decay is not None:
            lr_decay._optimizer = self

        self._lr_decay = lr_decay
        self._lr = lr

        self._optimizer = None
        self._lr_object = None

    @property
    def lr(self) -> float:
        if self._lr_object is None:
            return self._lr
        return self._lr_object.value

    @lr.setter
    def lr(self, value: float):
        if self._lr_object is not None:
            self._lr_object.value = value
        self._lr = value

    def initialize(self):
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__

    def get_lr_decay(self):
        if self._lr_decay is None:
            return None
        return self._lr_decay._initialize(self._lr_object)
