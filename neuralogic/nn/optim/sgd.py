from typing import Any

import jpype

from neuralogic.nn.optim.lr_scheduler import LRDecay
from neuralogic.nn.optim.optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """

    def __init__(self, lr: float = 0.1, lr_decay: LRDecay | None = None, weight_decay: float = 0.0):
        """
        Parameters
        ----------
        lr : float, optional
            The learning rate. Default: 0.1.
        lr_decay : LRDecay, optional
            Learning rate decay scheduler. Default: None.
        weight_decay : float, optional
            L2 penalty, exactly as :code:`torch.optim.SGD(weight_decay=)`: the term is added to the gradient
            and so is scaled by the learning rate along with it, giving
            :code:`w -= lr * (grad + weight_decay * w)`. Default: 0.0.
        """
        super().__init__(lr, lr_decay, weight_decay)

    def initialize(self) -> Any:
        """
        Initializes the Java representation of the SGD optimizer.

        Returns
        -------
        Any
            The Java optimizer object.
        """
        if self._optimizer:
            return self._optimizer

        sgd_class = jpype.JClass("cz.cvut.fel.ida.neural.networks.computation.training.optimizers.SGD")
        self._lr_object = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")(self._lr)
        self._optimizer = sgd_class(self._lr_object, self._weight_decay)

        return self._optimizer

    def __str__(self) -> str:
        return f"SGD(lr={self.lr}, weight_decay={self.weight_decay}, lr_decay={self._lr_decay})"
