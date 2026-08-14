from typing import Any

import jpype

from neuralogic.nn.optim.lr_scheduler import LRDecay
from neuralogic.nn.optim.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer.
    It implements the Adaptive Moment Estimation (Adam) algorithm.
    """

    def __init__(
        self,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        lr_decay: LRDecay | None = None,
        weight_decay: float = 0.0,
    ):
        """
        Parameters
        ----------
        lr : float, optional
            The learning rate. Default: 0.001.
        betas : Tuple[float, float], optional
            Coefficients used for computing running averages of gradient and its square. Default: (0.9, 0.999).
        eps : float, optional
            Term added to the denominator to improve numerical stability. Default: 1e-08.
        lr_decay : LRDecay, optional
            Learning rate decay scheduler. Default: None.
        weight_decay : float, optional
            L2 penalty, exactly as :code:`torch.optim.Adam(weight_decay=)`: the term joins the gradient
            *before* the moments and so accumulates in them. This is the coupled variant - it is not
            :code:`AdamW`, whose decay bypasses the adaptive scaling and gives a different update from the
            same numbers. Default: 0.0.
        """
        super().__init__(lr, lr_decay, weight_decay)
        self._betas = betas
        self._eps = eps

    @property
    def betas(self) -> tuple[float, float]:
        return self._betas

    @property
    def eps(self) -> float:
        return self._eps

    def initialize(self) -> Any:
        """
        Initializes the Java representation of the Adam optimizer.

        Returns
        -------
        Any
            The Java optimizer object.
        """
        if self._optimizer:
            return self._optimizer

        adam_class = jpype.JClass("cz.cvut.fel.ida.neural.networks.computation.training.optimizers.Adam")
        self._lr_object = jpype.JClass("cz.cvut.fel.ida.algebra.values.ScalarValue")(self._lr)
        self._optimizer = adam_class(
            self._lr_object, self._betas[0], self._betas[1], self._eps, self._weight_decay
        )

        return self._optimizer

    def __str__(self) -> str:
        return (
            f"Adam(lr={self.lr}, betas={self.betas}, eps={self.eps}, "
            f"weight_decay={self.weight_decay}, lr_decay={self._lr_decay})"
        )
