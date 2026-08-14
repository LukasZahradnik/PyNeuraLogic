from typing import Any

from neuralogic.nn.optim.lr_scheduler import LRDecay


class Optimizer:
    """
    Base class for all optimizers.
    Optimizers are used to update the weights of the neural network during training.
    """

    def __init__(self, lr: float, lr_decay: LRDecay | None = None, weight_decay: float = 0.0):
        """
        Parameters
        ----------
        lr : float
            Initial learning rate.
        lr_decay : LRDecay, optional
            Learning rate decay scheduler. Default: None.
        weight_decay : float, optional
            L2 penalty, as in torch. Default: 0.0.
        """
        if lr_decay is not None:
            lr_decay._optimizer = self

        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

        self._lr_decay = lr_decay
        self._lr = lr
        self._weight_decay = weight_decay

        self._optimizer = None
        self._lr_object = None

    @property
    def weight_decay(self) -> float:
        """The L2 penalty added to every gradient before the step."""
        return self._weight_decay

    @property
    def lr(self) -> float:
        """
        Returns the current learning rate.

        Returns
        -------
        float
            Current learning rate.
        """
        if self._lr_object is None:
            return self._lr
        return self._lr_object.value

    @lr.setter
    def lr(self, value: float) -> None:
        """
        Sets the learning rate.

        Parameters
        ----------
        value : float
            New learning rate value.
        """
        if self._lr_object is not None:
            self._lr_object.value = value
        self._lr = value

    def initialize(self) -> Any:
        """
        Initializes the Java representation of the optimizer.

        Returns
        -------
        Any
            The Java optimizer object.
        """
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__

    def get_lr_decay(self) -> Any | None:
        """
        Initializes and returns the learning rate decay object.

        Returns
        -------
        Any
            The Java learning rate decay object, or None if no decay is set.
        """
        if self._lr_decay is None:
            return None
        return self._lr_decay._initialize(self._lr_object)
