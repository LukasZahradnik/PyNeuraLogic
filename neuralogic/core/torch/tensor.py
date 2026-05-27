from typing import Any

import torch


class NeuralogicOptTensor(torch.Tensor):
    """
    Experimental PyTorch tensor subclass that synchronizes its data with PyNeuraLogic Java weights.
    It allows using PyTorch optimizers to directly update PyNeuraLogic weights by overriding
    in-place operations and provide gradients from the Java engine.
    """

    _neuralogic_weight_updater = None
    _neuralogic_weight = None
    _neuralogic_value_factory = None

    def add_(self, other: Any, *args: Any, **kwargs: Any) -> Any:
        r = super().add_(other, *args, **kwargs)
        if self._neuralogic_weight is None:
            return r

        self.__sync_neuralogic__()

        return r

    def addcdiv_(self, tensor1: Any, tensor2: Any, *args: Any, **kwargs: Any) -> Any:
        r = super().addcdiv_(tensor1, tensor2, *args, **kwargs)
        if self._neuralogic_weight is None:
            return r

        self.__sync_neuralogic__()

        return r

    def addcmul_(self, tensor1: Any, tensor2: Any, *args: Any, **kwargs: Any) -> Any:
        r = super().addcmul_(tensor1, tensor2, *args, **kwargs)
        if self._neuralogic_weight is None:
            return r

        self.__sync_neuralogic__()

        return r

    @property
    def grad(self) -> torch.Tensor | None:
        """
        Returns the gradient for this tensor, retrieved from the PyNeuraLogic weight updater.

        Returns
        -------
        torch.Tensor
            The gradient tensor, or None if no gradient is available.
        """
        grad_list = self._neuralogic_weight_updater.weightUpdates
        index = self._neuralogic_weight.index

        if len(grad_list) <= index:
            return None

        value = grad_list[index]
        if value is None:
            return None

        p = -torch.tensor(self._neuralogic_value_factory.from_java(value))
        return p

    @staticmethod
    def create(weight: Any, data: Any, weight_updater: Any, value_factory: Any) -> "NeuralogicOptTensor":
        """
        Creates a new NeuralogicOptTensor wrapping specific PyNeuraLogic weight.

        Parameters
        ----------
        weight : Any
            Java weight object to wrap.
        data : Any
            Initial data for the tensor.
        weight_updater : Any
            Java weight updater providing gradients.
        value_factory : Any
            Value factory for Java to Python conversion.

        Returns
        -------
        NeuralogicOptTensor
            The created specialized tensor.
        """
        tensor = torch.tensor(data, requires_grad=True)
        tensor.__class__ = NeuralogicOptTensor
        tensor._neuralogic_weight = weight
        tensor._neuralogic_weight_updater = weight_updater
        tensor._neuralogic_value_factory = value_factory

        return tensor

    def __sync_neuralogic__(self) -> None:
        """
        Synchronizes the current tensor data back to the underlying Java weight.
        """
        size = self.size()
        weight_value = self._neuralogic_weight.value

        if len(size) == 0:
            weight_value.set(0, float(self))
            return

        if len(size) == 1:
            for i, val in enumerate(self):
                weight_value.set(i, float(val))
            return

        cols = size[0]

        for i, values in enumerate(self):
            for j, val in enumerate(values):
                weight_value.set(i * cols + j, float(val))
