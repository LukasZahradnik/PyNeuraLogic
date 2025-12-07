import torch


class NeuralogicOptTensor(torch.Tensor):
    _neuralogic_weight_updater = None
    _neuralogic_weight = None
    _neuralogic_value_factory = None

    def add_(self, other, *args, **kwargs):
        r = super().add_(other, *args, **kwargs)
        if self._neuralogic_weight is None:
            return r

        self.__sync_neuralogic__()

        return r

    def addcdiv_(self, tensor1, tensor2, *args, **kwargs):
        r = super().addcdiv_(tensor1, tensor2, *args, **kwargs)
        if self._neuralogic_weight is None:
            return r

        self.__sync_neuralogic__()

        return r

    def addcmul_(self, tensor1, tensor2, *args, **kwargs):
        r = super().addcmul_(tensor1, tensor2, *args, **kwargs)
        if self._neuralogic_weight is None:
            return r

        self.__sync_neuralogic__()

        return r

    @property
    def grad(self):
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
    def create(weight, data, weight_updater, value_factory):
        tensor = torch.tensor(data, requires_grad=True)
        tensor.__class__ = NeuralogicOptTensor
        tensor._neuralogic_weight = weight
        tensor._neuralogic_weight_updater = weight_updater
        tensor._neuralogic_value_factory = value_factory

        return tensor

    def __sync_neuralogic__(self):
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
