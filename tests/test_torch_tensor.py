from types import SimpleNamespace

import torch

from neuralogic.core.torch.tensor import NeuralogicOptTensor


class WeightUpdater:
    def __init__(self, updates):
        self.weightUpdates = updates

    def clearUpdates(self):
        self.weightUpdates = []


class FlatValue:
    def __init__(self, size):
        self.values = [0.0] * size

    def set(self, index, value):
        self.values[index] = value


class ValueFactory:
    @staticmethod
    def from_java(value):
        return value


def test_optimizer_zero_grad_clears_backend_updates():
    updater = WeightUpdater([[1.0, 2.0]])
    weight = SimpleNamespace(index=0, value=FlatValue(2))
    parameter = NeuralogicOptTensor.create(weight, [0.0, 0.0], updater, ValueFactory())
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    assert parameter.grad is not None
    optimizer.zero_grad()

    assert parameter.grad is None


def test_rectangular_matrix_sync_uses_column_stride():
    value = FlatValue(6)
    weight = SimpleNamespace(index=0, value=value)
    parameter = NeuralogicOptTensor.create(weight, [[0.0] * 3 for _ in range(2)], WeightUpdater([]), ValueFactory())
    replacement = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    with torch.no_grad():
        parameter.add_(replacement)

    assert value.values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
