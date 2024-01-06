from typing import List

import pytest

from neuralogic import initialize, is_initialized
from neuralogic.optim import Adam
from neuralogic.optim.lr_scheduler import GeometricLR, ArithmeticLR


@pytest.mark.parametrize(
    "lr, max_steps, expected",
    [
        (100, 100, [99, 98, 97, 96]),
        (100, 10, [90, 80, 70, 60]),
    ],
)
def test_arithmetic_lr_decay(lr: float, max_steps: int, expected: List[float]):
    if not is_initialized():
        initialize()

    optimizer = Adam(lr=lr, lr_decay=ArithmeticLR(max_steps))
    optimizer.initialize()

    assert optimizer.lr == lr

    lr_decay = optimizer._lr_decay

    for exp in expected:
        lr_decay.decay(1)
        assert optimizer.lr == exp


def test_geometric_lr_decay():
    pass


@pytest.mark.parametrize(
    "lr, steps, rate, expected",
    [
        (100, 1, 1, [100, 100, 100, 100]),
        (100, 1, 0.9, [90, 90 * 0.9, 90 * (0.9**2), 90 * (0.9**3)]),
        (100, 2, 0.9, [100, 90, 90, 90 * 0.9, 90 * 0.9, 90 * (0.9**2), 90 * (0.9**2)]),
    ],
)
def test_geometric_lr_decay(lr: float, steps: int, rate: float, expected: List[float]):
    if not is_initialized():
        initialize()

    optimizer = Adam(lr=lr, lr_decay=GeometricLR(rate, steps))
    optimizer.initialize()

    assert optimizer.lr == lr

    lr_decay = optimizer._lr_decay

    for i, exp in enumerate(expected):
        lr_decay.decay(i + 1)
        assert optimizer.lr == exp
