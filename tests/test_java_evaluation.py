from typing import List

from neuralogic import manual_seed
from neuralogic.core import Settings, Model
from neuralogic.dataset.base import BaseDataset
from neuralogic.nn.optim import SGD
from neuralogic.utils.data import XOR, XOR_Vectorized, Trains, Mutagenesis

from examples.datasets import (
    multiple_examples_trains,
    multiple_examples_no_order_trains,
    naive_trains,
    naive_xor,
    horses,
    vectorized_xor,
)


import pytest


@pytest.mark.parametrize(
    "model, dataset, expected_results",
    [
        (*XOR(), [0.0, 1.0, 0.948, 1.0]),
        (*XOR_Vectorized(), [0.0, 0.619, 0.557, 0.513]),
        (
            *Trains(),
            [-0.73, 0.624, -0.055, 0.739, -0.058, 0.756, -0.042, -0.664, -0.758, -0.732, -0.748, 0.342, -0.748, -0.756, -0.745, -0.747, -0.732, -0.748, -0.757, -0.748],
        ),
        (
            *Mutagenesis(),
            [0.471, 0.356, -0.08, 0.024, -0.017, 0.21, 0.21, -0.037, 0.206, -0.048, 0.061, 0.065, 0.004, 0.004, 0.116, -0.045, 0.238, -0.206, 0.04, 0.252, 0.471, 0.024, 0.066, 0.21, 0.238, 0.12, 0.224, 0.036, -0.115, -0.006, -0.001, 0.352, 0.19, 0.181, -0.132, 0.273, -0.068, 0.425, 0.374, 0.373, 0.29, 0.273, 0.374, 0.471, 0.471, 0.247, -0.008, 0.219, 0.427, 0.469, -0.067, -0.048, 0.469, 0.224, -0.037, 0.038, -0.176, -0.081, -0.116, 0.031, 0.325, 0.064, 0.356, 0.273, -0.031, 0.18, 0.452, 0.246, 0.273, 0.427, 0.403, 0.281, 0.173, 0.049, 0.604, 0.303, 0.471, 0.246, 0.116, -0.032, -0.001, 0.057, -0.013, 0.077, 0.471, -0.14, 0.246, -0.037, -0.116, 0.12, 0.378, 0.303, 0.224, 0.031, 0.225, 0.427, 0.085, -0.08, 0.244, 0.356, -0.037, 0.115, 0.273, 0.075, -0.037, 0.18, 0.246, 0.378, -0.132, 0.013, -0.08, 0.305, 0.452, 0.224, 0.374, 0.403, 0.224, 0.165, 0.472, -0.068, 0.224, 0.2, -0.045, 0.029, -0.063, -0.077, 0.119, -0.099, -0.028, -0.023, -0.099, -0.138, 0.115, 0.004, -0.134, -0.157, -0.188, -0.051, -0.361, 0.004, 0.007, -0.09, 0.004, -0.153, -0.154, -0.147, -0.033, -0.13, -0.034, -0.096, 0.108, -0.051, -0.041, 0.007, 0.004, 0.074, 0.075, -0.037, -0.061, 0.004, -0.165, -0.032, -0.067, -0.097, 0.075, -0.158, -0.191, -0.062, 0.176, -0.032, -0.019, -0.041, -0.157, 0.014, -0.132, -0.116, -0.062, -0.157, 0.041, -0.154, -0.068, 0.175, -0.07, 0.12, -0.096, -0.029, 0.04, -0.012],
        ),
    ],
)
def test_evaluator_run_on_files(model: Model, dataset: BaseDataset, expected_results: List[float]) -> None:
    """Tests for running java evaluator on files"""
    manual_seed(0)
    settings = Settings(optimizer=SGD(0.1))

    model.build(settings)

    built_dataset = model.build_dataset(dataset)
    model.train(built_dataset, epochs=50)

    results = []
    for predicted in model.test(built_dataset):
        results.append(round(predicted, 3))

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert result == expected_result


@pytest.mark.parametrize(
    "model, dataset, expected_results, seed",
    [
        (naive_xor.model, naive_xor.dataset, [0.0, 0.978, 1.0, 1.0], 0),
        (vectorized_xor.model, vectorized_xor.dataset, [0.0, 0.949, 0.948, -0.004], 0),
        (horses.model, horses.dataset, [0.951, 0], 0),
        (
            naive_trains.model,
            naive_trains.dataset,
            [-0.424, 0.758, 0.735, 0.758, 0.735, 0.757, 0.735, 0.756, -0.43, -0.427, -0.588, -0.722, -0.588, -0.429, -0.758, -0.754, -0.428, -0.588, -0.758, -0.589],
            0,
        ),
        (
            multiple_examples_trains.model,
            multiple_examples_trains.dataset,
            [-0.424, 0.758, 0.735, 0.758, 0.735, 0.757, 0.735, 0.756, -0.43, -0.427, -0.588, -0.722, -0.588, -0.429, -0.758, -0.754, -0.428, -0.588, -0.758, -0.589],
            0,
        ),
        (
            multiple_examples_no_order_trains.model,
            multiple_examples_no_order_trains.dataset,
            [0.747, 0.745, 0.758, 0.759, 0.76, 0.759, 0.745, 0.759, -0.005, 0.759, -0.004, -0.742, -0.761, 0.762, -0.762, -0.001, -0.001, -0.761, -0.762, -0.755],
            1,
        ),
    ],
)
def test_evaluator_run_on_rules(model: Model, dataset: BaseDataset, expected_results: List[float], seed: int) -> None:
    """Tests for running java evaluator on rules"""
    manual_seed(seed)
    settings = Settings(optimizer=SGD(lr=0.1))

    model.build(settings)

    built_dataset = model.build_dataset(dataset)
    model.train(built_dataset, epochs=300)

    results = []
    for predicted in model.test(built_dataset):
        results.append(round(predicted, 3))

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert result == expected_result


@pytest.mark.parametrize(
    "model, dataset",
    [
        (naive_xor.model, naive_xor.dataset),
    ],
)
def test_evaluator_state_loading(model: Model, dataset: BaseDataset) -> None:
    """Tests for loading state"""
    settings = Settings(optimizer=SGD(0.1))

    model.build(settings)
    built_dataset = model.build_dataset(dataset)
    model.train(built_dataset)

    results = []
    for predicted in model.test(built_dataset):
        results.append(round(predicted, 5))

    second_model = model.clone()
    second_model.build(settings)

    built_dataset = second_model.build_dataset(dataset)

    second_results = []
    for predicted in second_model.test(built_dataset):
        second_results.append(round(predicted, 5))

    assert len(results) == len(second_results)
    assert any(result != second_result for result, second_result in zip(results, second_results))

    second_model.load_state_dict(model.state_dict())

    second_results = []
    for predicted in second_model.test(built_dataset):
        second_results.append(round(predicted, 5))

    assert len(results) == len(second_results)
    for result, second_result in zip(results, second_results):
        assert result == second_result
