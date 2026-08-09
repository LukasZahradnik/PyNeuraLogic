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
        (*XOR(), [0.0, 1.0, 0.96, 1.0]),
        (*XOR_Vectorized(), [0.0, 0.847, 0.836, -0.027]),
        (
            *Trains(),
            [0.737, 0.457, 0.746, 0.737, 0.448, 0.737, 0.454, 0.737, 0.737, 0.737, -0.689, 0.457, -0.688, 0.737, -0.688, -0.733, 0.737, -0.734, 0.749, -0.687],
        ),
        (
            *Mutagenesis(),
            [0.095, 0.064, -0.001, -0.007, 0.065, 0.009, 0.009, -0.052, 0.032, 0.054, 0.056, -0.081, 0.053, 0.054, -0.024, -0.035, 0.042, -0.187, -0.017, 0.079, 0.095, -0.009, 0.049, 0.009, 0.042, 0.025, 0.03, -0.052, -0.083, 0.053, 0.002, 0.095, 0.063, -0.021, -0.09, 0.079, -0.091, 0.047, 0.028, 0.034, 0.001, 0.081, 0.028, 0.094, 0.093, 0.063, 0.024, 0.031, 0.084, 0.093, -0.097, 0.054, 0.093, 0.026, -0.048, -0.05, -0.132, -0.0, -0.018, -0.022, -0.02, -0.077, 0.064, 0.079, -0.033, -0.018, 0.056, 0.049, 0.079, 0.086, 0.079, -0.063, -0.051, 0.049, 0.108, 0.075, 0.096, 0.048, -0.006, -0.113, 0.002, 0.006, -0.035, 0.055, 0.095, -0.082, 0.049, -0.051, -0.018, 0.025, 0.072, 0.073, 0.027, -0.022, 0.024, 0.083, 0.04, -0.001, 0.054, 0.064, -0.048, -0.021, 0.082, 0.052, -0.018, -0.018, 0.049, 0.089, -0.09, 0.04, -0.001, 0.076, 0.056, 0.027, 0.028, 0.079, 0.024, 0.086, 0.096, -0.094, 0.028, 0.007, 0.021, 0.044, 0.038, -0.091, -0.04, 0.023, 0.053, 0.07, 0.016, -0.002, -0.021, 0.054, -0.097, -0.071, -0.137, -0.185, -0.409, 0.051, -0.015, 0.008, 0.052, -0.081, -0.075, -0.108, -0.106, -0.099, 0.078, -0.054, -0.06, -0.103, 0.022, -0.017, 0.055, 0.015, 0.014, -0.054, 0.011, 0.05, -0.083, -0.104, -0.099, -0.082, 0.018, -0.007, -0.075, -0.092, 0.047, -0.103, -0.078, 0.017, -0.008, -0.049, 0.008, -0.077, -0.092, -0.07, 0.057, -0.074, -0.008, 0.051, 0.02, -0.039, -0.086, 0.074, 0.061, -0.045],
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
        (naive_xor.model, naive_xor.dataset, [0.0, 0.984, 1.0, 1.0], 0),
        (vectorized_xor.model, vectorized_xor.dataset, [0.0, 0.959, 0.958, -0.002], 0),
        (horses.model, horses.dataset, [0.953, 0.0], 0),
        (
            naive_trains.model,
            naive_trains.dataset,
            [0.746, 0.536, 0.751, 0.747, 0.53, 0.747, 0.534, 0.747, 0.749, 0.747, -0.726, 0.536, -0.711, 0.747, -0.711, -0.747, 0.746, -0.747, 0.754, -0.711],
            0,
        ),
        (
            multiple_examples_trains.model,
            multiple_examples_trains.dataset,
            [0.746, 0.536, 0.751, 0.747, 0.53, 0.747, 0.534, 0.747, 0.749, 0.747, -0.726, 0.536, -0.711, 0.747, -0.711, -0.747, 0.746, -0.747, 0.754, -0.711],
            0,
        ),
        (
            multiple_examples_no_order_trains.model,
            multiple_examples_no_order_trains.dataset,
            [0.689, 0.652, 0.76, 0.745, 0.722, 0.746, 0.659, 0.732, -0.122, 0.752, -0.074, -0.733, -0.761, 0.761, -0.762, -0.399, -0.222, -0.761, -0.761, -0.754],
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
