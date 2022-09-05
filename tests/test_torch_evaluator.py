from typing import List

from neuralogic import manual_seed
from neuralogic.core import Settings, Template, Backend
from neuralogic.dataset.base import BaseDataset
from neuralogic.nn import get_evaluator
from neuralogic.optim import SGD
from neuralogic.utils.data import XOR, XOR_Vectorized, Trains

from examples.datasets import (
    multiple_examples_trains,
    multiple_examples_no_order_trains,
    naive_trains,
    naive_xor,
    horses,
    vectorized_xor,
)

import torch
import pytest


@pytest.mark.parametrize(
    "template, dataset, expected_results",
    [
        (*XOR(), [0.0, 0.831, 0.833, 0.955]),
        (*XOR_Vectorized(), [0.0, 0.732, 0.707, -0.068]),
        (
            *Trains(),
            [
                0.757,
                0.727,
                0.758,
                0.759,
                0.741,
                0.757,
                0.758,
                0.755,
                0.73,
                0.712,
                -0.7,
                -0.701,
                -0.747,
                0.755,
                -0.755,
                -0.702,
                0.758,
                -0.756,
                -0.759,
                -0.743,
            ],
        ),
    ],
)
def test_evaluator_run_on_files(template: Template, dataset: BaseDataset, expected_results: List[float]) -> None:
    """Tests for running torch evaluator on files"""
    torch.manual_seed(1)
    manual_seed(0)

    settings = Settings(optimizer=SGD(lr=0.1), epochs=50)

    evaluator = get_evaluator(template, settings, Backend.TORCH)

    built_dataset = evaluator.build_dataset(dataset)
    evaluator.train(built_dataset, generator=False)

    results = []
    for predicted in evaluator.test(built_dataset):
        results.append(round(predicted, 3))

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert round(expected_result, 2) == round(result, 2)


@pytest.mark.parametrize(
    "template, dataset, expected_results",
    [
        (naive_xor.template, naive_xor.dataset, [0.0, 0.677, 0.631, 0.718]),
        (vectorized_xor.template, vectorized_xor.dataset, [0.0, 0.883, 0.881, -0.021]),
        (horses.template, horses.dataset, [0.908, -0.0]),
        (
            naive_trains.template,
            naive_trains.dataset,
            [
                0.76,
                0.746,
                0.76,
                0.761,
                0.752,
                0.76,
                0.761,
                0.76,
                0.749,
                0.743,
                -0.733,
                -0.733,
                -0.754,
                0.76,
                -0.759,
                -0.734,
                0.76,
                -0.76,
                -0.761,
                -0.752,
            ],
        ),
        (
            multiple_examples_trains.template,
            multiple_examples_trains.dataset,
            [
                0.76,
                0.746,
                0.76,
                0.761,
                0.752,
                0.76,
                0.761,
                0.76,
                0.749,
                0.743,
                -0.733,
                -0.733,
                -0.754,
                0.76,
                -0.759,
                -0.734,
                0.76,
                -0.76,
                -0.761,
                -0.752,
            ],
        ),
        # (
        #     multiple_examples_no_order_trains.template,
        #     multiple_examples_no_order_trains.dataset,
        #     [
        #         0.742,
        #         0.725,
        #         0.759,
        #         0.761,
        #         0.758,
        #         0.752,
        #         0.719,
        #         0.752,
        #         0.03,
        #         0.753,
        #         -0.004,
        #         -0.71,
        #         -0.757,
        #         0.761,
        #         -0.761,
        #         -0.005,
        #         0.006,
        #         -0.761,
        #         -0.761,
        #         -0.742,
        #     ],
        # ),
    ],
)
def test_evaluator_run_on_rules(
    template: Template, dataset: BaseDataset, expected_results: List[float], seed: int = 1
) -> None:
    """Tests for running torch evaluator on rules"""
    torch.manual_seed(seed)
    manual_seed(0)

    settings = Settings(optimizer=SGD(), epochs=100)

    evaluator = get_evaluator(template, settings, Backend.TORCH)

    built_dataset = evaluator.build_dataset(dataset)
    evaluator.train(built_dataset, generator=False)

    results = []
    for predicted in evaluator.test(built_dataset):
        results.append(round(predicted, 3))

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert round(expected_result, 1) == round(result, 1)


@pytest.mark.parametrize(
    "template, dataset",
    [
        (naive_xor.template, naive_xor.dataset),
    ],
)
def test_evaluator_state_loading(template: Template, dataset: BaseDataset, seed: int = 1) -> None:
    """Tests for loading state"""
    torch.manual_seed(seed)
    settings = Settings(optimizer=SGD(lr=0.1), epochs=20)

    evaluator = get_evaluator(template, settings, Backend.TORCH)
    built_dataset = evaluator.build_dataset(dataset)
    evaluator.train(built_dataset, generator=False)

    results = []

    for predicted in evaluator.test(built_dataset):
        results.append(round(predicted, 3))

    second_evaluator = get_evaluator(template, settings, Backend.TORCH)
    built_dataset = second_evaluator.build_dataset(dataset)

    second_results = []
    for predicted in second_evaluator.test(built_dataset):
        second_results.append(round(predicted, 3))

    assert len(results) == len(second_results)
    assert any(result != second_result for result, second_result in zip(results, second_results))

    second_evaluator.load_state_dict(evaluator.state_dict())

    second_results = []
    for predicted in second_evaluator.test(built_dataset):
        second_results.append(round(predicted, 3))

    assert len(results) == len(second_results)
    for result, second_result in zip(results, second_results):
        assert round(result, 2) == round(second_result, 2)
