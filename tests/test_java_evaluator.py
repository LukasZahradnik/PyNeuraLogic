from typing import List

from neuralogic import manual_seed
from neuralogic.core import Settings, Template
from neuralogic.dataset.base import BaseDataset
from neuralogic.nn import get_evaluator
from neuralogic.optim import SGD
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
    "template, dataset, expected_results",
    [
        (*XOR(), [0, 0.625, 0.645, 0.663]),
        (*XOR_Vectorized(), [0, 0.7, 0.657, -0.056]),
        (
            *Trains(),
            [
                0.693,
                -0.751,
                0.744,
                0.733,
                0.714,
                0.75,
                0.736,
                -0.726,
                0.754,
                0.758,
                0.403,
                -0.752,
                -0.624,
                -0.742,
                -0.378,
                -0.758,
                -0.756,
                -0.758,
                -0.687,
                -0.383,
            ],
        ),
        (
            *Mutagenesis(),
            [
                0.473,
                0.358,
                -0.083,
                0.069,
                -0.022,
                0.221,
                0.221,
                -0.014,
                0.216,
                -0.052,
                0.11,
                0.065,
                0.029,
                0.029,
                0.116,
                -0.008,
                0.255,
                -0.214,
                0.086,
                0.295,
                0.473,
                0.07,
                0.114,
                0.221,
                0.255,
                0.141,
                0.266,
                0.019,
                -0.106,
                -0.014,
                0.022,
                0.388,
                0.202,
                0.221,
                -0.14,
                0.316,
                -0.047,
                0.425,
                0.377,
                0.376,
                0.292,
                0.316,
                0.377,
                0.473,
                0.473,
                0.255,
                0.047,
                0.227,
                0.429,
                0.469,
                -0.045,
                -0.052,
                0.469,
                0.266,
                -0.015,
                0.021,
                -0.166,
                -0.084,
                -0.09,
                0.086,
                0.329,
                0.064,
                0.358,
                0.316,
                0.007,
                0.221,
                0.468,
                0.256,
                0.316,
                0.429,
                0.406,
                0.285,
                0.137,
                0.096,
                0.607,
                0.303,
                0.473,
                0.256,
                0.171,
                0.011,
                0.022,
                0.112,
                0.033,
                0.135,
                0.473,
                -0.147,
                0.26,
                -0.014,
                -0.09,
                0.141,
                0.381,
                0.303,
                0.266,
                0.086,
                0.267,
                0.429,
                0.134,
                -0.083,
                0.261,
                0.358,
                -0.015,
                0.115,
                0.316,
                0.129,
                0.015,
                0.221,
                0.26,
                0.379,
                -0.14,
                0.055,
                -0.083,
                0.309,
                0.468,
                0.266,
                0.377,
                0.406,
                0.267,
                0.224,
                0.477,
                -0.046,
                0.266,
                0.216,
                -0.008,
                0.075,
                -0.047,
                -0.103,
                0.128,
                -0.105,
                -0.038,
                -0.026,
                -0.106,
                -0.143,
                0.115,
                0.029,
                -0.12,
                -0.143,
                -0.194,
                -0.072,
                -0.4,
                0.029,
                0.038,
                -0.098,
                0.029,
                -0.149,
                -0.162,
                -0.136,
                -0.05,
                -0.118,
                -0.036,
                -0.093,
                0.084,
                -0.068,
                -0.013,
                0.042,
                0.029,
                0.058,
                0.061,
                -0.013,
                -0.052,
                0.029,
                -0.17,
                -0.048,
                -0.045,
                -0.087,
                0.06,
                -0.162,
                -0.196,
                -0.068,
                0.174,
                -0.048,
                0.02,
                -0.013,
                -0.161,
                0.01,
                -0.137,
                -0.107,
                -0.068,
                -0.143,
                0.072,
                -0.162,
                -0.065,
                0.174,
                -0.058,
                0.133,
                -0.086,
                -0.031,
                0.065,
                -0.025,
            ],
        ),
    ],
)
def test_evaluator_run_on_files(template: Template, dataset: BaseDataset, expected_results: List[float]) -> None:
    """Tests for running java evaluator on files"""
    manual_seed(0)
    settings = Settings(optimizer=SGD(0.1), epochs=50)

    evaluator = get_evaluator(template, settings)

    built_dataset = evaluator.build_dataset(dataset)
    evaluator.train(built_dataset, generator=False)

    results = []
    for predicted in evaluator.test(built_dataset):
        results.append(round(predicted, 3))

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert expected_result == result


@pytest.mark.parametrize(
    "template, dataset, expected_results, seed",
    [
        (naive_xor.template, naive_xor.dataset, [0, 0.936, 0.935, -0.002], 0),
        (vectorized_xor.template, vectorized_xor.dataset, [0, 0.955, 0.954, -0.003], 0),
        (horses.template, horses.dataset, [0.951, 0], 0),
        (
            naive_trains.template,
            naive_trains.dataset,
            [
                0.743,
                0.745,
                0.735,
                -0.761,
                -0.74,
                -0.761,
                0.739,
                0.761,
                -0.761,
                -0.761,
                -0.761,
                -0.761,
                -0.761,
                -0.761,
                -0.746,
                -0.761,
                -0.761,
                -0.749,
                -0.734,
                0.761,
            ],
            0,
        ),
        (
            multiple_examples_trains.template,
            multiple_examples_trains.dataset,
            [
                0.743,
                0.745,
                0.735,
                -0.761,
                -0.74,
                -0.761,
                0.739,
                0.761,
                -0.761,
                -0.761,
                -0.761,
                -0.761,
                -0.761,
                -0.761,
                -0.746,
                -0.761,
                -0.761,
                -0.749,
                -0.734,
                0.761,
            ],
            0,
        ),
        (
            multiple_examples_no_order_trains.template,
            multiple_examples_no_order_trains.dataset,
            [
                0.685,
                0.715,
                0.746,
                0.759,
                0.556,
                0.731,
                0.128,
                0.724,
                0.158,
                0.751,
                -0.699,
                -0.761,
                -0.759,
                0.761,
                -0.762,
                -0.756,
                -0.755,
                -0.761,
                -0.762,
                -0.754,
            ],
            1,
        ),
    ],
)
def test_evaluator_run_on_rules(
    template: Template, dataset: BaseDataset, expected_results: List[float], seed: int
) -> None:
    """Tests for running java evaluator on rules"""
    manual_seed(seed)
    settings = Settings(optimizer=SGD(lr=0.1), epochs=300)

    evaluator = get_evaluator(template, settings)

    built_dataset = evaluator.build_dataset(dataset)
    evaluator.train(built_dataset, generator=False)

    results = []
    for predicted in evaluator.test(built_dataset):
        results.append(round(predicted, 3))

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert result == expected_result


@pytest.mark.parametrize(
    "template, dataset",
    [
        (naive_xor.template, naive_xor.dataset),
    ],
)
def test_evaluator_state_loading(template: Template, dataset: BaseDataset) -> None:
    """Tests for loading state"""
    settings = Settings(optimizer=SGD(0.1), epochs=20)

    evaluator = get_evaluator(template, settings)
    built_dataset = evaluator.build_dataset(dataset)
    evaluator.train(built_dataset, generator=False)

    results = []
    for predicted in evaluator.test(built_dataset):
        results.append(round(predicted, 5))

    second_evaluator = get_evaluator(template, settings)
    built_dataset = second_evaluator.build_dataset(dataset)

    second_results = []
    for predicted in second_evaluator.test(built_dataset):
        second_results.append(round(predicted, 5))

    assert len(results) == len(second_results)
    assert any(result != second_result for result, second_result in zip(results, second_results))

    second_evaluator.load_state_dict(evaluator.state_dict())

    second_results = []
    for predicted in second_evaluator.test(built_dataset):
        second_results.append(round(predicted, 5))

    assert len(results) == len(second_results)
    for result, second_result in zip(results, second_results):
        assert result == second_result
