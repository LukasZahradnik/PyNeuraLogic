from typing import List

from neuralogic.core import Settings, Optimizer, Template, Dataset, Backend
from neuralogic.nn import get_evaluator
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
        (*XOR(), [0.0, 0.667, 0.648, 0.709]),
        (*XOR_Vectorized(), [0.0, 0.598, 0.545, 0.4]),
        (
            *Trains(),
            [
                0.75,
                0.746,
                0.752,
                0.759,
                0.679,
                0.759,
                0.73,
                0.752,
                0.716,
                0.747,
                -0.682,
                -0.76,
                -0.72,
                -0.744,
                -0.704,
                -0.76,
                -0.726,
                -0.71,
                -0.757,
                -0.745,
            ],
        ),
        (
            *Mutagenesis(),
            [
                0.274,
                0.145,
                -0.013,
                -0.045,
                0.03,
                0.115,
                0.114,
                -0.042,
                0.058,
                -0.004,
                -0.036,
                0.092,
                -0.057,
                -0.061,
                0.112,
                -0.101,
                0.083,
                -0.141,
                -0.004,
                0.098,
                0.303,
                -0.047,
                -0.038,
                0.115,
                0.083,
                0.031,
                0.124,
                0.057,
                -0.025,
                0.026,
                -0.032,
                0.185,
                0.055,
                0.107,
                -0.029,
                0.143,
                -0.044,
                0.249,
                0.296,
                0.291,
                0.194,
                0.139,
                0.296,
                0.313,
                0.318,
                0.084,
                -0.128,
                0.117,
                0.321,
                0.27,
                -0.05,
                -0.006,
                0.267,
                0.125,
                -0.038,
                0.058,
                -0.151,
                -0.017,
                -0.208,
                -0.096,
                0.273,
                0.088,
                0.143,
                0.143,
                -0.116,
                0.111,
                0.318,
                0.125,
                0.093,
                0.318,
                0.282,
                0.254,
                0.255,
                -0.073,
                0.476,
                0.175,
                0.244,
                0.125,
                -0.026,
                -0.025,
                -0.03,
                -0.059,
                -0.068,
                -0.034,
                0.303,
                -0.026,
                0.12,
                -0.04,
                -0.208,
                0.025,
                0.278,
                0.176,
                0.084,
                -0.096,
                0.128,
                0.321,
                0.008,
                -0.015,
                0.083,
                0.153,
                -0.036,
                0.108,
                0.14,
                -0.032,
                -0.128,
                0.106,
                0.12,
                0.233,
                -0.028,
                -0.082,
                -0.013,
                0.202,
                0.314,
                0.125,
                0.296,
                0.282,
                0.128,
                0.07,
                0.268,
                -0.047,
                0.12,
                0.069,
                -0.112,
                -0.071,
                -0.077,
                0.007,
                0.04,
                -0.006,
                0.007,
                0.032,
                -0.006,
                -0.121,
                0.107,
                -0.056,
                -0.011,
                -0.142,
                -0.136,
                0.024,
                -0.294,
                -0.046,
                -0.077,
                -0.01,
                -0.063,
                -0.073,
                -0.135,
                -0.117,
                -0.019,
                -0.195,
                0.037,
                -0.116,
                0.093,
                0.014,
                -0.002,
                -0.088,
                -0.059,
                0.166,
                0.165,
                -0.044,
                -0.008,
                -0.048,
                -0.082,
                -0.019,
                -0.052,
                -0.074,
                0.16,
                -0.123,
                -0.164,
                -0.089,
                0.131,
                -0.018,
                -0.13,
                -0.004,
                -0.119,
                0.091,
                -0.071,
                -0.024,
                -0.09,
                -0.142,
                -0.006,
                -0.135,
                -0.114,
                0.126,
                -0.003,
                0.032,
                -0.077,
                0.034,
                -0.022,
                0.04,
            ],
        ),
    ],
)
def test_evaluator_run_on_files(template: Template, dataset: Dataset, expected_results: List[float]) -> None:
    """Tests for running java evaluator on files"""
    settings = Settings(optimizer=Optimizer.SGD, learning_rate=0.1, epochs=50)

    evaluator = get_evaluator(template, Backend.JAVA, settings)

    built_dataset = evaluator.build_dataset(dataset)
    evaluator.train(built_dataset, generator=False)

    results = []
    for _, predicted in evaluator.test(built_dataset):
        results.append(round(predicted, 3))

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert expected_result == result


@pytest.mark.parametrize(
    "template, dataset, expected_results",
    [
        (naive_xor.template, naive_xor.dataset, [0.0, 0.878, 0.878, 0.014]),
        (vectorized_xor.template, vectorized_xor.dataset, [0.0, 0.923, 0.922, 0.004]),
        (horses.template, horses.dataset, [0.929, 0.0]),
        (
            naive_trains.template,
            naive_trains.dataset,
            [
                0.761,
                0.76,
                0.759,
                0.761,
                0.747,
                0.761,
                0.758,
                0.761,
                0.752,
                0.759,
                -0.747,
                -0.761,
                -0.754,
                -0.76,
                -0.751,
                -0.761,
                -0.755,
                -0.751,
                -0.761,
                -0.758,
            ],
        ),
        (
            multiple_examples_trains.template,
            multiple_examples_trains.dataset,
            [
                0.761,
                0.76,
                0.759,
                0.761,
                0.747,
                0.761,
                0.758,
                0.761,
                0.752,
                0.759,
                -0.747,
                -0.761,
                -0.754,
                -0.76,
                -0.751,
                -0.761,
                -0.755,
                -0.751,
                -0.761,
                -0.758,
            ],
        ),
        (
            multiple_examples_no_order_trains.template,
            multiple_examples_no_order_trains.dataset,
            [
                0.725,
                -0.761,
                0.762,
                0.76,
                0.742,
                0.756,
                0.632,
                0.317,
                0.586,
                0.749,
                0.003,
                -0.761,
                -0.762,
                -0.747,
                -0.762,
                -0.762,
                -0.737,
                -0.761,
                -0.762,
                -0.759,
            ],
        ),
    ],
)
def test_evaluator_run_on_rules(template: Template, dataset: Dataset, expected_results: List[float]) -> None:
    """Tests for running java evaluator on rules"""
    settings = Settings(optimizer=Optimizer.SGD, learning_rate=0.1, epochs=300)

    evaluator = get_evaluator(template, Backend.JAVA, settings)

    built_dataset = evaluator.build_dataset(dataset)
    evaluator.train(built_dataset, generator=False)

    results = []
    for _, predicted in evaluator.test(built_dataset):
        results.append(round(predicted, 3))

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert expected_result == result


@pytest.mark.parametrize(
    "template, dataset",
    [
        (naive_xor.template, naive_xor.dataset),
    ],
)
def test_evaluator_state_loading(template: Template, dataset: Dataset) -> None:
    """Tests for loading state"""
    settings = Settings(optimizer=Optimizer.SGD, learning_rate=0.1, epochs=20)

    evaluator = get_evaluator(template, Backend.JAVA, settings)
    built_dataset = evaluator.build_dataset(dataset)
    evaluator.train(built_dataset, generator=False)

    results = []
    for _, predicted in evaluator.test(built_dataset):
        results.append(round(predicted, 5))

    second_evaluator = get_evaluator(template, Backend.JAVA, settings)
    built_dataset = second_evaluator.build_dataset(dataset)

    second_results = []
    for _, predicted in second_evaluator.test(built_dataset):
        second_results.append(round(predicted, 5))

    assert len(results) == len(second_results)
    assert any(result != second_result for result, second_result in zip(results, second_results))

    second_evaluator.load_state_dict(evaluator.state_dict())

    second_results = []
    for _, predicted in second_evaluator.test(built_dataset):
        second_results.append(round(predicted, 5))

    assert len(results) == len(second_results)
    for result, second_result in zip(results, second_results):
        assert result == second_result
