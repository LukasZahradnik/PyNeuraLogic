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
        (*XOR(), [0.0, 0.85, 0.77, 0.923]),
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
                0.397,
                0.292,
                -0.076,
                -0.015,
                -0.001,
                0.116,
                0.116,
                -0.06,
                0.135,
                -0.035,
                0.028,
                0.015,
                -0.008,
                -0.008,
                0.072,
                -0.088,
                0.155,
                -0.218,
                -0.001,
                0.201,
                0.397,
                -0.015,
                0.033,
                0.116,
                0.155,
                0.054,
                0.17,
                0.068,
                -0.143,
                0.01,
                -0.017,
                0.293,
                0.202,
                0.122,
                -0.144,
                0.224,
                -0.097,
                0.346,
                0.298,
                0.298,
                0.212,
                0.224,
                0.298,
                0.397,
                0.397,
                0.269,
                -0.068,
                0.128,
                0.357,
                0.396,
                -0.097,
                -0.035,
                0.396,
                0.17,
                -0.06,
                0.069,
                -0.209,
                -0.077,
                -0.164,
                -0.015,
                0.245,
                0.015,
                0.292,
                0.224,
                -0.073,
                0.122,
                0.371,
                0.154,
                0.224,
                0.356,
                0.331,
                0.197,
                0.205,
                0.014,
                0.518,
                0.25,
                0.397,
                0.154,
                0.061,
                -0.083,
                -0.017,
                0.015,
                -0.058,
                0.034,
                0.397,
                -0.152,
                0.154,
                -0.06,
                -0.164,
                0.054,
                0.305,
                0.251,
                0.17,
                -0.015,
                0.17,
                0.357,
                0.05,
                -0.076,
                0.17,
                0.292,
                -0.06,
                0.072,
                0.224,
                0.035,
                -0.1,
                0.122,
                0.154,
                0.317,
                -0.144,
                -0.025,
                -0.076,
                0.252,
                0.371,
                0.17,
                0.298,
                0.331,
                0.17,
                0.136,
                0.398,
                -0.097,
                0.17,
                0.121,
                -0.088,
                -0.007,
                -0.08,
                0.003,
                0.067,
                -0.096,
                -0.014,
                -0.006,
                -0.096,
                -0.15,
                0.072,
                -0.008,
                -0.168,
                -0.19,
                -0.212,
                0.03,
                -0.376,
                -0.008,
                -0.017,
                -0.087,
                -0.008,
                -0.181,
                -0.169,
                -0.181,
                -0.017,
                -0.161,
                -0.017,
                -0.111,
                0.113,
                0.031,
                -0.074,
                -0.017,
                -0.008,
                0.112,
                0.113,
                -0.06,
                -0.076,
                -0.008,
                -0.183,
                -0.017,
                -0.097,
                -0.122,
                0.113,
                -0.171,
                -0.214,
                -0.056,
                0.14,
                -0.016,
                -0.057,
                -0.074,
                -0.171,
                0.036,
                -0.138,
                -0.143,
                -0.056,
                -0.19,
                0.034,
                -0.169,
                -0.077,
                0.139,
                -0.084,
                0.068,
                -0.121,
                -0.012,
                0.033,
                0.006,
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
    "template, dataset, expected_results",
    [
        (naive_xor.template, naive_xor.dataset, [0.0, 0.877, 0.875, 0.015]),
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
def test_evaluator_run_on_rules(template: Template, dataset: BaseDataset, expected_results: List[float]) -> None:
    """Tests for running java evaluator on rules"""
    manual_seed(0)
    settings = Settings(optimizer=SGD(lr=0.1), epochs=300)

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
