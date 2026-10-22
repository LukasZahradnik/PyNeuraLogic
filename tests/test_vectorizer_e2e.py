from typing import List

import pytest

pytest.importorskip("compute_graph_vectorize")
pytest.importorskip("torch")

import torch

from neuralogic import manual_seed
from neuralogic.core import Settings, Model
from neuralogic.dataset import Dataset
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


def _vectorized_results(model: Model, dataset: BaseDataset) -> List[float]:
    if isinstance(dataset, Dataset):
        # The vectorizer emits only the last query layer per call.  When the
        # full batch already returns one value per sample, use it directly;
        # otherwise (e.g. horses, which has two query atoms) vectorize one
        # sample at a time and concatenate the outputs in sample order.
        vectorized = model.vectorize(dataset)
        results = [round(float(value), 3) for value in vectorized(0).detach().flatten()]

        if len(results) == len(dataset.samples):
            return results

        results = []
        for sample in dataset.samples:
            vectorized = model.vectorize(Dataset([sample]))
            results.extend(round(float(value), 3) for value in vectorized(0).detach().flatten())
        return results

    vectorized = model.vectorize(dataset)

    return [round(float(value), 3) for value in vectorized(0).detach().flatten()]


@pytest.mark.parametrize(
    "model, dataset, expected_results",
    [
        (*XOR(), [0.0, 1.0, 0.942, 1.0]),
        (*XOR_Vectorized(), [0.0, 0.847, 0.836, -0.027]),
        (
            *Trains(),
            [-0.737, -0.703, -0.703, 0.022, -0.703, 0.021, -0.703, 0.013, 0.736, 0.007, -0.737, -0.704, 0.079, 0.012, 0.083, -0.737, -0.737, -0.737, -0.148, 0.079],
        ),
        pytest.param(
            *Mutagenesis(),
            [0.095, 0.064, -0.001, -0.007, 0.065, 0.009, 0.009, -0.052, 0.032, 0.054, 0.056, -0.081, 0.053, 0.054, -0.024, -0.035, 0.042, -0.187, -0.017, 0.079, 0.095, -0.009, 0.049, 0.009, 0.042, 0.025, 0.03, -0.052, -0.083, 0.053, 0.002, 0.095, 0.063, -0.021, -0.09, 0.079, -0.091, 0.047, 0.028, 0.034, 0.001, 0.081, 0.028, 0.094, 0.093, 0.063, 0.024, 0.031, 0.084, 0.093, -0.097, 0.054, 0.093, 0.026, -0.048, -0.05, -0.132, -0.0, -0.018, -0.022, -0.02, -0.077, 0.064, 0.079, -0.033, -0.018, 0.056, 0.049, 0.079, 0.086, 0.079, -0.063, -0.051, 0.049, 0.108, 0.075, 0.096, 0.048, -0.006, -0.113, 0.002, 0.006, -0.035, 0.055, 0.095, -0.082, 0.049, -0.051, -0.018, 0.025, 0.072, 0.073, 0.027, -0.022, 0.024, 0.083, 0.04, -0.001, 0.054, 0.064, -0.048, -0.021, 0.082, 0.052, -0.018, -0.018, 0.049, 0.089, -0.09, 0.04, -0.001, 0.076, 0.056, 0.027, 0.028, 0.079, 0.024, 0.086, 0.096, -0.094, 0.028, 0.007, 0.021, 0.044, 0.038, -0.091, -0.04, 0.023, 0.053, 0.07, 0.016, -0.002, -0.021, 0.054, -0.097, -0.071, -0.137, -0.185, -0.409, 0.051, -0.015, 0.008, 0.052, -0.081, -0.075, -0.108, -0.106, -0.099, 0.078, -0.054, -0.06, -0.103, 0.022, -0.017, 0.055, 0.015, 0.014, -0.054, 0.011, 0.05, -0.083, -0.104, -0.099, -0.082, 0.018, -0.007, -0.075, -0.092, 0.047, -0.103, -0.078, 0.017, -0.008, -0.049, 0.008, -0.077, -0.092, -0.07, 0.057, -0.074, -0.008, 0.051, 0.02, -0.039, -0.086, 0.074, 0.061, -0.045],
            marks=pytest.mark.xfail(reason="vectorizer does not yet match Java for the Mutagenesis GNN template"),
        ),
    ],
)
def test_vectorizer_run_on_files(model: Model, dataset: BaseDataset, expected_results: List[float]) -> None:
    """Tests for running the vectorizer on file datasets."""
    manual_seed(0)
    settings = Settings(optimizer=SGD(0.1), iso_value_compression=False, chain_pruning=False)

    model.build(settings)

    built_dataset = model.build_dataset(dataset)
    model.train(built_dataset, epochs=50)

    results = _vectorized_results(model, dataset)

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert result == expected_result


@pytest.mark.parametrize(
    "model, dataset, expected_results, seed",
    [
        (naive_xor.model, naive_xor.dataset, [0.0, 0.977, 1.0, 1.0], 0),
        (vectorized_xor.model, vectorized_xor.dataset, [0.0, 0.959, 0.958, -0.002], 0),
        (horses.model, horses.dataset, [0.953, 0.0], 0),
        (
            naive_trains.model,
            naive_trains.dataset,
            [0.683, -0.739, 0.73, 0.746, 0.733, 0.746, 0.733, -0.71, 0.745, 0.685, 0.14, -0.74, -0.157, -0.71, -0.132, -0.752, -0.748, -0.752, -0.69, -0.18],
            0,
        ),
        (
            multiple_examples_trains.model,
            multiple_examples_trains.dataset,
            [0.683, -0.739, 0.73, 0.746, 0.733, 0.746, 0.733, -0.71, 0.745, 0.685, 0.14, -0.74, -0.157, -0.71, -0.132, -0.752, -0.748, -0.752, -0.69, -0.18],
            0,
        ),
        (
            multiple_examples_no_order_trains.model,
            multiple_examples_no_order_trains.dataset,
            [0.733, 0.724, 0.759, 0.752, 0.75, 0.752, 0.723, 0.752, -0.006, 0.752, -0.002, -0.722, -0.76, 0.761, -0.761, -0.005, -0.001, -0.757, -0.761, -0.752],
            1,
        ),
    ],
)
def test_vectorizer_run_on_rules(model: Model, dataset: BaseDataset, expected_results: List[float], seed: int) -> None:
    """Tests for running the vectorizer on rule datasets."""
    manual_seed(seed)
    settings = Settings(optimizer=SGD(lr=0.1), iso_value_compression=False, chain_pruning=False)

    model.build(settings)

    built_dataset = model.build_dataset(dataset)
    model.train(built_dataset, epochs=300)

    results = _vectorized_results(model, dataset)

    assert len(results) == len(expected_results)

    for result, expected_result in zip(results, expected_results):
        assert result == expected_result
