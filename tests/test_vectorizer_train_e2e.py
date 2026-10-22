from typing import List

import pytest

pytest.importorskip("compute_graph_vectorize")
pytest.importorskip("torch")

import torch

from neuralogic import manual_seed
from neuralogic.core import Settings, Model
from neuralogic.dataset.base import BaseDataset
from neuralogic.nn.optim import SGD
from neuralogic.utils.data import XOR, XOR_Vectorized, Trains

from examples.datasets import (
    multiple_examples_trains,
    multiple_examples_no_order_trains,
    naive_trains,
    naive_xor,
    vectorized_xor,
)


def _build_targets(built_dataset) -> torch.Tensor:
    return torch.tensor([float(sample.target) for sample in built_dataset], dtype=torch.float32)


def _train_vectorized_sgd_online(
    model: Model, dataset: BaseDataset, *, lr: float, epochs: int
) -> List[float]:
    built_dataset = model.build_dataset(dataset)
    targets = _build_targets(built_dataset)

    vectorized = model.vectorize(dataset)
    optimizer = torch.optim.SGD(vectorized.parameters(), lr=lr)

    for _ in range(epochs):
        for index in range(len(targets)):
            optimizer.zero_grad()
            output = vectorized(0).flatten()
            loss = torch.nn.functional.mse_loss(output[index], targets[index])
            loss.backward()
            optimizer.step()

    return [round(float(value), 3) for value in vectorized(0).detach().flatten()]


@pytest.mark.parametrize(
    "model, dataset, expected_results, lr, epochs",
    [
        (*XOR(), [0.0, 1.0, 0.942, 1.0], 0.1, 50),
        (*XOR_Vectorized(), [0.0, 0.847, 0.836, -0.027], 0.1, 50),
        (naive_xor.model, naive_xor.dataset, [0.0, 0.977, 1.0, 1.0], 0.1, 300),
        (vectorized_xor.model, vectorized_xor.dataset, [0.0, 0.959, 0.958, -0.002], 0.1, 300),
    ],
)
def test_vectorized_training_matches_java(
    model: Model, dataset: BaseDataset, expected_results: List[float], lr: float, epochs: int
) -> None:
    """Training the vectorized graph with online SGD reproduces the Java training."""
    manual_seed(0)
    model.build(Settings(optimizer=SGD(lr), iso_value_compression=False, chain_pruning=False))

    results = _train_vectorized_sgd_online(model, dataset, lr=lr, epochs=epochs)

    assert len(results) == len(expected_results)
    assert results == expected_results


@pytest.mark.parametrize(
    "model, dataset, seed",
    [
        (*Trains(), 0),
        (naive_trains.model, naive_trains.dataset, 0),
        (multiple_examples_trains.model, multiple_examples_trains.dataset, 0),
        (multiple_examples_no_order_trains.model, multiple_examples_no_order_trains.dataset, 1),
    ],
)
def test_vectorized_training_reduces_loss(model: Model, dataset: BaseDataset, seed: int) -> None:
    """Training the vectorized graph with a PyTorch optimizer reduces the loss."""
    manual_seed(seed)
    model.build(Settings(iso_value_compression=False, chain_pruning=False))

    built_dataset = model.build_dataset(dataset)
    targets = _build_targets(built_dataset)

    vectorized = model.vectorize(dataset)
    optimizer = torch.optim.Adam(vectorized.parameters(), lr=0.01)

    initial_loss = torch.nn.functional.mse_loss(vectorized(0).flatten(), targets)

    for _ in range(100):
        optimizer.zero_grad()
        output = vectorized(0).flatten()
        loss = torch.nn.functional.mse_loss(output, targets)
        loss.backward()
        optimizer.step()

    final_output = vectorized(0).detach().flatten()

    assert float(loss.detach()) < float(initial_loss.detach())
    assert torch.isfinite(final_output).all()
