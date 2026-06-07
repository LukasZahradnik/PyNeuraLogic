from __future__ import annotations

from typing import Any

from neuralogic.core.builder.dataset import BuiltDataset, GroundedDataset
from neuralogic.core.neural_module import NeuralModule
from neuralogic.dataset import Dataset


def _ensure_built(
    module: NeuralModule,
    dataset: Dataset | GroundedDataset | BuiltDataset,
    batch_size: int,
) -> BuiltDataset:
    """Convert a raw or grounded dataset to a ``BuiltDataset`` if needed."""
    if isinstance(dataset, BuiltDataset):
        return dataset
    if isinstance(dataset, GroundedDataset):
        return dataset.neuralize()
    if isinstance(dataset, Dataset):
        return module.build_dataset(dataset, batch_size=batch_size)
    raise TypeError(f"Unsupported dataset type: {type(dataset)}")


def _unpack_results(results):
    """Unpack ``train()`` return value into three parallel lists.

    ``NeuralModule.train()`` returns a single ``(target, output, error)``
    tuple for one sample, or a list of such tuples for multiple samples.
    """
    if not isinstance(results, list):
        # Single-sample path (shouldn't happen with BuiltDataset, but be safe)
        t, o, e = results
        return [t], [o], [e]

    if len(results) == 0:
        return [], [], []

    first = results[0]
    if isinstance(first, tuple) and len(first) == 3 and not isinstance(first[0], tuple):
        targets = [r[0] for r in results]
        outputs = [r[1] for r in results]
        errors = [r[2] for r in results]
        return targets, outputs, errors

    # Fallback: single tuple in a list
    t, o, e = results
    return [t], [o], [e]


def _mean(values: list) -> float:
    """Mean of a list of numbers (floats or nested structures)."""
    import numpy as np

    if not values:
        return float("nan")
    try:
        arr = np.asarray(values, dtype=float)
        return float(np.mean(arr))
    except (ValueError, TypeError):
        return float(sum(float(v) for v in values) / len(values))


def _build_logs(train_loss: float, val_loss: float | None, lr: float | None = None) -> dict[str, Any]:
    """Assemble the per-epoch log dictionary."""
    logs: dict[str, Any] = {"train_loss": train_loss}
    if val_loss is not None:
        logs["val_loss"] = val_loss
    if lr is not None:
        logs["lr"] = lr
    return logs
