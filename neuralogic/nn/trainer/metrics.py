from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from enum import Enum
from typing import Callable, Union


_METRIC_REGISTRY: dict[str, Callable[[list, list], float]] = {}


def _register(name: str):
    """Decorator that registers a batch-level metric function."""

    def deco(fn: Callable[[list, list], float]) -> Callable[[list, list], float]:
        _METRIC_REGISTRY[name] = fn
        return fn

    return deco


class Metric(str, Enum):
    """Enum of available metric names.

    Inherits ``str`` so members can be used directly where a string is
    expected::

        >>> Metric.ACCURACY == "accuracy"
        True
        >>> trainer.fit(..., metrics=[Metric.ACCURACY, Metric.F1_MACRO])
    """

    # Regression
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"
    # Classification
    ACCURACY = "accuracy"
    PRECISION_MACRO = "precision_macro"
    RECALL_MACRO = "recall_macro"
    F1_MACRO = "f1_macro"


def _to_arrays(targets: list, outputs: list):
    """Convert parallel lists of target/output values to numpy arrays."""
    import numpy as np

    t_arr = np.asarray(targets, dtype=float)
    o_arr = np.asarray(outputs, dtype=float)
    return t_arr, o_arr


def _class_indices(arr) -> "np.ndarray":
    """Convert array to integer class indices.

    Scalars → threshold 0.5.  Vectors → argmax.  2D → row-wise argmax.
    """
    import numpy as np

    if arr.ndim == 0:
        return np.asarray(int(arr >= 0.5)).reshape(1)
    if arr.ndim == 1:
        return np.asarray(int(np.argmax(arr))).reshape(1)
    # 2D: row-wise argmax
    return np.argmax(arr, axis=-1)


def _macro_score(t_cls: "np.ndarray", o_cls: "np.ndarray", mode: str) -> float:
    """Compute macro-averaged precision, recall, or F1."""
    import numpy as np

    classes = np.unique(np.concatenate([t_cls, o_cls]))
    scores: list[float] = []

    for c in classes:
        tp = int(np.sum((t_cls == c) & (o_cls == c)))
        fp = int(np.sum((t_cls != c) & (o_cls == c)))
        fn = int(np.sum((t_cls == c) & (o_cls != c)))

        if mode == "precision":
            scores.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        elif mode == "recall":
            scores.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        elif mode == "f1":
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            scores.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


@_register("mae")
def _mae(targets, outputs) -> float:
    """Mean absolute error."""
    import numpy as np

    t, o = _to_arrays(targets, outputs)
    return float(np.mean(np.abs(t - o)))


@_register("mse")
def _mse(targets, outputs) -> float:
    """Mean squared error."""
    import numpy as np

    t, o = _to_arrays(targets, outputs)
    return float(np.mean((t - o) ** 2))


@_register("rmse")
def _rmse(targets, outputs) -> float:
    """Root mean squared error."""
    import numpy as np

    t, o = _to_arrays(targets, outputs)
    return float(math.sqrt(np.mean((t - o) ** 2)))


@_register("r2")
def _r2(targets, outputs) -> float:
    """R\u00b2 coefficient of determination."""
    import numpy as np

    t, o = _to_arrays(targets, outputs)
    ss_res = np.sum((t - o) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


@_register("accuracy")
def _accuracy(targets, outputs) -> float:
    """Fraction of samples where predicted class equals target class.

    Scalars are thresholded at 0.5; vectors and 2D rows use argmax.
    """
    import numpy as np

    t_cls = _class_indices(np.asarray(targets, dtype=float))
    o_cls = _class_indices(np.asarray(outputs, dtype=float))
    return float(np.mean(t_cls == o_cls))


@_register("precision_macro")
def _precision_macro(targets, outputs) -> float:
    """Macro-averaged precision (unweighted mean of per-class precision)."""
    import numpy as np

    t_arr, o_arr = _to_arrays(targets, outputs)
    t_cls = _class_indices(t_arr)
    o_cls = _class_indices(o_arr)
    return _macro_score(t_cls, o_cls, "precision")


@_register("recall_macro")
def _recall_macro(targets, outputs) -> float:
    """Macro-averaged recall (unweighted mean of per-class recall)."""
    import numpy as np

    t_arr, o_arr = _to_arrays(targets, outputs)
    t_cls = _class_indices(t_arr)
    o_cls = _class_indices(o_arr)
    return _macro_score(t_cls, o_cls, "recall")


@_register("f1_macro")
def _f1_macro(targets, outputs) -> float:
    """Macro-averaged F1 score (unweighted mean of per-class F1)."""
    import numpy as np

    t_arr, o_arr = _to_arrays(targets, outputs)
    t_cls = _class_indices(t_arr)
    o_cls = _class_indices(o_arr)
    return _macro_score(t_cls, o_cls, "f1")


def compute_metrics(
    targets: list,
    outputs: list,
    names: Sequence[Union[str, Metric]],
) -> dict[str, float]:
    """Compute named metrics over a batch of (target, output) pairs.

    Each metric receives the full batch and returns a single float.

    Parameters
    ----------
    targets : list
        Per-sample target values (floats, lists, or 2D lists).
    outputs : list
        Per-sample output values (same shapes as targets).
    names : Sequence[str or Metric]
        Metric names to compute, e.g. ``["accuracy"]`` or
        ``[Metric.MAE, Metric.R2]``.

    Returns
    -------
    dict[str, float]
        Mapping from metric name to its value across the batch.
    """
    result: dict[str, float] = {}
    for name in names:
        key = str(name)
        if key not in _METRIC_REGISTRY:
            warnings.warn(f"Unknown metric '{key}'. Available: {sorted(_METRIC_REGISTRY)}")
            continue
        fn = _METRIC_REGISTRY[key]
        result[key] = fn(targets, outputs)
    return result


def _validate_metrics(metrics: list[str]) -> None:
    """Warn about unknown metric names."""
    unknown = [m for m in metrics if str(m) not in _METRIC_REGISTRY]
    if unknown:
        warnings.warn(f"Unknown metric(s): {unknown}. Available: {sorted(_METRIC_REGISTRY)}")
