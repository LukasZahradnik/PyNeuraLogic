from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainerHistory:
    """Training history collected during a :meth:`Trainer.fit` run.

    Attributes
    ----------
    train_losses : list[float]
        Mean training loss per epoch.
    val_losses : list[float]
        Mean validation loss per epoch (empty if no validation set).
    train_metrics : dict[str, list[float]]
        Per-epoch extra metrics on the training set (each key maps to a
        list of epoch-level means).
    val_metrics : dict[str, list[float]]
        Per-epoch extra metrics on the validation set.
    learning_rates : list[float]
        Learning rate at each epoch.
    best_epoch : int
        Epoch (0-indexed) that achieved the lowest validation loss.
    best_val_loss : float
        Lowest validation loss observed.
    stopped_early : bool
        ``True`` if early stopping fired.
    """

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_metrics: dict[str, list[float]] = field(default_factory=dict)
    val_metrics: dict[str, list[float]] = field(default_factory=dict)
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False
