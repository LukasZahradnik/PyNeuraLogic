from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuralogic.nn.trainer.trainer import Trainer


class TrainerCallback:
    """Base class for training callbacks.

    Override any of the hooks.  The trainer calls them in the order they
    were passed to :meth:`Trainer.fit`.
    """

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called once before the first epoch."""

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, Any]) -> None:
        """Called after every epoch.

        Parameters
        ----------
        trainer : Trainer
            The trainer instance (access ``trainer.model``, etc.).
        epoch : int
            0-indexed epoch number that just finished.
        logs : dict
            Dictionary with keys ``"train_loss"``, ``"val_loss"`` (if
            available), ``"lr"``, and per-metric keys like
            ``"train_accuracy"``, ``"val_mae"``, etc.
        """

    def on_train_end(self, trainer: Trainer) -> None:
        """Called once after training finishes (or early-stops)."""


class EarlyStoppingCallback(TrainerCallback):
    """Stop training when validation loss stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement after which training stops.
    min_delta : float
        Minimum absolute change to qualify as an improvement.
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best: float = float("inf")

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, Any]) -> None:
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return

        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            trainer.stop_training = True

    def on_train_begin(self, trainer: Trainer) -> None:
        self._counter = 0
        self._best = float("inf")


class CheckpointCallback(TrainerCallback):
    """Save the model whenever validation loss improves.

    Parameters
    ----------
    directory : str or Path
        Directory to save checkpoints into (created if missing).
    filename : str
        Filename for the checkpoint file (default ``"best.pkl"``).
    """

    def __init__(self, directory: str | Path, filename: str = "best.pkl") -> None:
        self.directory = Path(directory)
        self.filename = filename
        self._best: float = float("inf")

    @property
    def path(self) -> Path:
        return self.directory / self.filename

    def on_train_begin(self, trainer: Trainer) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._best = float("inf")

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, Any]) -> None:
        val_loss = logs.get("val_loss")
        if val_loss is None:
            return  # can't decide without validation loss
        if val_loss < self._best:
            self._best = val_loss
            trainer.model.save(str(self.path))


class ProgressCallback(TrainerCallback):
    """Show a tqdm progress bar during training.

    Parameters
    ----------
    epochs : int
        Total number of epochs (used for progress bar length).
    """

    def __init__(self, epochs: int) -> None:
        self._epochs = epochs
        self._pbar: Any = None

    def on_train_begin(self, trainer: Trainer) -> None:
        try:
            from tqdm import tqdm

            self._pbar = tqdm(total=self._epochs, desc="Training", unit="epoch")
        except ImportError:
            self._pbar = None

    def on_epoch_end(self, trainer: Trainer, epoch: int, logs: dict[str, Any]) -> None:
        if self._pbar is None:
            return
        postfix: dict[str, Any] = {}
        for key in ("train_loss", "val_loss", "lr"):
            if key in logs:
                postfix[key] = f"{logs[key]:.4f}"
        for key, value in logs.items():
            if key.startswith("val_") and key not in ("val_loss", "lr"):
                postfix[key] = f"{value:.4f}"
        self._pbar.set_postfix(postfix)
        self._pbar.update(1)

    def on_train_end(self, trainer: Trainer) -> None:
        if self._pbar is not None:
            self._pbar.close()
