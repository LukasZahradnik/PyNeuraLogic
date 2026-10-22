from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Union

from neuralogic.core.builder.dataset import BuiltDataset, GroundedDataset
from neuralogic.core.builder.static_graph import StaticGraphDataset
from neuralogic.core.neural_module import NeuralModule
from neuralogic.dataset import Dataset
from neuralogic.nn.trainer.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    ProgressCallback,
    TrainerCallback,
)
from neuralogic.nn.trainer.helpers import _build_logs, _ensure_built, _mean, _unpack_results
from neuralogic.nn.trainer.history import TrainerHistory
from neuralogic.nn.trainer.metrics import JAVA_METRIC_NAMES, Metric, _validate_metrics, compute_metrics


class Trainer:
    def __init__(self, module: NeuralModule) -> None:
        if module._neural_model is None:
            raise ValueError("The model must be built before creating a Trainer. Call model.build(settings) first.")
        self.model = module
        self.stop_training: bool = False

    def fit(
        self,
        train_dataset: Dataset | GroundedDataset | BuiltDataset | StaticGraphDataset,
        val_dataset: Dataset | GroundedDataset | BuiltDataset | StaticGraphDataset | None = None,
        *,
        epochs: int = 1,
        batch_size: int = 1,
        early_stopping_patience: int | None = None,
        min_delta: float = 0.0,
        checkpoint_dir: str | Path | None = None,
        metrics: Sequence[Union[str, Metric]] | None = None,
        silent: bool = False,
        callbacks: Sequence[TrainerCallback] | None = None,
    ) -> TrainerHistory:
        """Run the training loop.

        Parameters
        ----------
        train_dataset :
            Training data.  Raw ``Dataset`` objects are built automatically;
            pass a ``BuiltDataset`` to skip repeated grounding.
        val_dataset :
            Optional validation data.  When provided, validation loss (and
            any requested metrics) are computed after every epoch.  Early
            stopping and checkpointing depend on validation loss.
        epochs : int
            Number of epochs to train.  Default 1.
        batch_size : int
            Batch size when building raw datasets.  Default 1.
        early_stopping_patience : int or None
            Stop after this many epochs without validation-loss improvement.
            Requires ``val_dataset``.  Default ``None`` (no early stopping).
        min_delta : float
            Minimum absolute change in validation loss to count as
            improvement.  Default 0.0.
        checkpoint_dir : str, Path, or None
            Directory to save the best model (by validation loss).  A file
            named ``best.pkl`` is written on every improvement.  Default
            ``None`` (no checkpointing).
        metrics : Sequence[str or Metric] or None
            Extra metrics to compute, e.g. ``[Metric.ACCURACY]`` or
            ``["mae", "r2"]``.  Loss is always tracked.  The deep
            classification metrics ``auc_roc``, ``auc_pr``,
            ``best_accuracy``, ``best_threshold`` and the ``saturation``
            ratio are computed by the Java training strategy.  Default
            ``None``.
        silent : bool
            If ``True``, suppress the tqdm progress bar.  Default ``False``.
        callbacks : Sequence[TrainerCallback] or None
            Additional callbacks to invoke.  Built-in callbacks (early
            stopping, checkpoint, progress) are appended automatically
            based on the other arguments.

        Returns
        -------
        TrainerHistory
            Losses and metrics for every epoch.
        """
        metric_names = [m.value if isinstance(m, Metric) else str(m) for m in metrics] if metrics else []
        _validate_metrics(metric_names)
        java_metric_names = [name for name in metric_names if name in JAVA_METRIC_NAMES]
        python_metric_names = [name for name in metric_names if name not in JAVA_METRIC_NAMES]

        built_train = _ensure_built(self.model, train_dataset, batch_size)
        built_val = _ensure_built(self.model, val_dataset, batch_size) if val_dataset is not None else None

        optimizer = self.model._settings.optimizer
        lr_decay = optimizer._lr_decay if hasattr(optimizer, "_lr_decay") else None

        cb_list: list[TrainerCallback] = []

        if early_stopping_patience is not None:
            cb_list.append(EarlyStoppingCallback(early_stopping_patience, min_delta))
        if checkpoint_dir is not None:
            cb_list.append(CheckpointCallback(checkpoint_dir))
        if not silent:
            cb_list.append(ProgressCallback(epochs))
        if callbacks:
            cb_list.extend(callbacks)

        history = TrainerHistory()
        self.stop_training = False

        for cb in cb_list:
            cb.on_train_begin(self)

        for epoch in range(epochs):
            if self.stop_training:
                history.stopped_early = True
                break

            if java_metric_names:
                train_results, train_java_metrics = self.model.train_with_metrics(built_train, epochs=1, metrics=java_metric_names)
                train_mets = dict(train_java_metrics)
            else:
                train_results = self.model.train(built_train, epochs=1)
                train_mets = {}
            train_targets, train_outputs, train_errors = _unpack_results(train_results)
            if python_metric_names and train_outputs:
                train_mets.update(compute_metrics(train_targets, train_outputs, python_metric_names))
            train_loss = _mean(train_errors)
            history.train_losses.append(train_loss)

            val_loss: float | None = None
            val_mets: dict[str, float] = {}
            if built_val is not None:
                # Validation used to train on the validation set and restore the weights afterwards. The
                # weights did come back, but the optimizer's own state did not - a state_dict carries weight
                # values only - so Adam kept the moments it picked up from validation and every later epoch
                # drifted. Evaluating touches neither.
                if java_metric_names:
                    val_results, val_java_metrics = self.model.validate_with_metrics(built_val, metrics=java_metric_names)
                    val_mets = dict(val_java_metrics)
                else:
                    val_results = self.model.validate(built_val)
                val_targets, val_outputs, val_errors = _unpack_results(val_results)
                if python_metric_names and val_outputs:
                    val_mets.update(compute_metrics(val_targets, val_outputs, python_metric_names))
                val_loss = _mean(val_errors)
                history.val_losses.append(val_loss)

                if val_loss < history.best_val_loss:
                    history.best_val_loss = val_loss
                    history.best_epoch = epoch

            current_lr = optimizer.lr
            history.learning_rates.append(current_lr)
            if lr_decay is not None:
                lr_decay.decay(epoch)

            logs = _build_logs(train_loss, val_loss, current_lr)
            for name, value in train_mets.items():
                history.train_metrics.setdefault(name, []).append(value)
                logs[f"train_{name}"] = value
            if built_val is not None:
                for name, value in val_mets.items():
                    history.val_metrics.setdefault(name, []).append(value)
                    logs[f"val_{name}"] = value

            for cb in cb_list:
                cb.on_epoch_end(self, epoch, logs)

        for cb in cb_list:
            cb.on_train_end(self)

        return history

    def test(
        self,
        dataset: Dataset | GroundedDataset | BuiltDataset | StaticGraphDataset,
        *,
        batch_size: int = 1,
    ) -> list:
        """Evaluate the model on a dataset (no weight updates).

        Parameters
        ----------
        dataset :
            Test data.  Raw ``Dataset`` objects are built automatically.
        batch_size : int
            Batch size when building raw datasets.  Default 1.

        Returns
        -------
        list
            Model outputs for every sample.
        """
        built = _ensure_built(self.model, dataset, batch_size)
        result = self.model.test(built)
        if isinstance(result, list):
            return result
        return [result]
