"""Lightning module that trains a model and logs split metrics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from lightning.pytorch import LightningModule
from torch import Tensor

from utils.metrics import MetricsTracker

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.utilities.types import OptimizerLRScheduler
    from torch import nn
    from torch.optim import Optimizer

    from utils.metrics import MetricTask

LOGGER = logging.getLogger(__name__)


class Model(LightningModule):
    """Train a wrapped ``nn.Module`` with loss, optimizer, and split metrics.

    Args:
        model: Network that maps features to logits.
        loss_fn: Loss module applied to logits and labels.
        optimizer: Optimizer over the network parameters.
        scheduler: Optional Lightning scheduler dict.
        task: Torchmetrics task name, such as ``multiclass``.
        num_classes: Number of classes in the task.
        device: Device used by the mean-loss metric.
        train_transform: Optional train-time batch transform.
        test_transform: Optional evaluation-time batch transform.
        print_metrics: If True, log a text summary at epoch end.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: dict[str, Any] | None = None,
        *,
        task: MetricTask,
        num_classes: int,
        device: torch.device,
        train_transform: Any = None,
        test_transform: Any = None,
        print_metrics: bool = False,
    ) -> None:
        """Store the network, loss, optimizer, and per-split metric trackers.

        Purpose:
            Bind training collaborators, persist hyperparameters, and construct
            per-split metric trackers.

        Strategy:
            Store the injected network, loss, optimizer, scheduler, transforms,
            and print flag; save Lightning hyperparameters while ignoring those
            objects; then construct independent train, validation, and test
            trackers.

        Args:
            model: Network that maps features to logits.
            loss_fn: Loss module applied to logits and labels.
            optimizer: Optimizer over the network parameters.
            scheduler: Optional Lightning scheduler dict.
            task: Torchmetrics task name, such as ``multiclass``.
            num_classes: Number of classes in the task.
            device: Device used by the mean-loss metric.
            train_transform: Optional train-time batch transform.
            test_transform: Optional evaluation-time batch transform.
            print_metrics: If True, log a text summary at epoch end.

        Returns:
            None.

        Flow:
            1. Bind collaborators
            2. Save hyperparameters
            3. Construct three split trackers
        """
        # 1. Bind collaborators
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.print_metrics = print_metrics
        # 2. Save hyperparameters
        self.save_hyperparameters(
            ignore=[
                "model",
                "loss_fn",
                "optimizer",
                "scheduler",
                "train_transform",
                "test_transform",
                "print_metrics",
                "task",
                "num_classes",
                "device",
            ]
        )

        # 3. Construct three split trackers
        self.train_metrics_tracker = MetricsTracker(
            task=task, num_classes=num_classes, device=device
        )
        self.val_metrics_tracker = MetricsTracker(
            task=task, num_classes=num_classes, device=device
        )
        self.test_metrics_tracker = MetricsTracker(
            task=task, num_classes=num_classes, device=device
        )

    def _shared_step(
        self, batch: tuple[Tensor, Tensor], tracker: MetricsTracker
    ) -> Tensor:
        """Run forward, compute loss, and update one split's metrics.

        Purpose:
            Process one batch through the network and update the given split
            tracker.

        Strategy:
            Unpack features and labels, run the forward pass, compute loss,
            reject a non-tensor loss, then update classification metrics and
            mean loss before returning the scalar.

        Args:
            batch: Feature tensor and label tensor.
            tracker: Metric tracker for the current split.

        Returns:
            Scalar batch loss.

        Raises:
            TypeError: If ``loss_fn`` does not return a ``torch.Tensor``.

        Flow:
            1. Unpack
            2. Forward
            3. Loss
            4. Type-check
            5. Update classification metrics
            6. Update loss
            7. Return
        """
        # 1. Unpack
        features, targets = batch
        # 2. Forward
        predictions = self.model(features)
        # 3. Loss
        loss = self.loss_fn(predictions, targets)
        # 4. Type-check
        if not isinstance(loss, Tensor):
            message = "loss_fn must return a torch.Tensor"
            raise TypeError(message)
        # 5. Update classification metrics
        tracker.update_metrics(predictions, targets)
        # 6. Update loss
        tracker.update_loss(loss.item())
        # 7. Return
        return loss

    def _log_epoch_metrics(
        self,
        tracker: MetricsTracker,
        *,
        split: str,
        title: str,
        progress_bar_top5: bool,
    ) -> None:
        """Log epoch metrics for one split, optionally print them, then reset.

        Purpose:
            Publish one split's epoch metrics, optionally print them, then
            reset the tracker.

        Strategy:
            Log each named scalar from the tracker, print a text summary when
            enabled, then reset so the next epoch starts empty.

        Args:
            tracker: Metric tracker that received updates this epoch.
            split: Log-name prefix such as ``train``, ``val``, or ``test``.
            title: Human-readable label written when ``print_metrics`` is set.
            progress_bar_top5: If True, show top-5 accuracy on the progress bar.

        Returns:
            None.

        Flow:
            1. Log named scalars
            2. Optionally print
            3. Reset
        """
        # 1. Log named scalars
        metric_logs = (
            ("loss", "avg_loss", True),
            ("acc_top1", "accuracy_top1", True),
            ("acc_top5", "accuracy_top5", progress_bar_top5),
            ("precision", "precision", False),
            ("recall", "recall", False),
            ("f1", "f1_score", False),
        )
        for suffix, metric_name, show_in_progress_bar in metric_logs:
            self.log(
                f"{split}_{suffix}",
                tracker.get_metric(metric_name).compute().item(),
                on_step=False,
                on_epoch=True,
                prog_bar=show_in_progress_bar,
                logger=True,
            )
        # 2. Optionally print
        if self.print_metrics:
            LOGGER.info("%s:", title)
            tracker.print_metrics()
        # 3. Reset
        tracker.reset()

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute training loss and update training metrics for one batch.

        Purpose:
            Run the shared training-batch step and log the running step loss.

        Strategy:
            Delegate the batch to the training tracker, log the current mean
            loss for the progress bar, and return the batch loss.

        Args:
            batch: Feature tensor and label tensor.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            Scalar training loss.

        Flow:
            1. Shared step
            2. Log step loss
            3. Return
        """
        del batch_idx
        # 1. Shared step
        loss = self._shared_step(batch, self.train_metrics_tracker)
        # 2. Log step loss
        self.log(
            "train_loss_step",
            self.train_metrics_tracker.get_metric("avg_loss"),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        # 3. Return
        return loss

    def on_train_epoch_end(self) -> None:
        """Log and reset training metrics after an epoch."""
        self._log_epoch_metrics(
            self.train_metrics_tracker,
            split="train",
            title="Training Metrics",
            progress_bar_top5=True,
        )

    def on_validation_epoch_end(self) -> None:
        """Log and reset validation metrics after an epoch."""
        self._log_epoch_metrics(
            self.val_metrics_tracker,
            split="val",
            title="Validation Metrics",
            progress_bar_top5=False,
        )

    def on_test_epoch_end(self) -> None:
        """Log and reset test metrics after an epoch."""
        self._log_epoch_metrics(
            self.test_metrics_tracker,
            split="test",
            title="Test Metrics",
            progress_bar_top5=False,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the stored optimizer and optional learning-rate scheduler.

        Returns:
            Optimizer alone, or a Lightning dict that includes the scheduler.
        """
        if self.scheduler is not None:
            return cast(
                "OptimizerLRScheduler",
                {"optimizer": self.optimizer, "lr_scheduler": self.scheduler},
            )
        return self.optimizer

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute validation loss and update validation metrics.

        Args:
            batch: Feature tensor and label tensor.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            Scalar validation loss.
        """
        del batch_idx
        return self._shared_step(batch, self.val_metrics_tracker)

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Compute test loss and update test metrics.

        Args:
            batch: Feature tensor and label tensor.
            batch_idx: Batch index supplied by Lightning.

        Returns:
            Scalar test loss.
        """
        del batch_idx
        return self._shared_step(batch, self.test_metrics_tracker)
