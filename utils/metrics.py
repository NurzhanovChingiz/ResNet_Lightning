"""Track classification metrics and mean loss for one training split."""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, MeanMetric, Metric, Precision, Recall

LOGGER = logging.getLogger(__name__)

MetricTask = Literal["binary", "multiclass", "multilabel"]


class NoNanMeanMetric(MeanMetric):
    """MeanMetric that casts inputs without rejecting NaN values."""

    def _cast_and_nan_check_input(
        self,
        x: float | torch.Tensor,
        weight: float | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cast ``x`` and ``weight`` onto this metric's device and dtype.

        Purpose:
            Coerce the input and weight onto this metric's device and dtype
            without rejecting NaN values.

        Strategy:
            Promote ``x`` to a tensor if needed, move it onto the metric
            device and dtype, coerce or default the weight the same way, then
            return the aligned pair.

        Args:
            x: Scalar or tensor loss value.
            weight: Optional weight aligned with ``x``.

        Returns:
            Device-local input tensor and weight tensor.

        Flow:
            1. Coerce x
            2. Move device/dtype
            3. Coerce or default weight
            4. Move weight
            5. Return
        """
        # 1. Coerce x
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        # 2. Move device/dtype
        x = x.to(device=self.device, dtype=self.dtype)
        # 3. Coerce or default weight
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight)
        if weight is None:
            weight = torch.ones_like(x)
        # 4. Move weight
        weight = weight.to(device=self.device, dtype=self.dtype)
        # 5. Return
        return x, weight


class MetricsTracker(nn.Module):
    """Update and compute accuracy, precision, recall, F1, and mean loss."""

    def __init__(
        self, *, task: MetricTask, num_classes: int, device: torch.device
    ) -> None:
        """Create metric modules for a classification task.

        Args:
            task: Torchmetrics task name, such as ``multiclass``.
            num_classes: Number of classes in the task.
            device: Device used when updating the mean-loss metric.
        """
        super().__init__()
        self._device = device
        self.avg_loss = NoNanMeanMetric()
        self.accuracy_top1 = Accuracy(
            task=task,
            num_classes=num_classes,
            average="weighted",
            top_k=1,
        )
        self.accuracy_top5 = Accuracy(
            task=task,
            num_classes=num_classes,
            average="weighted",
            top_k=5,
        )
        self.precision = Precision(task=task, num_classes=num_classes, average="macro")
        self.recall = Recall(task=task, num_classes=num_classes, average="macro")
        self.f1_score = F1Score(task=task, num_classes=num_classes, average="macro")

    def update_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate classification metrics from a batch.

        Args:
            preds: Model logits or probabilities.
            targets: Integer class labels.
        """
        self.accuracy_top1.update(preds, targets)
        self.accuracy_top5.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
        self.f1_score.update(preds, targets)

    def update_loss(self, loss_value: float | torch.Tensor) -> None:
        """Accumulate a batch loss into the running mean.

        Args:
            loss_value: Scalar loss from the current batch.
        """
        loss_tensor = torch.as_tensor(
            loss_value, dtype=self.avg_loss.dtype, device=self._device
        )
        self.avg_loss.update(loss_tensor, weight=torch.ones_like(loss_tensor))

    def compute(self) -> dict[str, float]:
        """Compute current metric values as Python floats.

        Returns:
            Mapping from metric name to scalar value.
        """
        return {
            "avg_loss": self.avg_loss.compute().item(),
            "accuracy_top1": self.accuracy_top1.compute().item(),
            "accuracy_top5": self.accuracy_top5.compute().item(),
            "precision": self.precision.compute().item(),
            "recall": self.recall.compute().item(),
            "f1_score": self.f1_score.compute().item(),
        }

    def reset(self) -> None:
        """Clear every tracked metric."""
        self.avg_loss.reset()
        self.accuracy_top1.reset()
        self.accuracy_top5.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_score.reset()

    def get_metric(self, name: str) -> Metric:
        """Return one named metric module.

        Args:
            name: Attribute name of a tracked metric.

        Returns:
            The torchmetrics module stored under ``name``.

        Raises:
            KeyError: If ``name`` is not a tracked metric.
        """
        metric = getattr(self, name, None)
        if not isinstance(metric, Metric):
            message = f"Unknown metric '{name}'"
            raise KeyError(message)
        return metric

    def print_metrics(self) -> None:
        """Log computed metric values when the tracker has received updates, then reset.

        Purpose:
            Emit a formatted metric summary for this split and reset the
            tracker after logging.

        Strategy:
            Return early when the loss window is empty; otherwise compute,
            format, log, then reset.

        Returns:
            None. Resets tracked metrics after a successful log.

        Flow:
            1. Empty-window guard
            2. Compute
            3. Format
            4. Log
            5. Reset
        """
        # 1. Empty-window guard
        if self.avg_loss.weight.sum() == 0:
            return
        # 2. Compute
        computed_metrics = self.compute()
        # 3. Format
        formatted = " | ".join(
            f"{name}: {value:.6f}" for name, value in computed_metrics.items()
        )
        # 4. Log
        LOGGER.info("%s", formatted)
        # 5. Reset
        self.reset()
