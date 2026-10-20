"""Track classification metrics and mean loss for one training split."""

from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import nn
from torchmetrics import (
    Accuracy,
    F1Score,
    MeanMetric,
    Metric,
    MetricCollection,
    Precision,
    Recall,
)

LOGGER = logging.getLogger(__name__)

MetricTask = Literal["binary", "multiclass", "multilabel"]

_CLASSIFICATION_METRIC_NAMES = (
    "accuracy_top1",
    "accuracy_top5",
    "precision",
    "recall",
    "f1_score",
)
_COMPUTE_GROUPS = (
    ("accuracy_top1", "f1_score", "precision", "recall"),
    ("accuracy_top5",),
)


class NoNanMeanMetric(MeanMetric):
    """MeanMetric that casts inputs without rejecting NaN values."""

    def update(
        self,
        value: float | torch.Tensor,
        weight: float | torch.Tensor | None = None,
    ) -> None:
        """Accumulate one weighted value into the running mean.

        Purpose:
            Fold an already-aligned 0-dim value and weight into the running
            mean without recasting, and keep the generic MeanMetric path for
            other inputs.

        Strategy:
            When value and weight are 0-dim tensors on this metric's device
            and dtype, add them in place; otherwise defer to MeanMetric.

        Args:
            value: Scalar or tensor contribution.
            weight: Optional weight aligned with ``value``.

        Returns:
            None. Updates ``mean_value`` and ``weight`` state.

        Flow:
            1. Detect aligned 0-dim tensors
            2. In-place accumulate
            3. Defer other inputs
        """
        # 1. Detect aligned 0-dim tensors
        if (
            isinstance(value, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and value.ndim == 0
            and weight.ndim == 0
            and value.dtype == self.dtype
            and weight.dtype == self.dtype
            and value.device == self.device
            and weight.device == self.device
        ):
            # 2. In-place accumulate
            self.mean_value.addcmul_(value, weight)
            self.weight.add_(weight)
            return
        # 3. Defer other inputs
        super().update(value, weight)

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

        Purpose:
            Store the split device and construct the mean-loss metric plus a
            classification collection on that device so later batches reuse
            grouped metric state without a host copy.

        Strategy:
            Keep mean loss on its own module, wrap accuracy, precision,
            recall, and F1 in ``MetricCollection`` with pinned compute
            groups and disabled per-batch argument checks, move both
            modules onto a non-CPU constructor device, store each group's
            leader for later batch updates, then pin 0-dim loss value,
            unit-weight slots, and the top-1 confusion bin buffer on that
            device.

        Args:
            task: Torchmetrics task name, such as ``multiclass``.
            num_classes: Number of classes in the task.
            device: Device that owns the mean-loss and classification modules.

        Flow:
            1. Store device and class count
            2. Construct mean-loss metric
            3. Construct classification collection
            4. Move metrics onto device
            5. Pin compute-group leaders
            6. Pin loss slots
            7. Pin top-1 confusion bins
        """
        super().__init__()
        # 1. Store device and class count
        self._device = device
        self._num_classes = num_classes
        # 2. Construct mean-loss metric
        self.avg_loss = NoNanMeanMetric()
        # 3. Construct classification collection
        self._classification = MetricCollection(
            {
                "accuracy_top1": Accuracy(
                    task=task,
                    num_classes=num_classes,
                    average="weighted",
                    top_k=1,
                    validate_args=False,
                ),
                "accuracy_top5": Accuracy(
                    task=task,
                    num_classes=num_classes,
                    average="weighted",
                    top_k=5,
                    validate_args=False,
                ),
                "precision": Precision(
                    task=task,
                    num_classes=num_classes,
                    average="macro",
                    validate_args=False,
                ),
                "recall": Recall(
                    task=task,
                    num_classes=num_classes,
                    average="macro",
                    validate_args=False,
                ),
                "f1_score": F1Score(
                    task=task,
                    num_classes=num_classes,
                    average="macro",
                    validate_args=False,
                ),
            },
            compute_groups=[list(group) for group in _COMPUTE_GROUPS],
        )
        # 4. Move metrics onto device
        if device.type != "cpu":
            self.avg_loss.to(device)
            self._classification.to(device)
        # 5. Pin compute-group leaders
        self._classification.reset()
        self._classification_metrics = tuple(
            getattr(self._classification, name) for name in _CLASSIFICATION_METRIC_NAMES
        )
        self._classification_groups = tuple(
            tuple(getattr(self._classification, name) for name in group)
            for group in _COMPUTE_GROUPS
        )
        # 6. Pin loss slots
        self._loss_value = torch.empty(
            (), dtype=self.avg_loss.dtype, device=self._device
        )
        self._loss_weight = torch.ones(
            (), dtype=self.avg_loss.dtype, device=self._device
        )
        # 7. Pin top-1 confusion bins
        self._top1_bin_counts = torch.zeros(
            (num_classes * num_classes,),
            dtype=torch.int64,
            device=self._device,
        )

    def _accumulate_top1_stat_scores(
        self,
        leader: Metric,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Fold one top-1 batch into confusion counts without ``bincount``.

        Purpose:
            Build the same per-class tp/fp/tn/fn as torchmetrics' top-1
            global confusion path, using a pinned scatter-add buffer.

        Strategy:
            Disable gradients, reduce logits to class indices, scatter
            one count per sample into the pinned ``C * C`` buffer, derive
            the four stat tensors, then apply the leader's existing state
            write and increment its update count.

        Args:
            leader: Compute-group leader whose state stores confusion counts.
            preds: Model logits or already-reduced class indices.
            targets: Integer class labels.

        Returns:
            None. Updates ``leader`` confusion state and ``_update_count``.

        Flow:
            1. Reduce indices
            2. Scatter into pinned bins
            3. Derive tp/fp/tn/fn
            4. Write leader state
        """
        # 1. Reduce indices
        with torch.set_grad_enabled(False):
            pred_idx = preds.argmax(dim=1) if preds.ndim == targets.ndim + 1 else preds
            pred_idx = pred_idx.reshape(-1)
            target_idx = targets.reshape(-1)
            unique_mapping = target_idx.to(
                dtype=torch.long
            ) * self._num_classes + pred_idx.to(dtype=torch.long)
            # 2. Scatter into pinned bins
            self._top1_bin_counts.zero_()
            self._top1_bin_counts.scatter_add_(
                0, unique_mapping, torch.ones_like(unique_mapping)
            )
            confmat = self._top1_bin_counts.view(self._num_classes, self._num_classes)
            # 3. Derive tp/fp/tn/fn
            tp = confmat.diag().clone()
            fp = confmat.sum(0) - tp
            fn = confmat.sum(1) - tp
            tn = confmat.sum() - (fp + fn + tp)
        # 4. Write leader state
        updater = object.__getattribute__(leader, "_update_state")
        updater(tp, fp, tn, fn)
        object.__setattr__(leader, "_update_count", leader.update_count + 1)

    def update_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate classification metrics from a batch.

        Purpose:
            Fold one batch into grouped classification state without repeating
            MetricCollection's sibling-state rebind or top-1 ``bincount``.

        Strategy:
            Clear cached computes on every classification metric, accumulate
            top-1 confusion with the pinned scatter-add path, update other
            group leaders through torchmetrics, then copy the leader update
            count onto siblings that already share tensor state.

        Args:
            preds: Model logits or probabilities.
            targets: Integer class labels.

        Flow:
            1. Invalidate cached computes
            2. Update leaders and sibling counts
        """
        # 1. Invalidate cached computes
        for metric in self._classification_metrics:
            object.__setattr__(metric, "_computed", None)
        # 2. Update leaders and sibling counts
        for group in self._classification_groups:
            leader = group[0]
            if getattr(leader, "top_k", None) == 1:
                self._accumulate_top1_stat_scores(leader, preds, targets)
            else:
                leader.update(preds, targets)
            update_count = leader.update_count
            for sibling in group[1:]:
                object.__setattr__(sibling, "_update_count", update_count)

    def update_loss(self, loss_value: float | torch.Tensor) -> None:
        """Accumulate a batch loss into the running mean.

        Purpose:
            Fold one batch loss into the running mean without allocating a
            new 0-dim value or weight tensor on the warmed path.

        Strategy:
            Copy a tensor loss or fill a Python float into the pinned 0-dim
            slot, then update mean loss with that slot and the pinned unit
            weight.

        Args:
            loss_value: Scalar loss from the current batch.

        Flow:
            1. Write loss into pinned slot
            2. Update mean loss
        """
        # 1. Write loss into pinned slot
        if isinstance(loss_value, torch.Tensor):
            source = loss_value.detach()
            if source.ndim != 0:
                source = source.reshape(())
            self._loss_value.copy_(
                source.to(device=self._device, dtype=self._loss_value.dtype)
            )
        else:
            self._loss_value.fill_(loss_value)
        # 2. Update mean loss
        self.avg_loss.update(self._loss_value, weight=self._loss_weight)

    def compute(self) -> dict[str, float]:
        """Compute current metric values as Python floats.

        Purpose:
            Return named scalar metrics in a stable key order.

        Strategy:
            Compute mean loss, compute the classification collection, then
            assemble the public mapping without changing key names.

        Returns:
            Mapping from metric name to scalar value.

        Flow:
            1. Compute mean loss
            2. Compute classification collection
            3. Assemble named floats
        """
        # 1. Compute mean loss
        avg_loss = self.avg_loss.compute().item()
        # 2. Compute classification collection
        classification_values = self._classification.compute()
        # 3. Assemble named floats
        return {
            "avg_loss": avg_loss,
            "accuracy_top1": classification_values["accuracy_top1"].item(),
            "accuracy_top5": classification_values["accuracy_top5"].item(),
            "precision": classification_values["precision"].item(),
            "recall": classification_values["recall"].item(),
            "f1_score": classification_values["f1_score"].item(),
        }

    def reset(self) -> None:
        """Clear every tracked metric.

        Purpose:
            Empty the mean-loss window and every classification metric.

        Strategy:
            Reset the mean-loss module, then reset the classification
            collection in one call.

        Flow:
            1. Reset mean-loss metric
            2. Reset classification collection
        """
        # 1. Reset mean-loss metric
        self.avg_loss.reset()
        # 2. Reset classification collection
        self._classification.reset()

    def get_metric(self, name: str) -> Metric:
        """Return one named metric module.

        Purpose:
            Resolve a public metric name to the live torchmetrics module.

        Strategy:
            Return the mean-loss module by name, otherwise look up the live
            classification metric without copying shared state.

        Args:
            name: Attribute name of a tracked metric.

        Returns:
            The torchmetrics module stored under ``name``.

        Raises:
            KeyError: If ``name`` is not a tracked metric.

        Flow:
            1. Resolve mean-loss name
            2. Look up classification metric
            3. Reject unknown names
        """
        # 1. Resolve mean-loss name
        if name == "avg_loss":
            return self.avg_loss
        # 2. Look up classification metric
        metric = getattr(self._classification, name, None)
        # 3. Reject unknown names
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
