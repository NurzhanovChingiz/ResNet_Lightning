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
_BatchStatSlots = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]

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


def _flat_index(tensor: torch.Tensor) -> torch.Tensor:
    """Return a 1-D index tensor without reshaping a vector that is already flat.

    Purpose:
        Avoid a no-op reshape dispatch when class indices are already 1-D.

    Strategy:
        Return the tensor when its rank is 1; otherwise reshape to a vector.

    Args:
        tensor: Predicted or target class indices.

    Returns:
        A 1-D view of ``tensor``, or a reshape when the rank is not 1.

    Flow:
        1. Keep a 1-D tensor
        2. Flatten other ranks
    """
    # 1. Keep a 1-D tensor
    if tensor.ndim == 1:
        return tensor
    # 2. Flatten other ranks
    return tensor.reshape(-1)


def _reduce_leader_pred_indices(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    top_k: int | None,
    shared_topk: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce one leader's predicted and target class indices.

    Purpose:
        Produce flattened predicted and target indices for one compute-group
        leader without repeating a logits top-k when the caller already
        reduced the batch, and without reshaping indices that are already 1-D.

    Strategy:
        Keep already-flat labels, reuse a shared top-k tensor when present,
        otherwise run top-k or argmax on logits, and keep already-reduced
        predictions.

    Args:
        preds: Model logits or already-reduced class indices.
        targets: Integer class labels.
        top_k: Leader top-k width, or 1.
        shared_topk: Optional top-k class indices already reduced from
            ``preds``.

    Returns:
        Flattened predicted class indices and flattened target indices.

    Flow:
        1. Keep already-flat labels
        2. Keep already-reduced predictions
        3. Reuse shared top-k or reduce logits
    """
    # 1. Keep already-flat labels
    target_idx = _flat_index(targets)
    # 2. Keep already-reduced predictions
    if preds.ndim != targets.ndim + 1:
        return _flat_index(preds), target_idx
    # 3. Reuse shared top-k or reduce logits
    if shared_topk is not None:
        top1_idx = shared_topk[:, 0]
        if top_k is not None and top_k > 1:
            in_topk = (
                shared_topk[:, : int(top_k)].eq(target_idx.unsqueeze(1)).any(dim=1)
            )
            pred_idx = torch.where(in_topk, target_idx, top1_idx)
        else:
            pred_idx = top1_idx
    elif top_k is not None and top_k > 1:
        topk_idx = preds.topk(k=int(top_k), dim=1).indices
        in_topk = topk_idx.eq(target_idx.unsqueeze(1)).any(dim=1)
        pred_idx = torch.where(in_topk, target_idx, topk_idx[:, 0])
    else:
        pred_idx = preds.argmax(dim=1)
    return _flat_index(pred_idx), target_idx


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
            leader and the widest top-k for a shared logits reduction,
            then pin 0-dim loss value, unit-weight slots, a true-positive
            scatter bin, a shared target-support histogram, match
            slots, and a top-k membership buffer on that device.

        Args:
            task: Torchmetrics task name, such as ``multiclass``.
            num_classes: Number of classes in the task.
            device: Device that owns the mean-loss and classification modules.

        Flow:
            1. Store device
            2. Construct mean-loss metric
            3. Construct classification collection
            4. Move metrics onto device
            5. Pin compute-group leaders
            6. Store widest top-k
            7. Pin loss slots
            8. Pin scatter scratch
            9. Pin top-k membership buffer
        """
        super().__init__()
        # 1. Store device
        self._device = device
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
        # 6. Store widest top-k
        self._max_top_k = max(
            int(getattr(group[0], "top_k", 1) or 1)
            for group in self._classification_groups
        )
        # 7. Pin loss slots
        self._loss_value = torch.empty(
            (), dtype=self.avg_loss.dtype, device=self._device
        )
        self._loss_weight = torch.ones(
            (), dtype=self.avg_loss.dtype, device=self._device
        )
        # 8. Pin scatter scratch
        self._stat_tp = torch.zeros(
            (num_classes,), dtype=torch.int64, device=self._device
        )
        self._stat_support = torch.zeros(
            (num_classes,), dtype=torch.int64, device=self._device
        )
        self._stat_ones = torch.ones((0,), dtype=torch.int64, device=self._device)
        self._stat_eq = torch.empty((0,), dtype=torch.bool, device=self._device)
        self._stat_match = torch.empty((0,), dtype=torch.int64, device=self._device)
        # 9. Pin top-k membership buffer
        self._topk_eq = torch.empty(
            (0, self._max_top_k), dtype=torch.bool, device=self._device
        )

    def _stat_slots(
        self, sample_count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pinned ones, equality, and match slices for one batch.

        Purpose:
            Grow the pinned per-sample scratch tensors when the batch is
            wider than the current slots, then return views for this batch.

        Strategy:
            Allocate ones, bool equality, and int64 match buffers when the
            stored length is too small, then slice each buffer to
            ``sample_count``.

        Args:
            sample_count: Number of flattened labels in the batch.

        Returns:
            Ones, equality, and match tensors of length ``sample_count``.

        Flow:
            1. Grow scratch when needed
            2. Slice this batch
        """
        # 1. Grow scratch when needed
        if self._stat_ones.numel() < sample_count:
            self._stat_ones = torch.ones(
                sample_count, dtype=torch.int64, device=self._device
            )
            self._stat_eq = torch.empty(
                sample_count, dtype=torch.bool, device=self._device
            )
            self._stat_match = torch.empty(
                sample_count, dtype=torch.int64, device=self._device
            )
        # 2. Slice this batch
        return (
            self._stat_ones[:sample_count],
            self._stat_eq[:sample_count],
            self._stat_match[:sample_count],
        )

    def _topk_eq_view(self, sample_count: int, top_k: int) -> torch.Tensor:
        """Return a pinned [N, K] bool view for one top-k membership compare.

        Purpose:
            Grow the pinned top-k equality buffer when the batch is wider
            than the current rows, then return the view this compare writes.

        Strategy:
            Allocate a bool buffer of width ``_max_top_k`` when the stored
            row count is too small, then slice to ``sample_count`` by
            ``top_k``.

        Args:
            sample_count: Number of flattened labels in the batch.
            top_k: Leader top-k width to compare.

        Returns:
            Bool tensor of shape ``(sample_count, top_k)``.

        Flow:
            1. Grow scratch when needed
            2. Slice this compare
        """
        # 1. Grow scratch when needed
        if self._topk_eq.shape[0] < sample_count:
            self._topk_eq = torch.empty(
                (sample_count, self._max_top_k),
                dtype=torch.bool,
                device=self._device,
            )
        # 2. Slice this compare
        return self._topk_eq[:sample_count, :top_k]

    def _prepare_batch_stats(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        shared_topk: torch.Tensor | None,
    ) -> _BatchStatSlots:
        """Flatten indices once and scatter the target-class histogram.

        Purpose:
            Share one long pred/target pair, one target-class support
            histogram, and one ranked-index equality buffer across compute
            groups so later leader updates do not flatten labels, scatter
            ``ones`` onto targets again, or compare the same top-k indices
            to labels a second time.

        Strategy:
            Reduce top-1 indices from the shared top-k tensor or logits,
            keep already-long dtypes, slice pinned match slots, scatter
            the target histogram into the pinned support buffer, then write
            membership of each label in the shared top-k rows when that
            tensor is present.

        Args:
            preds: Model logits or already-reduced class indices.
            targets: Integer class labels.
            shared_topk: Optional top-k class indices already reduced from
                ``preds``.

        Returns:
            Predicted indices, target indices, ones, equality buffer, and
            match buffer for this batch. Also fills ``_stat_support`` and,
            when ``shared_topk`` is set, ``_topk_eq``.

        Flow:
            1. Reduce indices
            2. Keep already-long dtypes
            3. Slice match slots
            4. Scatter target support
            5. Write shared top-k equality
        """
        # 1. Reduce indices
        pred_idx, target_idx = _reduce_leader_pred_indices(
            preds,
            targets,
            top_k=1,
            shared_topk=shared_topk,
        )
        # 2. Keep already-long dtypes
        if pred_idx.dtype != torch.long:
            pred_idx = pred_idx.to(dtype=torch.long)
        if target_idx.dtype != torch.long:
            target_idx = target_idx.to(dtype=torch.long)
        # 3. Slice match slots
        ones, eq_buf, match = self._stat_slots(target_idx.numel())
        # 4. Scatter target support
        self._stat_support.zero_()
        self._stat_support.scatter_add_(0, target_idx, ones)
        # 5. Write shared top-k equality
        if shared_topk is not None:
            eq_view = self._topk_eq_view(target_idx.numel(), shared_topk.shape[1])
            torch.eq(shared_topk, target_idx.unsqueeze(1), out=eq_view)
        return pred_idx, target_idx, ones, eq_buf, match

    def _add_leader_counts(self, leader: Metric) -> None:
        """Add this batch's tp and fn into one leader and bump its update count.

        Purpose:
            Fold the pinned per-class true-positive buffer and the shared
            target-support histogram into the leader without a second
            class-length scratch copy.

        Strategy:
            Add true positives, add support into false negatives, subtract
            true positives from false negatives, then increment
            ``_update_count``.

        Args:
            leader: Compute-group leader whose state stores confusion counts.

        Returns:
            None. Updates ``leader`` confusion state and ``_update_count``.

        Flow:
            1. Resolve leader tensors
            2. Add tp and support-minus-tp into fn
            3. Increment update count
        """
        # 1. Resolve leader tensors
        tp_state = leader.tp
        fn_state = leader.fn
        if not isinstance(tp_state, torch.Tensor) or not isinstance(
            fn_state, torch.Tensor
        ):
            message = "leader confusion state must be a tensor"
            raise TypeError(message)
        # 2. Add tp and support-minus-tp into fn
        tp_state.add_(self._stat_tp)
        fn_state.add_(self._stat_support)
        fn_state.sub_(self._stat_tp)
        # 3. Increment update count
        object.__setattr__(leader, "_update_count", leader.update_count + 1)

    def _accumulate_topk_hits(
        self,
        leader: Metric,
        batch_stats: _BatchStatSlots,
        shared_topk: torch.Tensor | None,
        top_k: int,
    ) -> None:
        """Fold top-k hits into tp and fn without reconstructing predictions.

        Purpose:
            Update a weighted top-k accuracy leader from label membership in
            the shared top-k index tensor, without a pred vector, a second
            target histogram, or unused false-positive scatter.

        Strategy:
            Prefer membership already written into the pinned [N, K]
            equality buffer, reduced with a pinned [N] any-slot. Fall back
            to exact index equality when that tensor is missing. Scatter
            hits into the pinned true-positive buffer, then add that buffer
            and support-minus-tp into the leader.

        Args:
            leader: Top-k accuracy leader whose state stores confusion counts.
            batch_stats: Shared pred indices, target indices, ones, equality
                buffer, and match buffer for this batch.
            shared_topk: Optional top-k class indices already reduced from
                logits.
            top_k: Leader top-k width.

        Returns:
            None. Updates ``leader`` tp, fn, and ``_update_count``.

        Flow:
            1. Reduce top-k membership
            2. Scatter tp
            3. Add tp and support-minus-tp into leader state
        """
        # 1. Reduce top-k membership
        pred_idx, target_idx, _ones, eq_buf, match = batch_stats
        if shared_topk is not None:
            eq_view = self._topk_eq[: target_idx.numel(), :top_k]
            torch.any(eq_view, dim=1, out=eq_buf)
            match.copy_(eq_buf)
        else:
            torch.eq(pred_idx, target_idx, out=eq_buf)
            match.copy_(eq_buf)
        # 2. Scatter tp
        self._stat_tp.zero_()
        self._stat_tp.scatter_add_(0, target_idx, match)
        # 3. Add tp and support-minus-tp into leader state
        self._add_leader_counts(leader)

    def _accumulate_stat_scores(
        self,
        leader: Metric,
        batch_stats: _BatchStatSlots,
        shared_topk: torch.Tensor | None,
    ) -> None:
        """Fold one batch into confusion counts without ``bincount`` or one-hot.

        Purpose:
            Build the same per-class tp/fp/fn as torchmetrics' multiclass
            accuracy, precision, recall, and F1 path, using a pinned
            true-positive scatter buffer, a shared target-support histogram,
            and pinned match slots.

        Strategy:
            Send a top-k accuracy leader through membership hits. For top-1,
            reuse column 0 of the shared top-k equality buffer when that
            tensor was reduced from logits, otherwise write equality into
            pinned match slots. Scatter matches into the pinned
            true-positive buffer, scatter the predicted-class histogram
            into the leader's false-positive tensor, then add true
            positives and support-minus-tp into the leader. Multiclass
            compute does not fill true negatives.

        Args:
            leader: Compute-group leader whose state stores confusion counts.
            batch_stats: Shared pred indices, target indices, ones, equality
                buffer, and match buffer for this batch.
            shared_topk: Optional top-k class indices already reduced from
                logits, shared across compute groups.

        Returns:
            None. Updates ``leader`` confusion state and ``_update_count``.

        Flow:
            1. Dispatch top-k hits
            2. Write top-1 match
            3. Scatter tp and fold false positives into the leader
            4. Add tp and support-minus-tp into the leader
        """
        # 1. Dispatch top-k hits
        pred_idx, target_idx, ones, eq_buf, match = batch_stats
        top_k = getattr(leader, "top_k", 1)
        if top_k is not None and int(top_k) > 1:
            self._accumulate_topk_hits(
                leader,
                batch_stats,
                shared_topk,
                int(top_k),
            )
            return
        # 2. Write top-1 match
        if shared_topk is not None:
            match.copy_(self._topk_eq[: target_idx.numel(), 0])
        else:
            torch.eq(pred_idx, target_idx, out=eq_buf)
            match.copy_(eq_buf)
        # 3. Scatter tp and fold false positives into the leader
        self._stat_tp.zero_()
        self._stat_tp.scatter_add_(0, target_idx, match)
        fp_state = leader.fp
        if not isinstance(fp_state, torch.Tensor):
            message = "leader confusion state must be a tensor"
            raise TypeError(message)
        fp_state.scatter_add_(0, pred_idx, ones)
        fp_state.sub_(self._stat_tp)
        # 4. Add tp and support-minus-tp into the leader
        self._add_leader_counts(leader)

    def update_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate classification metrics from a batch.

        Purpose:
            Fold one batch into grouped classification state without repeating
            MetricCollection's sibling-state rebind, a ``C * C`` confusion
            buffer, top-1 ``bincount``, a second logits ``argmax``, a second
            target-class histogram scatter, a second index flatten, top-k
            one-hot stat scores, a reconstructed top-k prediction vector,
            unused top-k false-positive scatter, a class-length false-negative
            scratch copy, a fresh top-k equality tensor, a second top-1
            equality against the same ranked indices, warmed dtype recasts
            of index and match tensors, a no-op 1-D index reshape, or
            autograd tracking on the shared logits top-k.

        Strategy:
            Clear cached computes on every classification metric, then under
            inference mode reduce logits to one shared top-k index tensor
            when the batch still has a class dimension, keep already-flat
            indices, scatter target support once, write one ranked-index
            equality buffer, accumulate each group's confusion from that
            shared state with pinned per-class scatter-add, then copy the
            leader update count onto siblings that already share tensor state.

        Args:
            preds: Model logits or probabilities.
            targets: Integer class labels.

        Flow:
            1. Invalidate cached computes
            2. Reduce logits once under inference mode
            3. Prepare shared indices, target support, and top-k equality
            4. Update leaders and sibling counts
        """
        # 1. Invalidate cached computes
        for metric in self._classification_metrics:
            object.__setattr__(metric, "_computed", None)
        with torch.inference_mode():
            # 2. Reduce logits once under inference mode
            shared_topk: torch.Tensor | None = None
            if preds.ndim == targets.ndim + 1 and self._max_top_k > 1:
                shared_topk = preds.topk(k=self._max_top_k, dim=1).indices
            # 3. Prepare shared indices, target support, and top-k equality
            batch_stats = self._prepare_batch_stats(preds, targets, shared_topk)
            # 4. Update leaders and sibling counts
            for group in self._classification_groups:
                leader = group[0]
                self._accumulate_stat_scores(leader, batch_stats, shared_topk)
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
