"""Contracts for split metric tracking without a global config object."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
import torch
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from utils.metrics import MetricsTracker, NoNanMeanMetric

if TYPE_CHECKING:
    from collections.abc import Callable

_CLASSIFICATION_METRIC_NAMES = (
    "accuracy_top1",
    "accuracy_top5",
    "precision",
    "recall",
    "f1_score",
)


def _cpu_tracker() -> MetricsTracker:
    """Build a 10-class multiclass tracker on CPU."""
    return MetricsTracker(
        task="multiclass",
        num_classes=10,
        device=torch.device("cpu"),
    )


def _cifar_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CIFAR-10-shaped logits and labels.

    Args:
        seed: Generator seed for the batch.

    Returns:
        Logits of shape ``(128, 10)`` and integer labels of shape ``(128,)``.
    """
    generator = torch.Generator().manual_seed(seed)
    preds = torch.randn((128, 10), generator=generator)
    targets = torch.randint(0, 10, (128,), generator=generator)
    return preds, targets


def test_no_nan_mean_metric_does_not_raise_on_nan() -> None:
    """NaN losses are accepted instead of aborting the mean metric."""
    metric = NoNanMeanMetric()
    metric.update(float("nan"))
    value = metric.compute()
    assert torch.isnan(value)


def test_update_loss_skips_per_batch_tensor_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warmed scalar loss update must not rebuild 0-dim tensors."""
    tracker = _cpu_tracker()
    tracker.update_loss(0.5)
    as_tensor = Mock(wraps=torch.as_tensor)
    ones_like = Mock(wraps=torch.ones_like)
    monkeypatch.setattr(torch, "as_tensor", as_tensor)
    monkeypatch.setattr(torch, "ones_like", ones_like)
    tracker.update_loss(0.25)
    assert as_tensor.call_count == 0
    assert ones_like.call_count == 0
    assert tracker.get_metric("avg_loss").compute().item() == pytest.approx(0.375)


def test_update_loss_skips_aligned_mean_recast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warmed scalar loss update must not recast already-aligned tensors."""
    tracker = _cpu_tracker()
    tracker.update_loss(0.5)
    broadcast_to = Mock(wraps=torch.broadcast_to)
    monkeypatch.setattr(torch, "broadcast_to", broadcast_to)
    tracker.update_loss(0.25)
    assert broadcast_to.call_count == 0
    assert tracker.get_metric("avg_loss").compute().item() == pytest.approx(0.375)


def test_metrics_tracker_computes_mean_loss() -> None:
    """A tracker constructed with explicit task settings records mean loss."""
    tracker = _cpu_tracker()
    logits = torch.zeros((2, 10))
    logits[0, 1] = 5.0
    logits[1, 2] = 5.0
    targets = torch.tensor([1, 2])
    tracker.update_metrics(logits, targets)
    tracker.update_loss(0.5)
    values = tracker.compute()
    assert values["avg_loss"] == pytest.approx(0.5)
    assert values["accuracy_top1"] == pytest.approx(1.0)
    assert math.isfinite(values["f1_score"])


def test_update_metrics_skips_duplicate_classification_updates_after_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second batch must not call update on every classification metric."""
    tracker = _cpu_tracker()
    warmup_preds, warmup_targets = _cifar_batch(0)
    tracker.update_metrics(warmup_preds, warmup_targets)
    update_counts = dict.fromkeys(_CLASSIFICATION_METRIC_NAMES, 0)
    for name in _CLASSIFICATION_METRIC_NAMES:
        metric = tracker.get_metric(name)
        original_update: Callable[..., object] = metric.update

        def wrapped(
            *args: object,
            metric_name: str = name,
            original: Callable[..., object] = original_update,
            **kwargs: object,
        ) -> object:
            update_counts[metric_name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(metric, "update", wrapped)
    preds, targets = _cifar_batch(1)
    tracker.update_metrics(preds, targets)
    observed_updates = sum(update_counts.values())
    assert observed_updates < len(_CLASSIFICATION_METRIC_NAMES)
    assert observed_updates >= 1


def test_update_metrics_skips_top1_bincount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A warmed top-1 batch must not call torch.bincount."""
    tracker = _cpu_tracker()
    warmup_preds, warmup_targets = _cifar_batch(0)
    tracker.update_metrics(warmup_preds, warmup_targets)
    bincount = Mock(wraps=torch.bincount)
    monkeypatch.setattr(torch, "bincount", bincount)
    preds, targets = _cifar_batch(1)
    tracker.update_metrics(preds, targets)
    assert bincount.call_count == 0
    values = tracker.compute()
    accuracy_top1 = Accuracy(
        task="multiclass", num_classes=10, average="weighted", top_k=1
    )
    accuracy_top1.update(warmup_preds, warmup_targets)
    accuracy_top1.update(preds, targets)
    assert values["accuracy_top1"] == pytest.approx(accuracy_top1.compute().item())


def test_update_metrics_skips_argument_validation() -> None:
    """Classification metrics must skip torchmetrics per-batch argument checks."""
    tracker = _cpu_tracker()
    for name in _CLASSIFICATION_METRIC_NAMES:
        assert vars(tracker.get_metric(name))["validate_args"] is False


def test_update_metrics_skips_collection_rebind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch update must not rebind MetricCollection sibling state."""
    tracker = _cpu_tracker()
    collection_updates = {"n": 0}
    original_update: Callable[..., object] = MetricCollection.update

    def wrapped(self: MetricCollection, *args: object, **kwargs: object) -> object:
        collection_updates["n"] += 1
        return original_update(self, *args, **kwargs)

    monkeypatch.setattr(MetricCollection, "update", wrapped)
    preds, targets = _cifar_batch(1)
    tracker.update_metrics(preds, targets)
    assert collection_updates["n"] == 0


def test_compute_matches_standalone_torchmetrics() -> None:
    """Grouped collection updates must match ungrouped torchmetrics values."""
    preds, targets = _cifar_batch(2)
    tracker = _cpu_tracker()
    tracker.update_metrics(preds, targets)
    tracker.update_loss(0.25)
    values = tracker.compute()
    accuracy_top1 = Accuracy(
        task="multiclass", num_classes=10, average="weighted", top_k=1
    )
    accuracy_top5 = Accuracy(
        task="multiclass", num_classes=10, average="weighted", top_k=5
    )
    precision = Precision(task="multiclass", num_classes=10, average="macro")
    recall = Recall(task="multiclass", num_classes=10, average="macro")
    f1_score = F1Score(task="multiclass", num_classes=10, average="macro")
    accuracy_top1.update(preds, targets)
    accuracy_top5.update(preds, targets)
    precision.update(preds, targets)
    recall.update(preds, targets)
    f1_score.update(preds, targets)
    assert values["avg_loss"] == pytest.approx(0.25)
    assert values["accuracy_top1"] == pytest.approx(accuracy_top1.compute().item())
    assert values["accuracy_top5"] == pytest.approx(accuracy_top5.compute().item())
    assert values["precision"] == pytest.approx(precision.compute().item())
    assert values["recall"] == pytest.approx(recall.compute().item())
    assert values["f1_score"] == pytest.approx(f1_score.compute().item())


def test_metrics_live_on_constructor_device() -> None:
    """Constructed modules stay on the tracker device instead of the host."""
    device = torch.device("cpu")
    tracker = _cpu_tracker()
    for name in ("avg_loss", *_CLASSIFICATION_METRIC_NAMES):
        assert tracker.get_metric(name).device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gpu_batch_updates_stay_on_constructor_device() -> None:
    """A CUDA batch must update in place on the constructor device."""
    device = torch.device("cuda", torch.cuda.current_device())
    tracker = MetricsTracker(
        task="multiclass",
        num_classes=10,
        device=device,
    )
    preds, targets = _cifar_batch(4)
    preds = preds.to(device)
    targets = targets.to(device)
    tracker.update_metrics(preds, targets)
    tracker.update_loss(torch.tensor(0.5, device=device))
    for name in ("avg_loss", *_CLASSIFICATION_METRIC_NAMES):
        assert tracker.get_metric(name).device == device
    values = tracker.compute()
    assert math.isfinite(values["avg_loss"])
    assert math.isfinite(values["accuracy_top1"])


def test_get_metric_returns_live_modules_and_rejects_unknown_names() -> None:
    """Trainer name lookup stays valid and unknown names stay rejected."""
    tracker = _cpu_tracker()
    preds, targets = _cifar_batch(3)
    tracker.update_metrics(preds, targets)
    tracker.update_loss(0.5)
    values = tracker.compute()
    for name in ("avg_loss", *_CLASSIFICATION_METRIC_NAMES):
        assert tracker.get_metric(name).compute().item() == pytest.approx(values[name])
    with pytest.raises(KeyError, match="Unknown metric"):
        tracker.get_metric("missing")
