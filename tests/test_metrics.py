"""Contracts for split metric tracking without a global config object."""

from __future__ import annotations

import math

import pytest
import torch

from utils.metrics import MetricsTracker, NoNanMeanMetric


def test_no_nan_mean_metric_does_not_raise_on_nan() -> None:
    """NaN losses are accepted instead of aborting the mean metric."""
    metric = NoNanMeanMetric()
    metric.update(float("nan"))
    value = metric.compute()
    assert torch.isnan(value)


def test_metrics_tracker_computes_mean_loss() -> None:
    """A tracker constructed with explicit task settings records mean loss."""
    tracker = MetricsTracker(
        task="multiclass",
        num_classes=10,
        device=torch.device("cpu"),
    )
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
