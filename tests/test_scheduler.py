"""Contracts for learning-rate scheduler construction."""

from __future__ import annotations

import pytest
import torch

from utils.scheduler import get_scheduler


def _optimizer() -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([parameter], lr=0.1)


def test_unknown_scheduler_type_raises() -> None:
    """An unsupported scheduler name is rejected before construction."""
    with pytest.raises(ValueError, match="Unknown scheduler_type"):
        get_scheduler(
            _optimizer(),
            "nope",
            epochs=2,
            warmup_epochs=1,
            eta_min=1e-6,
            step_size=1,
            gamma=0.1,
            max_lr=0.1,
        )


def test_onecycle_requires_steps_per_epoch() -> None:
    """OneCycleLR construction fails closed without steps_per_epoch."""
    with pytest.raises(ValueError, match="steps_per_epoch is required"):
        get_scheduler(
            _optimizer(),
            "onecycle",
            epochs=2,
            warmup_epochs=1,
            eta_min=1e-6,
            step_size=1,
            gamma=0.1,
            max_lr=0.1,
        )


def test_cosine_returns_epoch_interval() -> None:
    """A cosine scheduler is returned on the epoch interval."""
    config = get_scheduler(
        _optimizer(),
        "cosine",
        epochs=2,
        warmup_epochs=1,
        eta_min=1e-6,
        step_size=1,
        gamma=0.1,
        max_lr=0.1,
    )
    assert config["interval"] == "epoch"
    assert config["frequency"] == 1
