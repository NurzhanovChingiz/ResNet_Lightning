"""Contracts for model parameter summaries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from torch import nn

from utils.summary import summary

if TYPE_CHECKING:
    import pytest


def test_summary_logs_parameter_counts(caplog: pytest.LogCaptureFixture) -> None:
    """summary() logs total and trainable parameter counts."""
    model = nn.Linear(4, 2)
    expected_trainable = sum(parameter.numel() for parameter in model.parameters())
    with caplog.at_level(logging.INFO, logger="utils.summary"):
        summary(model)
    assert f"Total parameters: {expected_trainable:,}" in caplog.text
    assert f"Trainable parameters: {expected_trainable:,}" in caplog.text
