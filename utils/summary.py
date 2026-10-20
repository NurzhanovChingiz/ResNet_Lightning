"""Log a parameter-count summary for a PyTorch module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module

LOGGER = logging.getLogger(__name__)


def summary(model: Module) -> None:
    """Log parameter counts and an estimated float32 size for ``model``.

    Args:
        model: PyTorch module whose parameters are counted.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    total_params = trainable_params + non_trainable_params
    param_size = total_params * 4 / (1024**2)
    LOGGER.info("=" * 50)
    LOGGER.info("Model Information")
    LOGGER.info("=" * 50)
    LOGGER.info("Total parameters: %s", f"{total_params:,}")
    LOGGER.info("Trainable parameters: %s", f"{trainable_params:,}")
    LOGGER.info(
        "Non-trainable parameters: %s",
        f"{total_params - trainable_params:,}",
    )
    LOGGER.info("Estimated size: %.2f MB", param_size)
    LOGGER.info("=" * 50)
