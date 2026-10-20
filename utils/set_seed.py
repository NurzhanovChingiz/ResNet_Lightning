"""Seed Python, NumPy, PyTorch, and Lightning random number generators."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch
from lightning.pytorch import seed_everything

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed every RNG used by training.

    Args:
        seed: Value applied to Python, NumPy, PyTorch, and Lightning.
    """
    seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    LOGGER.info("Random Seed : %s", seed)
