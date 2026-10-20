"""Release cached GPU memory used by PyTorch."""

from __future__ import annotations

import gc
import logging
import time

import torch

LOGGER = logging.getLogger(__name__)


def clear_memory(*, verbose: bool = True) -> None:
    """Synchronize CUDA, empty the cache, and run garbage collection.

    Args:
        verbose: If True, log how long the cleanup took.
    """
    started_at = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        torch.cuda.empty_cache()

    gc.collect()

    if verbose:
        elapsed_seconds = time.perf_counter() - started_at
        LOGGER.info("Cleared memory. Time taken was %f secs", elapsed_seconds)
