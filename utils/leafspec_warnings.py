"""Keep Torch's LeafSpec pytree deprecation off the training console."""

from __future__ import annotations

import re
import warnings

LEAFSPEC_DEPRECATION_MESSAGE = (
    "`isinstance(treespec, LeafSpec)` is deprecated, "
    "use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead."
)


def silence_leafspec_deprecation() -> None:
    """Ignore Torch's LeafSpec pytree FutureWarning.

    Purpose:
        Stop Lightning's ``LeafSpec()`` construction from printing the
        deprecated ``isinstance(treespec, LeafSpec)`` warning.

    Strategy:
        Register a process-wide ignore filter for that exact FutureWarning
        message so other Lightning and Torch warnings stay visible.

    Returns:
        None. Updates the process warning filters.

    Flow:
        1. Ignore the LeafSpec deprecation FutureWarning
    """
    # 1. Ignore the LeafSpec deprecation FutureWarning
    warnings.filterwarnings(
        "ignore",
        message=re.escape(LEAFSPEC_DEPRECATION_MESSAGE),
        category=FutureWarning,
    )
