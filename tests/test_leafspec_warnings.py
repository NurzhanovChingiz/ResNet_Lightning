"""Contracts for silencing Torch's LeafSpec pytree deprecation."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from lightning.pytorch.utilities._pytree import _tree_flatten
from torch.utils._pytree import LeafSpec

from utils.leafspec_warnings import (
    LEAFSPEC_DEPRECATION_MESSAGE,
    silence_leafspec_deprecation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lightning_tree_flatten_emits_leafspec_without_filter() -> None:
    """Lightning leaf flattening still constructs deprecated LeafSpec."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        _tree_flatten(1)
    messages = [str(item.message) for item in caught]
    assert LEAFSPEC_DEPRECATION_MESSAGE in messages


def test_silence_hides_leafspec_future_warning() -> None:
    """LeafSpec construction must not raise after the ignore filter is installed."""
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter("error", FutureWarning)
        silence_leafspec_deprecation()
        LeafSpec()


def test_silence_hides_lightning_tree_flatten_warning() -> None:
    """The training-path flatten helper must stay quiet after the filter."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        silence_leafspec_deprecation()
        _tree_flatten({"a": 1, "b": (2, 3)})
    messages = [str(item.message) for item in caught]
    assert LEAFSPEC_DEPRECATION_MESSAGE not in messages


def test_silence_keeps_unrelated_future_warnings() -> None:
    """Other FutureWarnings must remain visible."""
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter("error", FutureWarning)
        silence_leafspec_deprecation()
        with pytest.raises(FutureWarning, match="unrelated"):
            warnings.warn("unrelated", FutureWarning, stacklevel=2)


def test_prepare_runtime_silences_leafspec_before_gpu_setup() -> None:
    """prepare_runtime must ignore LeafSpec before probing the GPU."""
    source = (REPO_ROOT / "config.py").read_text(encoding="utf-8")
    start = source.index("def prepare_runtime")
    body = source[start : source.index("def ensure_download_path")]
    quiet_call = body.index("silence_leafspec_deprecation()")
    gpu_call = body.index("resolve_gpu_runtime()")
    assert quiet_call < gpu_call
