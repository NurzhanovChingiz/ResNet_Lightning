"""Contracts for GPU selection and CSV-only training logs."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from utils.device import (
    disable_cudnn_benchmark,
    enable_medium_float32_matmul_precision,
    resolve_gpu_runtime,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_gpu_runtime_returns_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A visible GPU yields the CUDA accelerator pair used by ROCm."""
    monkeypatch.setattr("utils.device.torch.cuda.is_available", lambda: True)
    accelerator, device = resolve_gpu_runtime()
    assert accelerator == "cuda"
    assert device == torch.device("cuda")


def test_resolve_gpu_runtime_raises_when_gpu_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training must not fall back to CPU when PyTorch sees no GPU."""
    monkeypatch.setattr("utils.device.torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="No GPU is visible to PyTorch"):
        resolve_gpu_runtime()


def test_enable_medium_float32_matmul_precision_sets_medium() -> None:
    """The GPU runtime must select medium float32 matmul precision."""
    previous = torch.get_float32_matmul_precision()
    try:
        enable_medium_float32_matmul_precision()
        assert torch.get_float32_matmul_precision() == "medium"
    finally:
        torch.set_float32_matmul_precision(previous)


def test_config_uses_torch_gpu_probe_not_nvidia_smi() -> None:
    """config.py must require resolve_gpu_runtime instead of nvidia-smi."""
    source = (REPO_ROOT / "config.py").read_text(encoding="utf-8")
    assert "nvidia-smi" not in source
    assert "resolve_gpu_runtime" in source


def test_config_enables_medium_float32_matmul_precision() -> None:
    """config.py must apply medium float32 matmul precision on the GPU path."""
    source = (REPO_ROOT / "config.py").read_text(encoding="utf-8")
    assert "enable_medium_float32_matmul_precision" in source


def test_disable_cudnn_benchmark_sets_false() -> None:
    """Convolution autotune must stay off so gfx1151 MIOpen can find an algorithm."""
    previous = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = True
        disable_cudnn_benchmark()
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.benchmark = previous


def test_config_disables_cudnn_benchmark() -> None:
    """config.py must disable convolution autotune instead of enabling it."""
    source = (REPO_ROOT / "config.py").read_text(encoding="utf-8")
    assert "disable_cudnn_benchmark" in source
    assert "cudnn.benchmark = True" not in source


def test_train_script_uses_csv_logger_only() -> None:
    """train.py must log with CSVLogger and not construct TensorBoardLogger."""
    source = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    assert "TensorBoardLogger" not in source
    assert "CSVLogger" in source


def test_train_script_releases_dataloaders_after_training_error() -> None:
    """train.py must clean up in finally and let training exceptions propagate."""
    source = (REPO_ROOT / "train.py").read_text(encoding="utf-8")
    assert "except Exception" not in source
    assert "traceback.print_exc()" not in source
    assert "try:" in source
    assert "finally:" in source
