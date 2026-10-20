"""Require a PyTorch-visible GPU for training."""

from __future__ import annotations

import torch


def enable_medium_float32_matmul_precision() -> None:
    """Use Tensor Cores for float32 matrix multiplies at medium precision.

    Returns:
        None. Sets the process-wide PyTorch float32 matmul precision to medium.
    """
    torch.set_float32_matmul_precision("medium")


def disable_cudnn_benchmark() -> None:
    """Skip cuDNN/MIOpen convolution autotune for gfx1151.

    Purpose:
        Keep PyTorch from timing convolution solvers. On this Linux
        gfx1151 driver that search reports elapsed <= 0 and then raises
        ``miopenStatusUnknownError``.

    Strategy:
        Set ``torch.backends.cudnn.benchmark`` to False so Find keeps
        using FAST FindDb instead of EvaluateInvokers.

    Returns:
        None. Clears the process-wide cuDNN benchmark flag.
    """
    torch.backends.cudnn.benchmark = False


def resolve_gpu_runtime() -> tuple[str, torch.device]:
    """Return Lightning accelerator settings for a visible CUDA/ROCm GPU.

    Purpose:
        Fail closed when no GPU is visible so training cannot run on CPU.

    Strategy:
        Ask PyTorch whether a CUDA/ROCm device is available, then return the
        accelerator pair both backends expose as ``cuda``.

    Returns:
        Accelerator name and torch device for GPU training.

    Raises:
        RuntimeError: If PyTorch cannot see a GPU.

    Flow:
        1. Probe GPU visibility
        2. Return accelerator and device
    """
    # 1. Probe GPU visibility
    if not torch.cuda.is_available():
        message = (
            "No GPU is visible to PyTorch (torch.cuda.is_available() is False). "
            "This project requires a ROCm or CUDA GPU."
        )
        raise RuntimeError(message)
    # 2. Return accelerator and device
    return "cuda", torch.device("cuda")
