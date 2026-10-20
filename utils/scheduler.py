"""Build Lightning-compatible learning-rate scheduler configs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
    StepLR,
)

if TYPE_CHECKING:
    from torch.optim import Optimizer


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int,
    eta_min: float,
    *,
    step_size: int,
    gamma: float,
    max_lr: float,
    steps_per_epoch: int | None = None,
) -> dict[str, Any]:
    """Build a learning-rate scheduler and return a Lightning-compatible dict.

    All values must be provided by the caller (config.py).

    Purpose:
        Map a caller-selected scheduler name onto a Lightning dict the trainer
        can consume.

    Strategy:
        Normalize the type name, then construct the matching PyTorch scheduler
        and wrap it with Lightning interval and frequency, failing closed on
        an unknown name.

    Supported scheduler_type values:

    - ``cosine_warmup``: Linear warmup then CosineAnnealingLR
    - ``cosine``: CosineAnnealingLR with no warmup
    - ``step``: StepLR
    - ``onecycle``: OneCycleLR (requires steps_per_epoch)

    Args:
        optimizer: Optimizer whose learning rate is scheduled.
        scheduler_type: One of the supported scheduler names.
        epochs: Total training epochs.
        warmup_epochs: Warmup length for ``cosine_warmup``.
        eta_min: Minimum learning rate for cosine schedules.
        step_size: Epoch period for StepLR.
        gamma: StepLR multiplicative decay.
        max_lr: Peak learning rate for OneCycleLR.
        steps_per_epoch: Optimizer steps per epoch, required by OneCycleLR.

    Returns:
        Lightning scheduler dict with ``scheduler``, ``interval``, and
        ``frequency``.

    Flow:
        1. Normalize scheduler type
        2. Construct and return the requested Lightning scheduler dict
    """
    # 1. Normalize scheduler type
    scheduler_type = scheduler_type.lower()

    # 2. Construct and return the requested Lightning scheduler dict
    if scheduler_type == "cosine_warmup":
        warmup = LinearLR(
            optimizer,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=eta_min,
        )
        warmup_cosine = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return {"scheduler": warmup_cosine, "interval": "epoch", "frequency": 1}

    if scheduler_type == "cosine":
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )
        return {"scheduler": cosine_scheduler, "interval": "epoch", "frequency": 1}

    if scheduler_type == "step":
        step_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return {"scheduler": step_scheduler, "interval": "epoch", "frequency": 1}

    if scheduler_type == "onecycle":
        if steps_per_epoch is None:
            message = "steps_per_epoch is required for 'onecycle' scheduler."
            raise ValueError(message)
        onecycle_scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        return {"scheduler": onecycle_scheduler, "interval": "step", "frequency": 1}

    message = (
        f"Unknown scheduler_type '{scheduler_type}'. "
        "Choose from: cosine_warmup, cosine, step, onecycle."
    )
    raise ValueError(message)
