import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
    StepLR,
    OneCycleLR,
)


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int,
    eta_min: float,
    step_size: int,
    gamma: float,
    max_lr: float,
    steps_per_epoch: int | None = None,
) -> dict:
    """
    Build a learning-rate scheduler and return a Lightning-compatible dict.

    All values must be provided by the caller (config.py).

    Supported scheduler_type values
    --------------------------------
    - "cosine_warmup"  : Linear warmup → CosineAnnealingLR
    - "cosine"         : CosineAnnealingLR (no warmup)
    - "step"           : StepLR
    - "onecycle"       : OneCycleLR (requires steps_per_epoch)

    Returns
    -------
    dict  - {"scheduler": <scheduler>, "interval": "epoch" | "step", "frequency": 1}
            Ready to be returned from LightningModule.configure_optimizers().
    """
    scheduler_type = scheduler_type.lower()

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
        scheduler: LRScheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=eta_min,
        )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    elif scheduler_type == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for 'onecycle' scheduler.")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
        )
        return {"scheduler": scheduler, "interval": "step", "frequency": 1}

    else:
        raise ValueError(
            f"Unknown scheduler_type '{scheduler_type}'. "
            "Choose from: cosine_warmup, cosine, step, onecycle."
        )
