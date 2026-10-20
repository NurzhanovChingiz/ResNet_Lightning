"""Import-time training configuration and named setup operations."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from model import ResNet18
from utils.clear_gpu import clear_memory
from utils.device import (
    disable_cudnn_benchmark,
    enable_medium_float32_matmul_precision,
    resolve_gpu_runtime,
)
from utils.leafspec_warnings import silence_leafspec_deprecation
from utils.rocm_kernel_logs import (
    disable_miopen_winograd_fury_solvers,
    silence_rocm_kernel_logs,
    use_miopen_fast_find_mode,
)
from utils.scheduler import get_scheduler
from utils.set_seed import set_seed

if TYPE_CHECKING:
    from typing import Literal

    from utils.metrics import MetricTask

    PrecisionName = Literal[
        "transformer-engine",
        "transformer-engine-float16",
        "16-true",
        "16-mixed",
        "bf16-true",
        "bf16-mixed",
        "32-true",
        "64-true",
        "64",
        "32",
        "16",
        "bf16",
    ]

LOGGER = logging.getLogger(__name__)


def prepare_runtime(seed: int) -> tuple[str, torch.device]:
    """Prepare process-wide GPU runtime before the rest of config is built.

    Purpose:
        Initialize seeding, GPU memory, device selection, and GPU math
        settings for the training process.

    Strategy:
        Quiet ROCm CK host dumps, disable MIOpen Winograd Fury solvers,
        switch MIOpen to FAST find, ignore the LeafSpec pytree
        deprecation, apply the seed and clear cached GPU memory, resolve
        the Lightning accelerator pair, log the selected device, then
        enable medium float32 matmul precision and disable cuDNN/MIOpen
        convolution autotune.

    Args:
        seed: Integer seed stored on ``CFG.SEED``.

    Returns:
        Accelerator name and torch device from ``resolve_gpu_runtime``.

    Flow:
        1. Quiet ROCm kernel host logs
        2. Disable Winograd Fury convolution solvers
        3. Use MIOpen FAST find mode
        4. Ignore LeafSpec pytree deprecation
        5. Seed and clear memory
        6. Resolve and announce GPU runtime
        7. Enable GPU math settings
        8. Return accelerator and device
    """
    # 1. Quiet ROCm kernel host logs
    silence_rocm_kernel_logs()
    # 2. Disable Winograd Fury convolution solvers
    disable_miopen_winograd_fury_solvers()
    # 3. Use MIOpen FAST find mode
    use_miopen_fast_find_mode()
    # 4. Ignore LeafSpec pytree deprecation
    silence_leafspec_deprecation()
    # 5. Seed and clear memory
    set_seed(seed)
    clear_memory()
    # 6. Resolve and announce GPU runtime
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    accelerator, device = resolve_gpu_runtime()
    LOGGER.info("Using device: %s", device)
    # 7. Enable GPU math settings
    enable_medium_float32_matmul_precision()
    disable_cudnn_benchmark()
    # 8. Return accelerator and device
    return accelerator, device


def ensure_download_path(path: str) -> None:
    """Create ``path`` when the dataset download directory is missing.

    Args:
        path: Filesystem directory that should exist for dataset files.

    Returns:
        None. Creates the directory in place when it is absent.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def build_resnet18(
    num_classes: int,
    *,
    pretrained: bool,
    compile_model: bool,
) -> Any:
    """Construct ResNet18 and optionally compile it.

    Purpose:
        Produce the trainable ResNet18 instance used as ``CFG.MODEL``.

    Strategy:
        Build ResNet18 with the requested class count and pretrained flag.
        When compile is requested, try ``torch.compile`` and keep the eager
        model if compilation raises ``RuntimeError``.

    Args:
        num_classes: Classification head width.
        pretrained: Whether to load ImageNet pretrained weights.
        compile_model: Whether to attempt ``torch.compile``.

    Returns:
        The eager or compiled ResNet18 module.

    Flow:
        1. Construct ResNet18
        2. Optionally compile the model
        3. Return the model
    """
    # 1. Construct ResNet18
    model: Any = ResNet18(num_classes=num_classes, pretrained=pretrained)
    # 2. Optionally compile the model
    # NOTE: torch.compile() is disabled because ROCm/HIP has limited support for it
    if compile_model:
        try:
            model = torch.compile(
                model,
                fullgraph=True,
            )
        except RuntimeError as exc:
            LOGGER.warning(
                "torch.compile() disabled (fallback to eager). Reason: %s",
                exc,
            )
    # 3. Return the model
    return model


def build_optimization_stack(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int,
    eta_min: float,
    step_size: int,
    gamma: float,
    max_lr: float,
) -> tuple[Any, Any, Any]:
    """Build the loss, optimizer, and scheduler used for training.

    Purpose:
        Assemble the optimization objects that consume ``CFG.MODEL``.

    Strategy:
        Instantiate cross-entropy loss, wrap model parameters in AdamW,
        then bind the configured learning-rate scheduler.

    Args:
        model: Trainable module whose parameters are optimized.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        scheduler_type: Scheduler name forwarded to ``get_scheduler``.
        epochs: Total training epochs.
        warmup_epochs: Warmup length for cosine-warmup schedules.
        eta_min: Minimum learning rate for cosine schedules.
        step_size: Epoch period for StepLR.
        gamma: StepLR multiplicative decay.
        max_lr: Peak learning rate for OneCycleLR.

    Returns:
        Loss, optimizer, and Lightning-compatible scheduler dict.

    Flow:
        1. Instantiate the classification loss
        2. Construct AdamW on model parameters
        3. Bind the configured scheduler
        4. Return the optimization triple
    """
    # 1. Instantiate the classification loss
    loss_fn: Any = torch.nn.CrossEntropyLoss()
    # 2. Construct AdamW on model parameters
    optimizer: Any = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # 3. Bind the configured scheduler
    scheduler: Any = get_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        eta_min=eta_min,
        step_size=step_size,
        gamma=gamma,
        max_lr=max_lr,
    )
    # 4. Return the optimization triple
    return loss_fn, optimizer, scheduler


def build_image_transform(img_size: int, crop_size: int) -> Any:
    """Build the shared train/test image transform pipeline.

    Purpose:
        Produce the torchvision v2 pipeline assigned to both
        ``CFG.TRAIN_TRANSFORM`` and ``CFG.TEST_TRANSFORM``.

    Strategy:
        Resize and center-crop, convert to a scaled float tensor, then
        normalize with ImageNet mean and standard deviation.

    Args:
        img_size: Square resize length in pixels.
        crop_size: Square center-crop length in pixels.

    Returns:
        A composed torchvision v2 transform.

    Flow:
        1. Resize and center-crop the image
        2. Convert to a scaled float tensor
        3. Normalize with ImageNet statistics
    """
    return v2.Compose(
        [
            # 1. Resize and center-crop the image
            v2.Resize(
                img_size, interpolation=InterpolationMode.BILINEAR, antialias=True
            ),
            v2.CenterCrop(crop_size),
            # 2. Convert to a scaled float tensor
            v2.ToImage(),  # ensures Tensor image; works for PIL/ndarray too
            v2.ToDtype(
                torch.float, scale=True
            ),  # like convert_image_dtype(..., torch.float)
            # 3. Normalize with ImageNet statistics
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class CFG:
    """Hold training hyperparameters and invoke import-time setup operations."""

    SEED: int = 137
    ACCELERATOR, DEVICE = prepare_runtime(SEED)
    PERSISTENT_WORKERS = True

    # Dataset parameters
    # https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k
    DATASET_ID: str = "vitaliykinakh/stable-imagenet1k"
    DOWNLOAD_PATH: str = "data/raw/"
    ensure_download_path(DOWNLOAD_PATH)
    DATA: str = str(Path(DOWNLOAD_PATH) / "imagenet1k")

    # Data split parameters
    TEST_SIZE: float = 0.2
    VAL_SIZE: float = 0.15
    STRATIFY: bool = True

    # Directories
    MAIN_DIR: str = str(Path.cwd())
    BASE_DIR: str = f"{MAIN_DIR}/data"  # Must be pre-downloaded

    # Model parameters
    MODEL_NAME: str = "ResNet18"
    PRETRAIN: bool = False
    COMPILE_MODEL: bool = False
    MODEL: Any = build_resnet18(1000, pretrained=PRETRAIN, compile_model=COMPILE_MODEL)

    # Metrics
    PRINT_METRICS_TO_TERMINAL: bool = True
    TASK: MetricTask = "multiclass"
    NUM_CLASSES: int = 1000

    # Image parameters
    IMG_SIZE: int = 128
    CROP_SIZE: int = IMG_SIZE - 32  # 224

    # Training parameters
    BATCH: int = 128
    EPOCHS: int = 10
    LR: float = 0.003
    EPS: float = 1e-10
    WEIGHT_DECAY: float = 0.01
    MIN_DELTA: float = 0.001
    PATIENCE: int = 5
    NUM_WORKERS: int = os.cpu_count() or 1
    LOGGER.info("Using %s DataLoader workers.", NUM_WORKERS)
    PIN_MEMORY: bool = True

    # Precision
    AMP: bool = True
    PRECISION: PrecisionName = "bf16-mixed"
    # Learning rate scheduler
    SCHEDULER_TYPE: str = (
        "cosine_warmup"  # Options: cosine_warmup, cosine, step, onecycle
    )
    WARMUP_EPOCHS: int = 5
    ETA_MIN: float = 1e-6
    STEP_SIZE: int = 10
    GAMMA: float = 0.1
    MAX_LR: float = LR

    # Loss function and optimizer
    LOSS_FN, OPTIMIZER, SCHEDULER = build_optimization_stack(
        MODEL,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        scheduler_type=SCHEDULER_TYPE,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        eta_min=ETA_MIN,
        step_size=STEP_SIZE,
        gamma=GAMMA,
        max_lr=MAX_LR,
    )

    TRAIN_TRANSFORM: Any = build_image_transform(IMG_SIZE, CROP_SIZE)
    TEST_TRANSFORM: Any = build_image_transform(IMG_SIZE, CROP_SIZE)
