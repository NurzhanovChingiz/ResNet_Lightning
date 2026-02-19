from utils.clear_gpu import clear_memory
from utils.set_seed import set_seed, seed_worker, get_generator
from utils.scheduler import get_scheduler
from torchvision.transforms import v2  # type: ignore[import-untyped]
from torchvision.transforms.functional import InterpolationMode  # type: ignore[import-untyped]
import os
import torch
from torch.backends import cudnn
from model import ResNet18
from typing import Any, Literal



class CFG:
    SEED: int = 137
    set_seed(SEED)
    clear_memory()

    ACCELERATOR: str = "auto"
    DEVICE: Any = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = False
    
    # Dataset parameters
    # https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k
    DATASET_ID: str = "vitaliykinakh/stable-imagenet1k"
    DOWNLOAD_PATH: str = "data/raw/"
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)
    DATA: str = os.path.join(DOWNLOAD_PATH, 'imagenet1k')
    
    # Data split parameters
    TEST_SIZE: float = 0.3
    VAL_SIZE: float = 0.15
    STRATIFY: bool = True
    
    # Directories
    MAIN_DIR: str = os.getcwd()
    BASE_DIR: str = f"{MAIN_DIR}/data" # Must be pre-downloaded
    
    # Model parameters
    MODEL_NAME: str = "ResNet18"
    PRETRAIN: bool = False
    # Model
    MODEL: Any = ResNet18(num_classes=1000, pretrained=PRETRAIN)
    
    # Compile
    # NOTE: torch.compile() is disabled because ROCm/HIP has limited support for it
    COMPILE_MODEL: bool = True
    if COMPILE_MODEL:
        try:
            MODEL = torch.compile(MODEL,
                                  fullgraph=True,
                                #   mode="max-autotune"
                                  )
        except Exception as e:
            print(f"Warning: torch.compile() disabled (fallback to eager). Reason: {e}")
            
    # Metrics      
    PRINT_METRICS_TO_TERMINAL: bool = True
    TASK: Literal["binary", "multiclass", "multilabel"] = "multiclass"  
    NUM_CLASSES: int = 1000 
    
    # Image parameters
    IMG_SIZE: int = 256
    CROP_SIZE: int = IMG_SIZE-32 #224
    
    # Training parameters
    BATCH: int = 128
    EPOCHS: int = 50
    LR: float = 0.003
    EPS: float = 1e-10
    WEIGHT_DECAY: float = 0.01
    MIN_DELTA: float = 0.001
    PATIENCE: int = 4
    
    # DataLoader parameters
    NUM_WORKERS: int = os.cpu_count() or 1
    print(f"Using {NUM_WORKERS} DataLoader workers.")
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    SEED_WORKER: Any = seed_worker
    GENERATOR: Any = get_generator(SEED)
    
    # Precision
    AMP: bool = True
    PRECISION: Literal["transformer-engine", "transformer-engine-float16", "16-true", "16-mixed", "bf16-true", "bf16-mixed", "32-true", "64-true", "64", "32", "16", "bf16"] = "bf16-true"
    
    # Learning rate scheduler
    SCHEDULER_TYPE: str = "cosine_warmup"  # Options: cosine_warmup, cosine, step, onecycle
    WARMUP_EPOCHS: int = 5
    ETA_MIN: float = 1e-6
    STEP_SIZE: int = 10
    GAMMA: float = 0.1
    MAX_LR: float = LR
    
    # Loss function and optimizer
    LOSS_FN: Any = torch.nn.CrossEntropyLoss()
    OPTIMIZER: Any = torch.optim.AdamW(MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=EPS)
    SCHEDULER: Any = get_scheduler(
        OPTIMIZER,
        scheduler_type=SCHEDULER_TYPE,
        epochs=EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        eta_min=ETA_MIN,
        step_size=STEP_SIZE,
        gamma=GAMMA,
        max_lr=MAX_LR,
    )
    
    TRAIN_TRANSFORM: Any = v2.Compose([
        v2.Resize(IMG_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True),
        v2.CenterCrop(CROP_SIZE),
        v2.ToImage(),  # ensures Tensor image; works for PIL/ndarray too
        v2.ToDtype(torch.float, scale=True),  # like convert_image_dtype(..., torch.float)
        v2.Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)),
    ])
        
    TEST_TRANSFORM: Any = v2.Compose([
        v2.Resize(IMG_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True),
        v2.CenterCrop(CROP_SIZE),
        v2.ToImage(),  # ensures Tensor image; works for PIL/ndarray too
        v2.ToDtype(torch.float, scale=True),  # like convert_image_dtype(..., torch.float)
        v2.Normalize(mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)),
    ])