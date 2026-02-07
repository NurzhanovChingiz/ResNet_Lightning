from utils.clear_gpu import clear_memory
from utils.set_seed import set_seed
from utils.scheduler import get_scheduler
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import os
import torch
from torch.backends import cudnn
from model import ResNet18
from typing import Any



class CFG:
    SEED: int = 137
    set_seed(SEED)
    clear_memory()
    nvidia_smi_available = os.system("nvidia-smi") == 0 
    if nvidia_smi_available:
        ACCELERATOR: str = "cuda"
        DEVICE: Any = torch.device("cuda")
        print(f"Using device: {DEVICE}")
        cudnn.benchmark: bool = True
    else:
        ACCELERATOR: str = "cpu"
        DEVICE: Any = torch.device("cpu")
        print(f"Using device: {DEVICE}")
        cudnn.benchmark: bool = False

    # Dataset parameters
    # https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k
    DATASET_ID: str = "vitaliykinakh/stable-imagenet1k"
    DOWNLOAD_PATH: str = "data/raw/"
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)
    DATA: str = os.path.join(DOWNLOAD_PATH, 'imagenet1k')
    
    # Data split parameters
    TEST_SIZE: float = 0.2
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
    COMPILE_MODEL: bool = False
    if COMPILE_MODEL:
        try:
            MODEL: Any = torch.compile(MODEL,
                                  fullgraph=True,
                                #   mode="max-autotune"
                                  )
        except Exception as e:
            print(f"Warning: torch.compile() disabled (fallback to eager). Reason: {e}")
            
    # Metrics      
    PRINT_METRICS_TO_TERMINAL: bool = True
    TASK: str = "multiclass"  
    NUM_CLASSES: int = 1000 
    
    # Image parameters
    IMG_SIZE: int = 128
    CROP_SIZE: int = IMG_SIZE-32 #224
    
    # Training parameters
    BATCH: int = 128
    EPOCHS: int = 20
    LR: float = 0.001
    EPS: float = 1e-10
    WEIGHT_DECAY: float = 0.01
    MIN_DELTA: float = 0.001
    PATIENCE: int = 5
    NUM_WORKERS: int = (os.cpu_count() or 1) // 2 if nvidia_smi_available else 0
    print(f"Using {NUM_WORKERS} DataLoader workers.")
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = False

    # Precision
    AMP: bool = True
    PRECISION: str = "32-true"
    # Learning rate scheduler
    SCHEDULER_TYPE: str = "cosine_warmup"  # Options: cosine_warmup, cosine, step, onecycle
    WARMUP_EPOCHS: int = 5
    ETA_MIN: float = 1e-6
    STEP_SIZE: int = 10
    GAMMA: float = 0.1
    MAX_LR: float = LR
    

    # Loss function and optimizer
    LOSS_FN: Any = torch.nn.CrossEntropyLoss()
    OPTIMIZER: Any = torch.optim.AdamW(MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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