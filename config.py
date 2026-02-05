from utils.clear_gpu import clear_memory
from utils.set_seed import set_seed
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import os
import torch
from torch.backends import cudnn
from model import ResNet18
from typing import Any

# Workaround for ROCm/HIP kernel compatibility issues on newer AMD GPUs
# Try setting HSA_OVERRIDE_GFX_VERSION if you get "device kernel image is invalid" errors
# Example: set HSA_OVERRIDE_GFX_VERSION=11.0.0 in your environment before running
# torch.set_float32_matmul_precision('high')  # Disabled for ROCm compatibility


class CFG:
    SEED: int = 137
    set_seed(SEED)
    clear_memory()
    nvidia_smi_available = os.system("nvidia-smi") == 0 
    if nvidia_smi_available:
        ACCELERATOR: str = "cuda"
        DEVICE: torch.device = torch.device("cuda")
        print(f"Using device: {DEVICE}")
        cudnn.benchmark = True
    else:
        ACCELERATOR: str = "cpu"
        DEVICE: torch.device = torch.device("cpu")
        print(f"Using device: {DEVICE}")
        cudnn.benchmark = False

    # Dataset parameters
    # https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k
    DATASET_ID: str = "vitaliykinakh/stable-imagenet1k"
    DOWNLOAD_PATH: str = "data/raw/"
    if not os.path.exists(DOWNLOAD_PATH):
        os.makedirs(DOWNLOAD_PATH)
    IMAGE_DIR: str = os.path.join(DOWNLOAD_PATH, 'imagenet1k')
    
    # Data split parameters
    TEST_SIZE: float = 0.6
    VAL_SIZE: float = 0.2
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
            MODEL = torch.compile(MODEL,
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
    BATCH_SIZE: int = 128
    EPOCHS: int = 20
    LR: float = 0.001
    EPS: float = 1e-10
    WEIGHT_DECAY: float = 0.01
    MIN_DELTA: float = 0.001
    PATIENCE: int = 5
    NUM_WORKERS: int = os.cpu_count() // 2 if nvidia_smi_available else 0
    print(f"Using {NUM_WORKERS} DataLoader workers.")
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = False

    # Precision
    PRECISION: str = "32-true"
    # Loss function and optimizer
    LOSS_FN: Any = torch.nn.CrossEntropyLoss()
    OPTIMIZER: Any = torch.optim.AdamW(MODEL.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
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