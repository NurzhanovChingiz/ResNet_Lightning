from utils.clear_gpu import clear_memory
from utils.set_seed import set_seed
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
import os
import torch
from torch.backends import cudnn
from model import ResNet18
from typing import Any
torch.set_float32_matmul_precision('high')


class CFG:
    SEED: int = 137
    set_seed(SEED)
    clear_memory()

    cudnn.benchmark = True

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
    
    # Device
    ACCELERATOR: str = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    # Directories
    MAIN_DIR: str = os.getcwd()
    BASE_DIR: str = f"{MAIN_DIR}/data" # Must be pre-downloaded
    
    # CKPT_PATH: str = "lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
    
    MODEL_NAME: str = "ResNet18"
    PRETRAIN: bool = False
    # Model
    MODEL: Any = ResNet18(num_classes=1000, pretrained=PRETRAIN)
    
    # Compile
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
    
    NUM_WORKERS: int = 32
    print(f"Using {NUM_WORKERS} DataLoader workers.")
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    PRECISION: str = "bf16-mixed" if torch.cuda.is_available() else "32-true"
    # Automatic Mixed Precision (AMP)
    # USE_AMP: bool = True
    # AMP_DTYPE: torch.dtype | None = torch.float16 if USE_AMP else None
    # SCALER = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP) if USE_AMP else None
    
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