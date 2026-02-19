import torch
import numpy as np
import random
from lightning.pytorch import seed_everything
import os

def set_seed(SEED: int = 42) -> None:
    seed_everything(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print('Random Seed : {0}'.format(SEED))

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32 # get the seed from the DataLoader's generator
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def get_generator(SEED: int = 42) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return generator