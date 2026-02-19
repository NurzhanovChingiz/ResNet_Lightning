from torch.utils.data import Dataset
from PIL import Image
from typing import Any
import torch


class ImageDataset(Dataset):
    def __init__(self, data: list[str], labels: list[int], transform: Any = None) -> None:
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | Image.Image, int]:
        sample: Image.Image | torch.Tensor = Image.open(self.data[idx]).convert("RGB")
        
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
    
