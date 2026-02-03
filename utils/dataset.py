from torch.utils.data import Dataset
from PIL import Image

class Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = Image.open(sample).convert("RGB")
        
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label
    
