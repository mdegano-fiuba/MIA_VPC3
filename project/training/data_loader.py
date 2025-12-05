from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch

from training.augmentations import get_transforms
from configs.config import config

class CatsDogsDataset(torch.utils.data.Dataset):
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]['image']
        label = self.ds[idx]['label']
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloaders():
    dataset = load_dataset(config['dataset']['name'])

    transform = get_transforms(config['dataset']['image_size'])
    
    train_loader = DataLoader(CatsDogsDataset(dataset['train'], transform), 
                              batch_size=config['dataset']['batch_size'], shuffle=True)
    val_loader = DataLoader(CatsDogsDataset(dataset['test'], transform), 
                            batch_size=config['dataset']['batch_size'])
    return train_loader, val_loader

