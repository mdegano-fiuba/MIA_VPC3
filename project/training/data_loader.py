import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from training.augmentations import get_train_transforms, get_val_transforms
from configs.config import config

class CatsDogsDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloaders():
    # Cargar dataset original
    dataset = load_dataset(config['dataset']['name'])["train"]  # Hugging Face solo tiene train
    images = dataset["image"]
    labels = dataset["labels"]

    # Split estratificado
    train_imgs, val_imgs, train_labels, val_labels = train_test_split(
        images,
        labels,
        test_size=config['dataset']['val_split'],
        stratify=labels,
        random_state=config['dataset']['seed']
    )

    # Crear datasets PyTorch con augmentations
    train_ds = CatsDogsDataset(train_imgs, train_labels, transform=get_train_transforms(config['dataset']['image_size']))
    val_ds = CatsDogsDataset(val_imgs, val_labels, transform=get_val_transforms(config['dataset']['image_size']))

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config['dataset']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['dataset']['batch_size'], shuffle=False)

    return train_loader, val_loader

