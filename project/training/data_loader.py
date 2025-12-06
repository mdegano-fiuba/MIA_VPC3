import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from training.augmentations import get_train_transforms, get_val_transforms
from configs.config import config

class CatsDogsDataset(torch.utils.data.Dataset):
    """
    Dataset personalizado para Cats vs Dogs.
    Toma la imagen y la etiqueta desde Hugging Face dataset.
    """
    def __init__(self, ds, transform=None):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        idx = int(idx)  # Convertir a int puro para PyTorch
        img = self.ds[idx]['image']  # PIL Image
        label = self.ds[idx]['labels']  
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloaders():
    """
    Carga el dataset, realiza split balanceado y devuelve DataLoaders.
    """
    # 1) Cargar dataset completo
    dataset = load_dataset(config['dataset']['name'])

    # 2) Hacer split train/val balanceado
    split = dataset['train'].train_test_split(
        test_size=config['dataset']['val_split'],
        stratify_by_column='labels',
        seed=config['dataset']['seed']
    )
    train_ds = CatsDogsDataset(split['train'], transform=get_train_transforms(config['dataset']['image_size']))
    val_ds = CatsDogsDataset(split['test'], transform=get_val_transforms(config['dataset']['image_size']))

    # 3) Crear DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config['dataset']['batch_size'],
        shuffle=True,
        num_workers=2,  
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['dataset']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader

