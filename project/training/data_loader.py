import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
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
        img = self.ds[idx]['image']  # PIL Image
        label = self.ds[idx]['labels']  
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloaders():
    """
    Carga el dataset Cats vs Dogs y devuelve DataLoaders de entrenamiento y validación.
    Split del dataset manteniendo balance de clases
    """
    # Cargar dataset completo
    dataset = load_dataset(config['dataset']['name'])

    # Split balanceado
    split_dataset = dataset['train'].train_test_split(
        test_size=config['dataset']['val_split'], 
        stratify_by_column='labels', 
        seed=config['dataset']['seed']
    )

    train_ds = CatsDogsDataset(split_dataset['train'], transform=get_train_transforms(config['dataset']['image_size']))
    val_ds = CatsDogsDataset(split_dataset['test'], transform=get_val_transforms(config['dataset']['image_size']))

    train_loader = DataLoader(train_ds, batch_size=config['dataset']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['dataset']['batch_size'], shuffle=False)

    return train_loader, val_loader

