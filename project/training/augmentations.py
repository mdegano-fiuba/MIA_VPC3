from torchvision import transforms
from configs.config import CONFIG

def get_train_transforms(image_size):
    """
    Transformaciones para el entrenamiento.
    Incluye augmentations ligeras (geométricas) para evitar overfitting.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(int(CONFIG['transforms']['rotation'])),
    ])

def get_val_transforms(image_size):
    """
    Transformaciones para validación/test.
    Solo resize y crop central para consistencia.
    """
    from torchvision.transforms import Resize, CenterCrop
    return transforms.Compose([
        Resize(int(image_size * float(CONFIG['transforms']['escalation']))),
        CenterCrop(image_size)
    ])

