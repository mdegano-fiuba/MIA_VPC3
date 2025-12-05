from torchvision import transforms

def get_train_transforms(image_size):
    """
    Devuelve las transformaciones de data augmentation para entrenamiento.
    Se aplican solo durante entrenamiento para aumentar la variabilidad de los datos.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Redimensiona a tamaño fijo
        transforms.RandomHorizontalFlip(),            # Voltea horizontal aleatoriamente
        transforms.RandomRotation(15),               # Rota aleatoriamente hasta 15 grados
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Ajustes de color
        transforms.ToTensor()                         # Convierte a tensor
    ])

def get_val_transforms(image_size):
    """
    Transformaciones de validación: solo redimensionamiento y conversión a tensor.
    No se aplican augmentations para evaluar correctamente el desempeño.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

