# training/data_loader.py
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter

def load_cats_dogs_dataset(test_size, seed, max_samples=None):
    """
    Carga el dataset Cats vs Dogs y opcionalmente reduce a un número máximo de imágenes.
    Mantiene el balance de clases.
    """
    dataset = load_dataset("cats_vs_dogs")["train"]
    labels = np.array(dataset["labels"])
    n_classes = len(set(labels))

    if max_samples is not None and max_samples < len(dataset):
        # Calcular cuántas muestras por clase (floor division)
        samples_per_class = max_samples // n_classes

        selected_idx = []
        for cls in range(n_classes):
            cls_idx = np.where(labels == cls)[0]
            if len(cls_idx) < samples_per_class:
                raise ValueError(f"No hay suficientes muestras de la clase {cls} para reducir a {samples_per_class}")
            cls_selected = np.random.choice(cls_idx, size=samples_per_class, replace=False)
            selected_idx.extend(cls_selected)

        # Si hay un residuo, tomar aleatoriamente del resto
        remainder = max_samples - len(selected_idx)
        if remainder > 0:
            remaining_idx = list(set(range(len(dataset))) - set(selected_idx))
            extra_idx = np.random.choice(remaining_idx, size=remainder, replace=False)
            selected_idx.extend(extra_idx)

        dataset = dataset.select(selected_idx)
        labels = np.array([dataset[i]["labels"] for i in range(len(dataset))])

    # Dividir en train/test manteniendo balance
    train_idx, test_idx = train_test_split(
        range(len(labels)),
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )
    
    return dataset.select(train_idx), dataset.select(test_idx)


def get_dataloaders(test_size, seed, max_samples=None):
    train_ds, test_ds = load_cats_dogs_dataset(test_size, seed, max_samples)
    return DatasetDict({"train": train_ds, "test": test_ds})



