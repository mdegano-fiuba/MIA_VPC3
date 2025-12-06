from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

# DATASET REDUCIDO PARA PRUEBAS
from datasets import load_from_disk
#

def load_cats_dogs_dataset(test_size, seed):

    # DATASET REDUCIDO PARA PRUEBAS
    # dataset = load_dataset("cats_vs_dogs")["train"] 
    dataset = load_from_disk("./data/tinyDS/")
    ####

    labels = dataset["labels"]
    train_idx, test_idx = train_test_split(
        range(len(labels)),
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )
    return dataset.select(train_idx), dataset.select(test_idx)

def get_dataloaders(test_size, seed):
    train_ds, test_ds = load_cats_dogs_dataset(test_size, seed)
    return DatasetDict({"train": train_ds, "test": test_ds})

