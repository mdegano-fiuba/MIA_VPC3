from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

def load_cats_dogs_dataset(test_size=0.2, seed=42):
    dataset = load_dataset("cats_vs_dogs")["train"]
    labels = dataset["labels"]

    train_idx, test_idx = train_test_split(
        range(len(labels)),
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )

    train_ds = dataset.select(train_idx)
    test_ds = dataset.select(test_idx)

    return DatasetDict({"train": train_ds, "test": test_ds})

