import torch

def preprocess_dataset(dataset, feature_extractor, transforms):
    """
    Aplica augmentations y convierte imágenes a tensores.
    Devuelve dataset listo para PyTorch.
    """
    def _preprocess(examples):
        images = [transforms(img.convert("RGB")) for img in examples["image"]]
        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        examples["pixel_values"] = pixel_values
        examples["labels"] = examples["labels"]
        return examples

    dataset = dataset.map(lambda x: _preprocess(x), batched=True)
    dataset = dataset.remove_columns(["image"])
    dataset.set_format(type="torch", columns=["pixel_values", "labels"])
    return dataset

