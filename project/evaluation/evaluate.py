import torch
from transformers import MobileViTForImageClassification, MobileViTImageProcessor

from evaluation.metrics import compute_all_metrics

def evaluate(model_path, dataset):
    model = MobileViTForImageClassification.from_pretrained(model_path)
    processor = MobileViTImageProcessor.from_pretrained(model_path)

    logits = []
    labels = []

    for sample in dataset:
        inputs = processor(images=sample["image"], return_tensors="pt")
        with torch.no_grad():
            out = model(inputs.pixel_values)
        logits.append(out.logits)
        labels.append(sample["labels"])

    preds = torch.softmax(torch.cat(logits), dim=1).argmax(axis=1)

    return compute_all_metrics(labels, preds)

