import mlflow
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from training.data_loader import load_cats_dogs_dataset
from training.augmentations import get_augmentations
from training.model_builder import build_model
from training.trainer_utils import compute_metrics
from training.mlflow_utils import (
    init_mlflow, log_confusion_matrix, log_roc_curve,
    log_probability_histogram
)

from configs.config import CONFIG


def preprocess_fn(feature_extractor, augmentations):
    def preprocess(examples):
        images = [augmentations(img.convert("RGB")) for img in examples["image"]]
        pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
        return {"pixel_values": pixel_values, "labels": examples["labels"]}
    return preprocess


def main():

    init_mlflow()

    dataset = load_cats_dogs_dataset()

    model, feature_extractor = build_model(CONFIG["model_name"])

    augmentations = get_augmentations()

    preprocess = preprocess_fn(feature_extractor, augmentations)

    dataset = dataset.map(preprocess, batched=True)
    dataset = dataset.remove_columns(["image"])
    dataset.set_format(type="torch")

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        load_best_model_at_end=True,
        report_to=["mlflow"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    with mlflow.start_run():

        trainer.train()

        trainer.save_model("./model/mobilevit_cats_dogs.pt")
        feature_extractor.save_pretrained("./model/")

        preds_output = trainer.predict(dataset["test"])
        logits = preds_output.predictions
        labels = preds_output.label_ids
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        preds = probs.argmax(axis=1)

        log_confusion_matrix(labels, preds)
        log_roc_curve(labels, probs)
        log_probability_histogram(probs)


if __name__ == "__main__":
    main()

