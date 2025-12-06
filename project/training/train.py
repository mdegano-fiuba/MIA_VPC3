# training/train.py

from configs.config import CONFIG
from training.data_loader import get_dataloaders
from training.augmentations import get_train_transforms, get_val_transforms
from training.preprocessing import preprocess_dataset
from training.model_builder import get_model_and_processor
from training.trainer_utils import compute_metrics
from training.mlflow_utils import start_mlflow_run
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch
from training.callbacks import MLflowLoggerCallback

def train():
    """Función principal de entrenamiento de MobileViT Cats vs Dogs."""
    
    # Cargar dataset
    dataset = get_dataloaders(
        test_size=CONFIG["dataset"]["val_split"],
        seed=CONFIG["dataset"]["seed"]
    )

    # Transformaciones
    train_transforms = get_train_transforms(CONFIG["dataset"]["image_size"])
    val_transforms   = get_val_transforms(CONFIG["dataset"]["image_size"])

    # Construir modelo y feature extractor
    model, feature_extractor = get_model_and_processor(
        model_name=CONFIG["model"]["name"],
        num_classes=CONFIG["model"]["num_classes"]
    )

    # Preprocesamiento dataset
    dataset["train"] = preprocess_dataset(dataset["train"], feature_extractor, train_transforms)
    dataset["test"]  = preprocess_dataset(dataset["test"], feature_extractor, val_transforms)

    # Pasar modelo a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Configurar Trainer
    training_args = TrainingArguments(
        output_dir="./mobilevit_cats_vs_dogs",
        per_device_train_batch_size=CONFIG["dataset"]["batch_size"],
        per_device_eval_batch_size=CONFIG["dataset"]["batch_size"],
        num_train_epochs=CONFIG["training"]["epochs"],
        learning_rate=CONFIG["training"]["learning_rate"],
        weight_decay=CONFIG["training"]["weight_decay"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=["mlflow"],
        logging_steps=CONFIG["training"]["logging_steps"],
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG["training"]["patience"]), MLflowLoggerCallback()]
    )

    # Entrenar y loguear en MLflow
    with start_mlflow_run(CONFIG["mlflow"]):
        trainer.train()
        # Guardar modelo y feature extractor juntos
        trainer.model.save_pretrained("./best_model")
        feature_extractor.save_pretrained("./best_model")
        mlflow.log_artifacts("./best_model", artifact_path="best_model")

if __name__ == "__main__":
    train()

