# training/train.py
import torch
from configs.config import CONFIG
from training.data_loader import get_dataloaders
from training.augmentations import get_train_transforms, get_val_transforms
from training.preprocessing import preprocess_dataset
from training.model_builder import get_model_and_processor
from training.trainer_utils import compute_metrics, get_training_args, save_test_dataset
from training.mlflow_utils import start_mlflow_run, log_mlflow_artifacts, get_active_run
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from training.callbacks import MLflowLoggerCallback
from evaluation.plots import plot_loss_and_metrics

def train():
    """Función principal de entrenamiento de MobileViT Cats vs Dogs."""
    
    # Cargar dataset
    dataset = get_dataloaders(
        test_size=float(CONFIG["dataset"]["val_split"]),
        seed=int(CONFIG["dataset"]["seed"]),
        max_samples=int(CONFIG["dataset"]["max_samples"])
    )
    # Guardar partición de test para evaluación
    save_test_dataset(dataset['test'])

    # Transformaciones
    train_transforms = get_train_transforms(int(CONFIG["transforms"]["image_size"]))
    val_transforms   = get_val_transforms(int(CONFIG["transforms"]["image_size"]))

    # Construir modelo y feature extractor
    model, feature_extractor = get_model_and_processor(
        model_name=CONFIG["model"]["name"],
        num_classes=int(CONFIG["model"]["num_classes"])
    )

    # Preprocesamiento dataset
    dataset["train"] = preprocess_dataset(dataset["train"], feature_extractor, train_transforms)
    dataset["test"]  = preprocess_dataset(dataset["test"], feature_extractor, val_transforms)

    # Pasar modelo a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Configurar Trainer
    training_args = TrainingArguments(**get_training_args())

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(CONFIG["training"]["patience"])), MLflowLoggerCallback()]
    )

    # Entrenar y loguear en MLflow
    with start_mlflow_run(CONFIG["mlflow"]):
        trainer.train()
        
        run = get_active_run()  
        model_dir = f"{CONFIG['folders']['best_model']}{run.info.run_id}"
    
        # Guardar modelo y feature extractor juntos
        trainer.model.save_pretrained(model_dir)
        feature_extractor.save_pretrained(model_dir)
        log_mlflow_artifacts(model_dir, artifact_path="best_model")

        # Curvas de loss y métricas
        log_history = trainer.state.log_history
        loss_path, metrics_path = plot_loss_and_metrics(log_history, save_dir=CONFIG['folders']['root'], prefix="training_")

        # Loguear gráficos en MLflow
        log_mlflow_artifacts(loss_path)
        if metrics_path:
            log_mlflow_artifacts(metrics_path)
            
if __name__ == "__main__":
    train()

