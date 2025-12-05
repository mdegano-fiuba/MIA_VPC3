import torch
from training.model_builder import get_model
from training.data_loader import get_dataloaders
from evaluation.metrics import compute_accuracy, compute_f1, compute_confusion_matrix
from evaluation.plots import plot_confusion_matrix
from training.mlflow_utils import start_mlflow_run
from configs.config import config
import mlflow
import os

def evaluate():
    # Carga DataLoaders (usa validación)
    _, val_loader = get_dataloaders()

    # Carga modelo
    model = get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with start_mlflow_run() as run:
        # Loguear que es evaluación
        mlflow.log_param("phase", "evaluation")

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs).logits
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds)
                all_labels.append(labels)

        # Concatenar todos los batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Calcular métricas
        acc = compute_accuracy(all_preds, all_labels)
        f1 = compute_f1(all_preds, all_labels)
        cm = compute_confusion_matrix(all_preds, all_labels)

        # Loguear métricas en MLflow
        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1", f1)

        # Guardar matriz de confusión como artefacto
        os.makedirs("evaluation_artifacts", exist_ok=True)
        cm_path = "evaluation_artifacts/confusion_matrix.png"
        plot_confusion_matrix(cm, class_names=["Cat","Dog"], save_path=cm_path)
        mlflow.log_artifact(cm_path)

        print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")
        print("Matriz de confusión guardada en MLflow.")

if __name__ == "__main__":
    evaluate()

