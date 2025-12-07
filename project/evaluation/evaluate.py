import torch
import numpy as np
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
from evaluation.metrics import compute_all_metrics
from evaluation.plots import plot_loss_and_metrics, plot_roc, plot_confusion
from datasets import Dataset
from configs.config import CONFIG
from training.mlflow_utils import start_mlflow_run, log_mlflow_artifacts, get_active_run

def evaluate(model_path, dataset):
    model = MobileViTForImageClassification.from_pretrained(model_path)
    processor = MobileViTImageProcessor.from_pretrained(model_path)

    logits = []
    labels = []
    probs = []

    for sample in dataset:
        inputs = processor(images=sample["image"], return_tensors="pt")
        with torch.no_grad():
            out = model(inputs.pixel_values)
            
        logits.append(out.logits)
        labels.append(sample["labels"])
        
        prob = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        probs.append(prob)

    # Convertir a tensores
    logits = torch.cat(logits)
    labels = torch.tensor(labels).numpy()
    preds = logits.argmax(dim=1).numpy()
    probs = np.vstack(probs)  # (N,2)

    return compute_all_metrics(labels, preds), labels, preds, probs


if __name__ == "__main__":

    model_path = CONFIG["folders"]["model"] # Modelo entrenado
    dataset_path = CONFIG["folders"]["test_dataset"] # Dataset de test
    
    print("\nDataset loading...\n", flush=True)
    dataset = Dataset.load_from_disk(dataset_path)

    # Llamar a la función evaluate y obtener las métricas
    print("\nMetrics evaluation...\n", flush=True)
    metrics, labels, preds, probs = evaluate(model_path, dataset)

    # Imprimir las métricas obtenidas
    print("Eval Dataset Metrics:")
    for key, val in metrics.items():
        print(f"{key}: {val}")
     
    # Loguear métricas en MLflow
    with start_mlflow_run(CONFIG["mlflow"]):

        # Graficar y guardar la matriz de confusión y ROC
        img_path = CONFIG["folders"]["metrics"]
        
        img_path_cm = img_path + "/confusion_matrix.png"
        plot_confusion(labels, preds, path=img_path_cm)
        log_mlflow_artifacts(img_path_cm)
     
        img_path_rc = img_path + "/roc_curve.png"
        plot_roc(labels, probs, path=img_path_rc)
        log_mlflow_artifacts(img_path_rc)


