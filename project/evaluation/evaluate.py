import torch
from transformers import MobileViTForImageClassification, MobileViTImageProcessor
from evaluation.metrics import compute_all_metrics
from datasets import DatasetDict
from configs.config import CONFIG

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


if __name__ == "__main__":

    model_path = CONFIG["folders"]["model"] # Modelo entrenado
    dataset_path = CONFIG["folders"]["test_dataset"] # Dataset de test
    
    dataset = DatasetDict.load_from_disk(dataset_path)

    # Llamar a la función evaluate y obtener las métricas
    metrics = evaluate(model_path, dataset)

    # Loguear métricas en MLflow
    with start_mlflow_run():
    
        log_metrics(metrics)

        # Generar y guardar los gráficos
        loss_path, metrics_path = plot_loss_and_metrics(dataset, save_dir=CONFIG["folders"]["aux"], prefix="eval_")
        log_mlflow_artifacts([loss_path, metrics_path])

        # Graficar y guardar la matriz de confusión y ROC
        plot_confusion(labels, preds, path="confusion_matrix.png")
        plot_roc(labels, torch.softmax(torch.cat(logits), dim=1).cpu().numpy(), path="roc_curve.png")
        # Registra los gráficos adicionales
        log_mlflow_artifacts(["confusion_matrix.png", "roc_curve.png"])

    # Imprimir las métricas obtenidas
    print("Metrics:", metrics)
