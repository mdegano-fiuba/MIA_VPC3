import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from configs.config import CONFIG

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
    }


def get_training_args():
    training_args_kwargs = {
        "output_dir": CONFIG["folders"]["output"],
        "per_device_train_batch_size": int(CONFIG["dataset"]["batch_size"]),
        "per_device_eval_batch_size": int(CONFIG["dataset"]["batch_size"]),
        "num_train_epochs": int(CONFIG["training"]["epochs"]),
        "learning_rate": float(CONFIG["training"]["learning_rate"]),
        "weight_decay": float(CONFIG["training"]["weight_decay"]),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "report_to": ["mlflow"],
        "logging_steps": int(CONFIG["training"]["logging_steps"]),
        "push_to_hub": False,
    }
    return training_args_kwargs


def save_test_dataset(dataset, save_dir=CONFIG["folders"]["test_dataset"]):
    try:
        # Intenta guardar el dataset en el directorio proporcionado
        dataset.save_to_disk(save_dir)
    except Exception as e:
        raise RuntimeError(f"Error saving the test dataset to {save_dir}: {str(e)}") from e

