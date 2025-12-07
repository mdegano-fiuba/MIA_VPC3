# evaluation/plots.py
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_confusion(labels, preds, path="confusion.png"):
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.savefig(path)

def plot_roc(labels, probs, path="roc.png"):
    fpr, tpr, _ = roc_curve(labels, probs[:,1])
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC {auc_val:.2f}")
    plt.legend()
    plt.savefig(path)


def plot_loss_and_metrics(log_history , save_dir=".", prefix=""):
    """
    Grafica la loss de entrenamiento y validación y métricas por época.
    Guarda los gráficos como PNG en save_dir con prefijo opcional.
    """
    loss_path, metrics_path = None, None
        
    # Loss sólo en train
    if (prefix=="train_"):
        train_loss = [x['loss'] for x in log_history if 'loss' in x]
        eval_loss  = [x['eval_loss'] for x in log_history if 'eval_loss' in x]
        epochs = range(1, len(eval_loss)+1)

        plt.figure()
        plt.plot(epochs, train_loss[:len(epochs)], label="Train Loss")
        plt.plot(epochs, eval_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        loss_path = os.path.join(save_dir, f"{prefix}loss_plot.png")
        plt.savefig(loss_path)
        plt.close()

    # Métricas: Accuracy, F1, Precision, Recall (si están en log_history)
    accuracy   = [x['eval_accuracy'] for x in log_history if 'eval_accuracy' in x]
    f1_score_m = [x['eval_f1'] for x in log_history if 'eval_f1' in x]
    precision  = [x['eval_precision'] for x in log_history if 'eval_precision' in x]
    recall     = [x['eval_recall'] for x in log_history if 'eval_recall' in x]

    if accuracy:
        plt.figure()
        plt.plot(epochs, accuracy, label="Accuracy")
        if f1_score_m: plt.plot(epochs, f1_score_m, label="F1 Score")
        if precision: plt.plot(epochs, precision, label="Precision")
        if recall: plt.plot(epochs, recall, label="Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("Evaluation Metrics per Epoch")
        plt.legend()
        metrics_path = os.path.join(save_dir, f"{prefix}metrics_plot.png")
        plt.savefig(metrics_path)
        plt.close()

    return (loss_path, metrics_path)

