import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

def init_mlflow(experiment_name="mobilevit_cats_vs_dogs", tracking_uri="./mlruns"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")


def log_roc_curve(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")


def log_probability_histogram(probs):
    plt.figure()
    sns.histplot(probs[:,1], kde=True)
    plt.title("Probability Distribution (Dog)")
    plt.savefig("prob_hist.png")
    mlflow.log_artifact("prob_hist.png")

