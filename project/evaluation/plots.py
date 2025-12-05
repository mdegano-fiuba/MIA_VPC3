import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Dibuja la matriz de confusión usando Seaborn.
    """
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.close()

