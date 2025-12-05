from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch

def compute_accuracy(preds, labels):
    """
    Calcula la exactitud (accuracy) de predicciones vs etiquetas.
    """
    return accuracy_score(labels.cpu(), preds.cpu())

def compute_f1(preds, labels):
    """
    Calcula F1-score de predicciones vs etiquetas.
    """
    return f1_score(labels.cpu(), preds.cpu(), average='binary')

def compute_confusion_matrix(preds, labels):
    """
    Devuelve la matriz de confusión.
    """
    return confusion_matrix(labels.cpu(), preds.cpu())

