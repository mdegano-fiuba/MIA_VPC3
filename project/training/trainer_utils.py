import torch.nn as nn
import torch.optim as optim

def get_optimizer(model, lr):
    """
    Devuelve el optimizador Adam con los parámetros que requieren gradiente.
    """
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

def get_criterion():
    """
    Función de pérdida: CrossEntropyLoss.
    """
    return nn.CrossEntropyLoss()

