import torch
import torch.nn as nn
import torch.optim as optim

def get_optimizer(model, lr):
    return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

def get_criterion():
    return nn.CrossEntropyLoss()

