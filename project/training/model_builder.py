# training/model_builder.py
import torch
from transformers import MobileViTForImageClassification
from configs.config import config

def get_model():
    """
    Carga MobileViT preentrenado desde Hugging Face,
    adapta la cabeza a num_classes y deja solo la cabeza entrenable.
    """
    num_classes = config['model']['num_classes']
    model_name = config['model']['name']  

    # Cargar MobileViT con la cabeza adaptada a num_classes
    model = MobileViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )

    # Congelar todo el backbone
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True  # cabeza entrenable
        else:
            param.requires_grad = False  # backbone congelado

    return model

