import torch
from transformers import AutoModelForImageClassification
from configs.config import config

def get_model():
    """
    Carga MobileViT desde Hugging Face, reemplaza la capa final para 'num_classes',
    y congela los primeros 'frozen_blocks' bloques según el config.yaml.
    """
    num_classes = config['model']['num_classes']
    freeze_blocks = config['model']['frozen_blocks']
    model_name = config['model']['name']

    # Cargar modelo preentrenado
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )

    # Congelar los primeros bloques
    # En MobileViT, los bloques están en model.model.layers
    for i, layer in enumerate(model.model.layers):
        if i < freeze_blocks:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    # Congelar embeddings iniciales
    for param in model.model.conv_stem.parameters():
        param.requires_grad = False

    # La capa final 'classifier' queda entrenable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

