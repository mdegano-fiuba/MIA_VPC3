import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageClassification
from configs.config import config

def get_model():
    num_classes = config['model']['num_classes']
    freeze_blocks = config['model']['frozen_blocks']
    model_name = config['model']['name']

    # Cargar configuración del modelo preentrenado
    model_config = AutoConfig.from_pretrained(model_name)

    # Crear modelo sin la capa final
    model = AutoModelForImageClassification.from_config(model_config)

    # Reemplazar la capa final
    model.classifier = nn.Linear(model_config.hidden_size, num_classes)

    # Congelar primeros bloques + embeddings
    for name, param in model.named_parameters():
        if any(name.startswith(f"vit.blocks.{i}") for i in range(freeze_blocks)):
            param.requires_grad = False
        elif name.startswith("vit.embeddings"):
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model

