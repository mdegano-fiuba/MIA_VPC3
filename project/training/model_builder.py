import torch
from transformers import AutoImageProcessor, MobileViTForImageClassification
from configs.config import config

def get_model():
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    freeze_blocks = config['model']['frozen_blocks']

    # Cargar modelo preentrenado
    model = MobileViTForImageClassification.from_pretrained(model_name)

    # Reemplazar la cabeza final (classifier) por el número de clases de Cats vs Dogs
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, num_classes)

    # Congelar bloques iniciales y embeddings
    for name, param in model.named_parameters():
        if any(name.startswith(f"mobilevit.encoder.{i}") for i in range(freeze_blocks)):
            param.requires_grad = False
        elif name.startswith("embeddings"):
            param.requires_grad = False

    return model, AutoImageProcessor.from_pretrained(model_name)

