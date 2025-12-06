from transformers import MobileViTForImageClassification, MobileViTFeatureExtractor
from configs.config import config

def get_model():
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    freeze_blocks = config['model'].get('frozen_blocks', 0)  # opcional, default=0

    # Cargar modelo preentrenado
    model = MobileViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )

    # Congelar primeros bloques si se indicó
    if freeze_blocks > 0:
        for name, param in model.named_parameters():
            if any(f"mobilevit.blocks.{i}" in name for i in range(freeze_blocks)):
                param.requires_grad = False
            elif "embeddings" in name:
                param.requires_grad = False

    return model

def get_feature_extractor():
    model_name = config['model']['name']
    return MobileViTFeatureExtractor.from_pretrained(model_name)

