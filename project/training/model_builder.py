from transformers import AutoModelForImageClassification

from configs.config import config

def get_model(num_labels=2):
    model = AutoModelForImageClassification.from_pretrained(
        "apple/mobilevit-small",
        num_labels=num_labels
    )

    # Fine-tuning progresivo: solo capas del classifier
    for name, param in model.named_parameters():
        if config['training']['fine_tune_layers'] in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model

