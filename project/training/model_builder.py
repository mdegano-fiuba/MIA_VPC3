from transformers import MobileViTForImageClassification, MobileViTImageProcessor

def get_model_and_processor(model_name, num_classes):
    feature_extractor = MobileViTImageProcessor.from_pretrained(model_name)
    model = MobileViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model, feature_extractor

