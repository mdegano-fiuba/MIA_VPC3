from transformers import MobileViTForImageClassification, MobileViTImageProcessor

def build_model(model_name="apple/mobilevit-small", num_labels=2):
    feature_extractor = MobileViTImageProcessor.from_pretrained(model_name)

    model = MobileViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )

    return model, feature_extractor

