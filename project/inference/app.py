import torch
from PIL import Image
from transformers import AutoFeatureExtractor
from training.model_builder import get_model
from configs.config import config
import gradio as gr

# ------------------------------
# Cargar modelo y extractor
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo entrenado (MobileViT)
model = get_model()
# Si guardaste el modelo en archivo .pt, puedes usar:
# model.load_state_dict(torch.load("model/mobilevit_cats_dogs.pt", map_location=device))
model.to(device)
model.eval()

# AutoFeatureExtractor para preprocesar imágenes al tamaño correcto
feature_extractor = AutoFeatureExtractor.from_pretrained("apple/mobilevit-small")

# Etiquetas de clases
class_names = ["Cat", "Dog"]

# ------------------------------
# Función de predicción
# ------------------------------
def predict(image: Image.Image):
    """
    Recibe una imagen PIL, la transforma y devuelve la predicción de la clase.
    """
    # Preprocesamiento: tamaño, normalización
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)

    with torch.no_grad():
        outputs = model(pixel_values).logits
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item()

    return {class_names[i]: float(probs[0, i]) for i in range(len(class_names))}, f"{class_names[pred_idx]} ({confidence:.2f})"

# ------------------------------
# Interfaz Gradio
# ------------------------------
title = "Clasificación Cats vs Dogs"
description = "Sube una imagen de un gato o perro y el modelo te dirá la clase."

interface = gr.Interface(
    fn=predict,                     # Función que hace la predicción
    inputs=gr.Image(type="pil"),    # Input: imagen PIL
    outputs=[gr.Label(num_top_classes=2), gr.Textbox()], # Salida: probabilidades + predicción
    title=title,
    description=description
)

# Ejecutar la app
if __name__ == "__main__":
    # Para Hugging Face Spaces, la app se ejecutará automáticamente
    interface.launch()

