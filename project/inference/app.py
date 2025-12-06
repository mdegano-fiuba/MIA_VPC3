import torch
import gradio as gr
from PIL import Image
from transformers import MobileViTForImageClassification, MobileViTImageProcessor

model_path = "./model/trained"
model = MobileViTForImageClassification.from_pretrained(model_path)
processor = MobileViTImageProcessor.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

CLASS_NAMES = ["Cat", "Dog"]

def predict(img: Image.Image):
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    with torch.no_grad():
        logits = model(pixel_values).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(2)}

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Cats vs Dogs Classifier",
)

if __name__ == "__main__":
    iface.launch(share=True)

