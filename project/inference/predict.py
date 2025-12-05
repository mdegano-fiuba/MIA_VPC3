import torch
from PIL import Image
from training.model_builder import get_model
from configs.config import config
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model()
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((config['dataset']['image_size'], config['dataset']['image_size'])),
    transforms.ToTensor()
])

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img).logits
        pred = torch.argmax(logits, dim=1).item()
    return "dog" if pred == 1 else "cat"

