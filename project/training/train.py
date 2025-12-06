import torch
from training.data_loader import get_dataloaders
from training.model_builder import get_model
from configs.config import config
import mlflow

def train():
    train_loader, val_loader = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo
    model = get_model()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    with mlflow.start_run():
        mlflow.log_params(config['training'])
        print("Logging training parameters:", config['training'])

        for epoch in range(config['training']['epochs']):
            model.train()
            total_loss = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(pixel_values=imgs).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}/{config['training']['epochs']} - Train loss: {avg_loss:.4f}")

            # Validación
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(pixel_values=imgs).logits
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            print(f"Epoch {epoch+1}/{config['training']['epochs']} - Val loss: {avg_val_loss:.4f} - Val acc: {val_acc:.4f}")

        # Guardar modelo
        model_save_path = config['training']['output_dir']
        model.save_pretrained(model_save_path)
        print(f"Modelo guardado en: {model_save_path}")
        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()

