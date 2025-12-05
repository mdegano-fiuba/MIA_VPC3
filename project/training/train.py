import torch
from training.data_loader import get_dataloaders
from training.model_builder import get_model
from training.trainer_utils import get_optimizer, get_criterion
from training.mlflow_utils import start_mlflow_run
from configs.config import config
import mlflow

def train():
    train_loader, val_loader = get_dataloaders()
    model = get_model()
    optimizer = get_optimizer(model, config['training']['learning_rate'])
    criterion = get_criterion()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with start_mlflow_run() as run:
        # Loguear parámetros de entrenamiento
        mlflow.log_params(config['training'])

        for epoch in range(config['training']['epochs']):
            model.train()
            total_loss = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}: Train loss={avg_loss}")

            # Validación
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs).logits
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            print(f"Epoch {epoch+1}: Val loss={avg_val_loss}, Val acc={val_acc}")

        # Guardar modelo en MLflow
        mlflow.pytorch.log_model(model, "model")
        print("Modelo guardado en MLflow.")

if __name__ == "__main__":
    train()

