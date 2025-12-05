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

        mlflow.pytorch.log_model(model, "model")
        print("Modelo guardado en MLFlow.")

if __name__ == "__main__":
    train()


