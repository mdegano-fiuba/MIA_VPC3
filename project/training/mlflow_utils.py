import mlflow
import mlflow.pytorch
from configs.config import config

def start_mlflow_run():
    """
    Inicia un run de MLflow.
    Se asegura de que los experimentos se registren en la URI configurada.
    """
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    return mlflow.start_run()

