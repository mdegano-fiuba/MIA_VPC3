import mlflow
import mlflow.pytorch
from configs.config import config

def start_mlflow_run():
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    return mlflow.start_run()

