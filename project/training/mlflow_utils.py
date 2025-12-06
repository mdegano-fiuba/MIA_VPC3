import mlflow
from contextlib import contextmanager

@contextmanager
def start_mlflow_run(mlflow_config):
    """
    Context manager para iniciar y cerrar un run de MLflow.
    """
    mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])
    with mlflow.start_run():
        yield

