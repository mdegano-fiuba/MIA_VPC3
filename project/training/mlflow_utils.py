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

def log_mlflow_artifacts(path, artifact_path="artifacts"):
    mlflow.log_artifacts(path, artifact_path=artifact_path)
    
def get_active_run():
    """Devuelve el run activo de MLflow. Lanza error si no hay ninguno."""
    run = mlflow.active_run()
    if not run:
        raise RuntimeError("No hay ningún MLflow run activo. Usar start_mlflow_run primero.")
    return run

def log_metrics(metrics):
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)

