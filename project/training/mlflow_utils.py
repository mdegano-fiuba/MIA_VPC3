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
    try:
        mlflow.log_artifacts(path, artifact_path=artifact_path)
    except Exception as e:
        # Si ocurre algún error, se captura y muestra un mensaje de error
        print(f"\nError while logging artifacts {path}: {e}\n")
        raise  # Volver a lanzar la excepción para que no se continúe sin el log adecuado
    
def get_active_run():
    """Devuelve el run activo de MLflow. Lanza error si no hay ninguno."""
    run = mlflow.active_run()
    if not run:
        raise RuntimeError("No hay ningún MLflow run activo. Usar start_mlflow_run primero.")
    return run


