# training/callbacks.py
from transformers import TrainerCallback
import mlflow

class MLflowLoggerCallback(TrainerCallback):
    """
    Callback para imprimir métricas en consola al final de cada epoch
    y verificar que MLflow está logueando correctamente.
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        run = mlflow.active_run()
        if run:
            # Mostrar métricas parciales del epoch actual
            print(f"Epoch {state.epoch:.0f} - Metrics so far: {state.log_history[-1]}")

