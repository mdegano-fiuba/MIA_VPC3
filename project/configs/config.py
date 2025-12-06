# configs/config.py

import yaml
import os

# Obtener la ruta absoluta del archivo YAML relativo a este script
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Verificar que el archivo exista
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Falta el archivo de configuración: {CONFIG_PATH}")

# Cargar el YAML
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)


