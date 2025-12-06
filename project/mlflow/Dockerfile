# Dockerfile
FROM python:3.12-slim

# Evitar prompts de instalación
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instalar MLflow con extras (para logging y UI)
RUN pip install --no-cache-dir mlflow[extras]

# Directorio de trabajo
WORKDIR /mlflow

# Puerto de la UI
EXPOSE 5000

# Comando por defecto
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/mlflow/mlruns"]

