FROM python:3.10-slim

# Evitar caches de Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Instalar MLflow globalmente
RUN pip install --no-cache-dir mlflow

# Crear usuario no-root
RUN useradd -ms /bin/bash mlflowuser

# Crear carpeta de artefactos y dar permisos
RUN mkdir -p /app/mlruns && chown -R mlflowuser:mlflowuser /app

# Cambiar a usuario no-root
USER mlflowuser

EXPOSE 5000

# Healthcheck
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:5000 || exit 1

# Comandos
CMD ["mlflow", "server", "--backend-store-uri", "/app/mlruns", "--default-artifact-root", "/app/mlruns", "--host", "0.0.0.0", "--port", "5000", "--workers", "4", "--serve-artifacts"]
