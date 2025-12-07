#!/bin/bash

echo "  MLflow Environment"

# Comprobar archivo de entorno
if [ ! -f ".env" ]; then
    echo "[ERROR] No existe archivo .env"
    exit 1
fi

# Cargar variables
source .env

echo ">> MLflow HOST: $MLFLOW_HOST"
echo ">> MLflow PORT: $MLFLOW_PORT"
echo ">> Artefactos internos en: $MLFLOW_ARTIFACTS"
echo ">> Artefactos locales en: $MLRUNS_DIR"

# Crear carpeta local si no existe
if [ ! -d "$MLRUNS_DIR" ]; then
    echo ">> Creando carpeta local de artefactos $MLRUNS_DIR ..."
    mkdir -p "$MLRUNS_DIR"
fi

# Ajustar permisos para que el contenedor pueda acceder
sudo chown -R 1000:1000 "$MLRUNS_DIR"
chmod -R 775 "$MLRUNS_DIR"

echo ">> Construyendo imagen..."
docker-compose build --no-cache

echo ">> Levantando contenedor MLflow..."
docker-compose up -d

echo ">> Esperando al healthcheck..."
for i in {1..20}; do
    state=$(docker inspect --format='{{json .State.Health.Status}}' mlflow_server 2>/dev/null)
    if [[ "$state" == "\"healthy\"" ]]; then
        echo ">> MLflow está healthy ✔"
        break
    fi
    echo "   Esperando... ($i/20)"
    sleep 2
done

echo "==============================================="
echo " MLflow UI disponible en:"
echo "  http://localhost:${MLFLOW_PORT}"
echo ""
echo " Carpeta local de artefactos:"
echo "  $(pwd)/${MLRUNS_DIR}"
echo "==============================================="

