#!/bin/bash

# -------------------------------
# Script pro para levantar MLflow
# -------------------------------

# Cargar variables del .env si existe
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Valores por defecto
MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLRUNS_DIR=${MLRUNS_DIR:-./mlruns}
CONTAINER_NAME="mlflow_ui"
LOG_FILE="mlflow_run.log"

echo "✅ MLflow puerto: $MLFLOW_PORT, mlruns: $MLRUNS_DIR"
echo "📄 Logs: $LOG_FILE"

# Limpiar contenedores viejos
if [ $(docker ps -a -q -f name=$CONTAINER_NAME) ]; then
    echo "🧹 Deteniendo contenedor viejo..."
    docker stop $CONTAINER_NAME >> $LOG_FILE 2>&1
    docker rm $CONTAINER_NAME >> $LOG_FILE 2>&1
fi

# Levantar MLflow con docker-compose
echo "🔨 Construyendo imagen y levantando contenedor..."
docker-compose up --build -d >> $LOG_FILE 2>&1

# Esperar a que la UI esté lista
echo "⏳ Esperando 5 segundos a que MLflow UI se inicie..."
sleep 5

# Abrir navegador según sistema operativo
URL="http://localhost:$MLFLOW_PORT"
echo "🌐 MLflow UI disponible en $URL"

OS=$(uname)
if [[ "$OS" == "Linux" ]]; then
    if command -v xdg-open > /dev/null; then
        xdg-open $URL
    fi
elif [[ "$OS" == "Darwin" ]]; then
    if command -v open > /dev/null; then
        open $URL
    fi
elif [[ "$OS" == "MINGW"* || "$OS" == "CYGWIN"* ]]; then
    start $URL
fi

echo "✅ MLflow UI levantado correctamente. Revisa el log en $LOG_FILE"

