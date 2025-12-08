# 📁 Estructura del proyecto  

🧩 project/
|
├─ 📂 configs/
│  ├─ 🐍 `config.py`        # Configuraciones de Python
│  └─ 📝 `config.yaml`      # Archivo YAML de parámetros
|
├─ 📂 data/
│  └─ 📂 test_dataset/      # Dataset de prueba
│     └─ ...               # Archivos de imagen / datos
|
├─ 📂 doc/
│  └─ 📂 imgs/              # Imágenes de documentación / figuras
|
├─ 📂 evaluation/
│  ├─ 🐍 `__init__.py`      # Inicializador del paquete evaluation
│  ├─ 🐍 `evaluate.py`      # Script de evaluación de modelos
│  ├─ 🐍 `metrics.py`       # Cálculo de métricas
│  └─ 🐍 `plots.py`         # Funciones para graficar resultados
|
├─ 📂 inference/
│  ├─ 🐍 `__init__.py`      # Inicializador del paquete inference
│  └─ 🐍 `app.py`           # App de inferencia
|
├─ 📂 metrics/
│  ├─ 🧮 confusion_matrix.png   # Matriz de confusión
│  ├─ 📈 roc_curve.png          # Curva ROC
│  ├─ 📉 train_loss_plot.png    # Pérdida de entrenamiento
│  └─ 📉 train_metrics_plot.png # Métricas de entrenamiento
|
├─ 📂 mlflow/
│  └─ 📂 mlruns/             # Directorio de experimentos MLflow
│     └─ ...
│  ├─ 🔑 .env               # Variables de entorno
│  ├─ 🐳 Dockerfile          # Dockerfile para MLflow
│  ├─ 🐳 docker-compose.yml  # Compose para MLflow
│  └─ ⚡ run_mlflow.sh        # Script de ejecución de MLflow
|
├─ 📂 model/
│  └─ 📂 trained/            # Modelos entrenados
│     └─ ...
|
├─ 📂 notebooks/
│  ├─ 📓 EDA.ipynb           # Notebook de análisis exploratorio
│  ├─ 📓 Eval.ipynb          # Notebook de evaluación de modelos
│  └─ 📓 Train.ipynb         # Notebook de entrenamiento
|
├─ 📂 tests/
│  └─ 📂 samples/            # Datos de prueba / samples
│     └─ ...
│  └─ 🎞️ Inference_app.png  # Captura de la app de inferencia
|
├─ 📂 training/
│  ├─ 🐍 `__init__.py`       # Inicializador del paquete training
│  ├─ 🐍 augmentations.py    # Funciones de augmentación de datos
│  ├─ 🐍 callbacks.py        # Callbacks para entrenamiento
│  ├─ 🐍 data_loader.py      # Carga y preprocesamiento de datos
│  ├─ 🐍 mlflow_utils.py     # Integración con MLflow
│  ├─ 🐍 model_builder.py    # Definición del modelo MobileViT
│  ├─ 🐍 preprocessing.py    # Funciones de preprocesamiento
│  ├─ 🐍 train.py            # Script principal de entrenamiento
│  └─ 🐍 trainer_utils.py    # Utilidades para entrenamiento
|
├─ 📦 requirements.txt       # Dependencias Python
└─ 📘 README.md              # Documentación principal del proyecto

   










