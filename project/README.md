# 📁 Estructura del proyecto  

```plaintext
🧩 project/
|
├─ 📂 configs/                   # Archivos de configuración
│  ├─ 🐍 config.py               # Carga de parámetros
│  └─ 📝 config.yaml             # Archivo de parámetros
|
├─ 📂 data/                      # Datos  
│  └─ 📂 test_dataset/           # Dataset para validación
│     └─ ...                      
|
├─ 📂 doc/                       # Documentación  
│  └─ 📂 imgs/                   # Imágenes de documentación / figuras
|
├─ 📂 evaluation/                # Archivos para evaluación del modelo
│  ├─ 🐍 __init__.py      
│  ├─ 🐍 evaluate.py      
│  ├─ 🐍 metrics.py       
│  └─ 🐍 plots.py         
|
├─ 📂 inference/                 # App para demo de inferencia
│  ├─ 🐍 __init__.py     
│  └─ 🐍 app.py           
|
├─ 📂 metrics/                   # Gráficos generados durante el entrenamiento
│  ├─ 🧮 confusion_matrix.png   
│  ├─ 📈 roc_curve.png          
│  ├─ 📉 train_loss_plot.png    
│  └─ 📉 train_metrics_plot.png 
|
├─ 📂 mlflow/                    # Archivos para la creación de contenedor con MLFlow UI 
│  └─ 📂 mlruns/                 # Experimentos
│     └─ ...
│  ├─ 🔑 .env              
│  ├─ 🐳 Dockerfile          
│  ├─ 🐳 docker-compose.yml  
│  └─ ⚡ run_mlflow.sh       
|
├─ 📂 model/                     # Modelos
│  └─ 📂 trained/                # Modelo entrenado
│     └─ ...
|
├─ 📂 notebooks/              
│  ├─ 📓 EDA.ipynb               # Análisis exploratorio de datos
│  ├─ 📓 Eval.ipynb              
│  └─ 📓 Train.ipynb         
|
├─ 📂 tests/                     # Pruebas
│  └─ 📂 samples/                # Ejemplos de imágenes y resultados
│     └─ ...
│  └─ 🎞️ Inference_app.png       # Captura de la app demo de inferencia
|
├─ 📂 training/                  # Archivos para entrenamiento del modelo 
│  ├─ 🐍 __init__.py       
│  ├─ 🐍 augmentations.py    
│  ├─ 🐍 callbacks.py        
│  ├─ 🐍 data_loader.py      
│  ├─ 🐍 mlflow_utils.py     
│  ├─ 🐍 model_builder.py    
│  ├─ 🐍 preprocessing.py    
│  ├─ 🐍 train.py            
│  └─ 🐍 trainer_utils.py    
|
├─ 📦 requirements.txt            # Dependencias Python
└─ 📘 README.md                   

   













