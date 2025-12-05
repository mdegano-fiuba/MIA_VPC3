# 🐶🐱 Clasificador de Perros vs Gatos con MobileViT  
### Entrenamiento → Docker + MLflow • Inferencia → Hugging Face Spaces + Gradio  
---

## 🚀 Resumen del proyecto

Este repositorio contiene un pipeline completo, modular y preproductivo para un modelo de visión por computadora basado en **MobileViT**, fine-tuneado para clasificar **gatos** y **perros**.  

Incluye:

- 🎯 **Entrenamiento modular** (transformers + Trainer)
- 🧱 **Arquitectura limpia y escalable**
- 📦 **Entrenamiento en Docker**
- 📊 **MLflow** para tracking, métricas, artefactos y modelos versionados
- 🌐 **Inferencia Web** con **Gradio**
- 🚀 **Deployment en Hugging Face Spaces**
- 🧪 **Tests automatizados** (pytest)
- 🔁 **CI/CD con GitHub Actions**

---

## 📁 Estructura del proyecto

project/
├─ training/
│   ├─ __init__.py
│   ├─ train.py
│   ├─ data_loader.py
│   ├─ augmentations.py
│   ├─ model_builder.py
│   ├─ trainer_utils.py
│   └─ mlflow_utils.py
│
├─ evaluation/
│   ├─ __init__.py
│   ├─ evaluate.py
│   ├─ metrics.py
│   └─ plots.py
│
├─ inference/
│   ├─ __init__.py
│   └─ app.py
│
├─ model/
│   └─ mobilevit_cats_dogs.pt
│
├─ configs/
│   └─ config.yaml
│
└─ requirements.txt



