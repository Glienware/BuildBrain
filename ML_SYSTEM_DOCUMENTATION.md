# 🧠 BuildBrain ML System - Complete Implementation

## Overview

BuildBrain es una plataforma GUI profesional para machine learning construida con **Flet** y **Python**, que integra:

- **25+ modelos ML pre-implementados** (scikit-learn, XGBoost, PyTorch)
- **Sistema de fábrica (Factory Pattern)** para instanciación de modelos
- **API unificada de entrenamiento (ModelTrainer)** para todos los tipos de modelos
- **Interfaz gráfica moderna** con diseño inspirado en Android Studio Dark
- **Soporte para GPU/CUDA** en modelos de deep learning

---

## 🏗️ Arquitectura del Sistema

### Estructura de Carpetas

```
src/
├── training/
│   ├── models/
│   │   ├── __init__.py                 # Exporta todos los modelos
│   │   ├── supervised_models.py        # 10 modelos supervisados
│   │   ├── unsupervised_models.py      # 7 modelos no supervisados
│   │   ├── anomaly_detection.py        # 2 modelos de detección de anomalías
│   │   └── deep_learning.py            # 5 modelos PyTorch (incluyendo ResNets)
│   ├── model_factory.py                # Factory Pattern para crear modelos
│   ├── model_trainer.py                # API unificada de entrenamiento
│   ├── trainer.py                      # Legacy trainer
│   └── models.py                       # Legacy models
├── gui/
│   ├── welcome_screen.py               # Pantalla de bienvenida con grid de proyectos
│   ├── new_project_wizard.py           # Wizard dinámico (5-7 pasos según modelo)
│   ├── dataset_uploader.py             # Cargador de datasets
│   ├── main_window.py                  # Ventana principal
│   └── ... (otros componentes GUI)
└── ...
```

---

## 📚 Modelos Implementados

### 1. **Supervised Learning (10 modelos)**

Archivo: `src/training/models/supervised_models.py`

| Modelo | Tipo | Casos de Uso |
|--------|------|-------------|
| `LinearRegressionModel` | Regresión | Predicción linear simple |
| `RandomForestRegressorModel` | Regresión | Regresión no-linear robusta |
| `XGBoostRegressorModel` | Regresión | Regresión con boosting |
| `SVRModel` | Regresión | Máquinas de soporte vectorial |
| `LogisticRegressionModel` | Clasificación | Clasificación binaria/multiclase |
| `RandomForestClassifierModel` | Clasificación | Clasificación robusta |
| `XGBoostClassifierModel` | Clasificación | Clasificación con boosting |
| `KNNModel` | Clasificación | K-vecinos más cercanos |
| `SVMModel` | Clasificación | Máquinas de soporte vectorial |
| `NaiveBayesModel` | Clasificación | Clasificación probabilística |

**Métricas de evaluación:**
- Regresión: MSE, RMSE, MAE, R²
- Clasificación: Accuracy, Precision, Recall, F1 Score

### 2. **Unsupervised Learning (7 modelos)**

Archivo: `src/training/models/unsupervised_models.py`

| Modelo | Tipo | Casos de Uso |
|--------|------|-------------|
| `KMeansModel` | Clustering | Agrupación K-medias |
| `DBSCANModel` | Clustering | Clustering basado en densidad |
| `MeanShiftModel` | Clustering | Clustering basado en modos |
| `GaussianMixtureModel` | Clustering | Clustering probabilístico |
| `PCAModel` | Dimensionality Reduction | Reducción a componentes principales |
| `TSNEModel` | Dimensionality Reduction | Visualización t-SNE |
| `UMAPModel` | Dimensionality Reduction | Reducción UMAP |

**Métricas de evaluación:**
- Silhouette Score
- Davies-Bouldin Index
- BIC/AIC (para modelos probabilísticos)
- Explained Variance (para reducción dimensional)

### 3. **Anomaly Detection (2 modelos)**

Archivo: `src/training/models/anomaly_detection.py`

| Modelo | Técnica | Casos de Uso |
|--------|---------|-------------|
| `IsolationForestModel` | Ensemble | Detección de outliers con Isolation Forest |
| `OneClassSVMModel` | SVM | One-class SVM para detección de anomalías |

**Métricas de evaluación:**
- Anomaly Rate
- Contamination Score
- Decision Function Values

### 4. **Deep Learning (5 modelos con PyTorch)**

Archivo: `src/training/models/deep_learning.py`

| Modelo | Tipo | Arquitectura |
|--------|------|-------------|
| `ResNet18Model` | CNN | ResNet con 2 residual blocks |
| `ResNet34Model` | CNN | ResNet con 3-4-6-3 residual blocks |
| `ResNet50Model` | CNN | ResNet con 3-4-6-3 bottleneck blocks |
| `PyTorchCNNModel` | CNN | Red convolucional personalizada (3 capas) |
| `PyTorchMLPModel` | MLP | Red densa de 4 capas |

**Características:**
- ✅ Soporte GPU/CUDA automático
- ✅ Adam optimizer
- ✅ Cross Entropy Loss
- ✅ Training history tracking
- ✅ Batch processing

---

## 🔧 APIs Principales

### ModelFactory - Factory Pattern

```python
from src.training.model_factory import ModelFactory

# Crear un modelo
model = ModelFactory.create_model("RandomForestClassifier", n_estimators=100)

# Obtener categoría del modelo
category = ModelFactory.get_model_category("ResNet18")
# Output: "deep_learning"

# Obtener hiperparámetros por defecto
hyperparams = ModelFactory.get_default_hyperparameters("LogisticRegression")
# Output: {'max_iter': 200, 'C': 1.0, ...}
```

### ModelTrainer - Unified Training API

```python
from src.training.model_trainer import ModelTrainer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear trainer
config = {
    "name": "my_project",
    "model_type": "LogisticRegression",
    "model_category": "supervised",
    "hyperparameters": {"max_iter": 200}
}
trainer = ModelTrainer("LogisticRegression", config)

# Entrenar modelo
trainer.train(X_train, y_train, validation_split=0.2)

# Evaluar modelo
metrics = trainer.evaluate(X_test, y_test)
print(metrics)  # {'accuracy': 0.95, 'precision': 0.94, ...}

# Hacer predicciones
predictions = trainer.predict(X_test)

# Guardar modelo
trainer.save_model("model.pkl")

# Cargar modelo
loaded_trainer = ModelTrainer.load_model("model.pkl", config)
```

---

## 🎨 Interfaz de Usuario (GUI)

### Tema Dark (Android Studio Inspired)

```python
Colors = {
    "bg_primary": "#0D0D0D",      # Fondo principal
    "bg_secondary": "#1A1A1A",    # Tarjetas/Contenedores
    "border": "#2D2D2D",           # Bordes
    "accent_green": "#3DDC84",     # Botones principales
    "accent_blue": "#82B1FF",      # Botones secundarios
    "accent_orange": "#FFB74D",    # Warnings
    "accent_red": "#FF6B6B",       # Errors
    "text_primary": "#CCCCCC",     # Texto principal
    "text_secondary": "#AAAAAA",   # Texto secundario
    "text_tertiary": "#888888",    # Texto terciario
}
```

### Flujos del Wizard

El wizard es **dinámico** y cambia los pasos según el tipo de modelo seleccionado:

#### Supervised Learning (6 pasos)
1. **Project Name** - Nombre del proyecto
2. **Task Config** - Tipo de tarea (Classification/Regression)
3. **Classes Setup** - Definir clases (para clasificación)
4. **Balance Dataset** - ¿Balancear clases?
5. **Dataset Upload** - Cargar datos
6. **Training Logs** - Entrenar y ver resultados

#### Unsupervised Learning (5 pasos)
1. **Project Name** - Nombre del proyecto
2. **Model Selection** - Seleccionar modelo (KMeans, DBSCAN, etc.)
3. **Unsupervised Params** - Parámetros específicos (n_clusters, etc.)
4. **Dataset Upload** - Cargar datos
5. **Training Logs** - Entrenar y ver resultados

#### Anomaly Detection (5 pasos)
1. **Project Name** - Nombre del proyecto
2. **Anomaly Config** - Configurar contamination %
3. **Training Params** - Parámetros de entrenamiento
4. **Dataset Upload** - Cargar datos
5. **Training Logs** - Entrenar y detectar anomalías

#### Deep Learning (7 pasos)
1. **Project Name** - Nombre del proyecto
2. **NN Type** - Tipo de red (ResNet, CNN, MLP)
3. **NN Config** - Arquitectura de la red
4. **Training Params** - Épocas, batch size, learning rate
5. **Dataset Upload** - Cargar datos
6. **Training Logs** - Entrenar con feedback visual
7. **Export Model** - Guardar modelo entrenado

---

## 💻 Ejemplos de Uso

### Ejemplo 1: Clasificación Supervisada

```python
from src.training.model_trainer import ModelTrainer
from src.training.model_factory import ModelFactory

# Crear configuración
config = {
    "name": "iris_classifier",
    "model_type": "RandomForestClassifier",
    "model_category": "supervised",
    "hyperparameters": ModelFactory.get_default_hyperparameters("RandomForestClassifier")
}

# Crear trainer
trainer = ModelTrainer("RandomForestClassifier", config)

# Entrenar
trainer.train(X_train, y_train, validation_split=0.2)

# Evaluar
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")

# Predicciones
predictions = trainer.predict(X_new)
```

### Ejemplo 2: Clustering No Supervisado

```python
# Crear trainer para KMeans
config = {
    "name": "customer_segmentation",
    "model_type": "KMeans",
    "model_category": "unsupervised",
    "hyperparameters": {"n_clusters": 3}
}

trainer = ModelTrainer("KMeans", config)

# Entrenar (sin labels)
trainer.train(X_unlabeled, None)

# Asignar clusters
clusters = trainer.predict(X_new)
print(f"Clusters asignados: {clusters}")

# Evaluar con Silhouette Score
metrics = trainer.evaluate(X_unlabeled, None)
print(f"Silhouette Score: {metrics:.4f}")
```

### Ejemplo 3: Deep Learning con ResNet

```python
import torch

# Crear trainer para ResNet50
config = {
    "name": "image_classifier",
    "model_type": "ResNet50",
    "model_category": "deep_learning",
    "hyperparameters": {
        "num_classes": 10,
        "pretrained": False
    }
}

trainer = ModelTrainer("ResNet50", config)

# Entrenar con épocas
trainer.train(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)

# Resultados
summary = trainer.get_summary()
print(f"Training history: {summary['training_history']}")
```

---

## 🚀 Características Principales

### ✅ Completado

- [x] 25+ modelos ML implementados
- [x] ResNet18, ResNet34, ResNet50 con arquitecturas reales
- [x] Factory Pattern para instanciación
- [x] API unificada (ModelTrainer)
- [x] Soporte GPU/CUDA automático
- [x] Validación y preprocesamiento (StandardScaler)
- [x] Evaluación de modelos con métricas reales
- [x] Persistencia (save/load con pickle)
- [x] Interfaz dinámmica del wizard
- [x] Tema dark profesional
- [x] Integración con GUI

### 🔄 En Progreso

- 🟡 Hiperpámetros en tiempo real (UI ↔ Model)
- 🟡 Callbacks de progreso (entrenamiento en vivo)
- 🟡 Visualización de métricas (gráficos)

### ⭕ Por Implementar

- ⭕ Export modelo a ONNX
- ⭕ Deployment a cloud
- ⭕ Versioning de modelos
- ⭕ Cross-validation automática

---

## 📦 Dependencias

```
flet>=0.21.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
plotly>=5.15.0
joblib>=1.3.0
numpy>=1.24.0
pillow>=10.0.0
imbalanced-learn>=0.11.0
onnx>=1.14.0
onnxruntime>=1.15.0
```

Instalar con:
```bash
pip install -r requirements.txt
```

---

## 🎯 Próximos Pasos

1. **Visualización de Métricas**
   - Gráficos de training loss
   - Curvas de precisión-recall
   - Matrices de confusión

2. **Optimización Automática**
   - Grid Search para hiperparámetros
   - Cross-validation automática

3. **Exportación y Deployment**
   - Export a ONNX
   - Contenedorización con Docker
   - Deployment a Azure/AWS

4. **Mejoras de UX**
   - Progreso en tiempo real
   - Cancelación de entrenamientos
   - Historial de experimentos

---

## 📝 Notas Técnicas

### Jerarquía de Clases

```
BaseModel (Abstract)
├── LinearRegressionModel
├── RandomForestRegressorModel
├── XGBoostRegressorModel
├── SVRModel
├── LogisticRegressionModel
├── RandomForestClassifierModel
├── XGBoostClassifierModel
├── KNNModel
├── SVMModel
└── NaiveBayesModel

BaseUnsupervisedModel (Abstract)
├── KMeansModel
├── DBSCANModel
├── MeanShiftModel
├── GaussianMixtureModel
├── PCAModel
├── TSNEModel
└── UMAPModel

BaseAnomalyDetectionModel (Abstract)
├── IsolationForestModel
└── OneClassSVMModel

BasePyTorchModel (Abstract)
├── ResNet18Model
├── ResNet34Model
├── ResNet50Model
├── PyTorchCNNModel
└── PyTorchMLPModel
```

### Patrones de Diseño Utilizados

1. **Factory Pattern** (ModelFactory)
   - Creación centralizada de modelos
   - Defaults automáticos por tipo

2. **Strategy Pattern** (ModelTrainer)
   - Diferentes estrategias de entrenamiento por categoría
   - API unificada para todos

3. **Template Method** (BaseModel classes)
   - Métodos comunes: fit, predict, evaluate
   - Implementación específica en subclases

---

## 🐛 Troubleshooting

### Error: CUDA no disponible
```python
# Los modelos detectan automáticamente GPU
# Si no hay CUDA, usan CPU
# Ver en los logs: "GPU not available, using CPU"
```

### Error: Módulo no encontrado
```bash
# Reinstalar dependencias
pip install --upgrade -r requirements.txt
```

### Error: Memoria insuficiente
```python
# Reducir batch_size en training
trainer.train(X, y, batch_size=16)
```

---

## 📄 Licencia

BuildBrain © 2024 - Open Source

---

## 👨‍💻 Autor

Developed with ❤️ by GitHub Copilot
