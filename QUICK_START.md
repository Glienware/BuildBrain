# 🚀 BuildBrain ML System - Quick Start Guide

## Instalación Rápida

### 1. Clonar el Repositorio
```bash
git clone <repository-url>
cd BuildBrain
```

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar la Aplicación
```bash
python main.py
```

La aplicación abrirá una ventana con la interfaz gráfica de BuildBrain.

---

## 🎯 Flujo Rápido: Crear tu Primer Modelo

### Paso 1: Bienvenida
1. Ejecuta `python main.py`
2. Se abrirá la pantalla de bienvenida con un botón **"Get Started"**

### Paso 2: Crear Proyecto
1. Click en **"Get Started"**
2. Se abrirá el wizard dinámico
3. **Nombre del proyecto**: ej. `iris_classifier`

### Paso 3: Seleccionar Modelo
Elige uno de los 25+ modelos disponibles:

**Supervisados (para tus datos etiquetados):**
- LogisticRegression (clasificación)
- RandomForestClassifier (clasificación robusta)
- LinearRegression (regresión)
- SVR (regresión no-linear)

**No Supervisados (para descubrir patrones):**
- KMeans (agrupación)
- DBSCAN (agrupación por densidad)
- PCA (reducción dimensional)

**Deep Learning (redes neuronales):**
- ResNet18, ResNet34, ResNet50
- PyTorchCNN (convolucional)
- PyTorchMLP (densa)

**Detección de Anomalías:**
- IsolationForest
- OneClassSVM

### Paso 4: Cargar Dataset
1. Haz click en **"Seleccionar Dataset"**
2. Selecciona tu archivo CSV/XLSX
3. El sistema automáticamente:
   - Detecta features y labels
   - Preprocesa datos (normalización)
   - Divide en train/test

### Paso 5: Entrenar Modelo
1. Haz click en **"Crear Modelo"**
2. El sistema iniciará el entrenamiento
3. Verás en tiempo real:
   - Progreso de entrenamiento
   - Logs de cada paso
   - Métricas de rendimiento

### Paso 6: Ver Resultados
Una vez completado, obtendrás:
- ✅ Accuracy, Precision, Recall, F1 Score (si aplica)
- ✅ Modelo guardado en `projects/<nombre_proyecto>/models/`
- ✅ Logs en `projects/<nombre_proyecto>/logs/`

---

## 💻 Ejemplo Desde Código Python

Si prefieres usar BuildBrain desde Python en lugar de la GUI:

### Clasificación Supervisada
```python
from src.training.model_trainer import ModelTrainer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Cargar datos
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear configuración
config = {
    "name": "mi_proyecto",
    "model_type": "RandomForestClassifier",
    "model_category": "supervised",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10
    }
}

# Crear trainer
trainer = ModelTrainer("RandomForestClassifier", config)

# Entrenar
trainer.train(X_train, y_train, validation_split=0.2)

# Evaluar
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Predicciones
predictions = trainer.predict(X_test[:5])
print(f"Predicciones: {predictions}")

# Guardar modelo
trainer.save_model("models/mi_modelo.pkl")
```

### Clustering No Supervisado
```python
from src.training.model_trainer import ModelTrainer
import numpy as np

# Datos sin etiquetas
X = np.random.randn(100, 2)

config = {
    "name": "clustering",
    "model_type": "KMeans",
    "model_category": "unsupervised",
    "hyperparameters": {"n_clusters": 3}
}

trainer = ModelTrainer("KMeans", config)
trainer.train(X, None)  # None porque no hay labels

# Asignar clusters
clusters = trainer.predict(X[:10])
print(f"Clusters: {clusters}")
```

### Deep Learning (Redes Neuronales)
```python
from src.training.model_trainer import ModelTrainer
import numpy as np

# Datos para deep learning
X_train = np.random.randn(1000, 784)  # 28x28 imágenes (flattened)
y_train = np.random.randint(0, 10, 1000)  # 10 clases

config = {
    "name": "deep_learning_model",
    "model_type": "ResNet50",
    "model_category": "deep_learning",
}

trainer = ModelTrainer("ResNet50", config)

# Entrenar con épocas
trainer.train(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    learning_rate=0.001
)

# Ver historial de entrenamiento
summary = trainer.get_summary()
print(f"Training history: {summary['training_history']}")
```

---

## 📊 Todos los Modelos Disponibles

### Supervisados (10 modelos)
```python
modelos = [
    "LinearRegression",           # Regresión lineal
    "RandomForestRegressor",      # Random Forest para regresión
    "XGBoostRegressor",          # Boosting para regresión
    "SVR",                       # Support Vector Regressor
    "LogisticRegression",        # Clasificación logística
    "RandomForestClassifier",    # Random Forest para clasificación
    "XGBoostClassifier",         # Boosting para clasificación
    "KNN",                       # K-Vecinos más cercanos
    "SVM",                       # Support Vector Machine
    "NaiveBayes",               # Clasificador Naive Bayes
]
```

### No Supervisados (7 modelos)
```python
modelos = [
    "KMeans",              # K-medias
    "DBSCAN",              # Clustering por densidad
    "MeanShift",           # Clustering por modos
    "GaussianMixture",     # Modelos de mezcla gaussiana
    "PCA",                 # Análisis de componentes principales
    "TSNE",                # t-Distributed Stochastic Neighbor Embedding
    "UMAP",                # Uniform Manifold Approximation and Projection
]
```

### Detección de Anomalías (2 modelos)
```python
modelos = [
    "IsolationForest",     # Isolation Forest
    "OneClassSVM",         # SVM de una clase
]
```

### Deep Learning (5 modelos)
```python
modelos = [
    "ResNet18",            # ResNet con 18 capas
    "ResNet34",            # ResNet con 34 capas
    "ResNet50",            # ResNet con 50 capas (bottleneck)
    "PyTorchCNN",          # Red convolucional personalizada
    "PyTorchMLP",          # Red densa multicapa
]
```

---

## 🧪 Ejecutar Tests

Para validar que todo funciona correctamente:

```bash
python test_ml_system.py
```

Esto ejecutará tests de:
- ✅ Instanciación de todos los modelos
- ✅ Entrenamiento de cada modelo
- ✅ Evaluación y métricas
- ✅ Persistencia (save/load)
- ✅ Predicciones

---

## 📁 Estructura de Archivos Después de Crear un Proyecto

```
BuildBrain/
├── projects/
│   └── mi_proyecto/
│       ├── project_config.json      # Configuración del proyecto
│       ├── models/
│       │   └── trained_model.pkl    # Modelo entrenado
│       ├── data/
│       │   ├── train.csv           # Datos de entrenamiento
│       │   └── test.csv            # Datos de prueba
│       └── logs/
│           └── training_log.txt    # Log del entrenamiento
├── models/                          # Modelos globales
├── logs/                            # Logs globales
└── ...
```

---

## ⚙️ Configuración Avanzada

### Cambiar Hiperparámetros

```python
from src.training.model_factory import ModelFactory

# Obtener hiperparámetros por defecto
defaults = ModelFactory.get_default_hyperparameters("RandomForestClassifier")
print(defaults)  
# Output: {'n_estimators': 100, 'max_depth': None, 'random_state': 42}

# Usar hiperparámetros personalizados
config = {
    "name": "custom_params",
    "model_type": "RandomForestClassifier",
    "model_category": "supervised",
    "hyperparameters": {
        "n_estimators": 200,  # Más árboles
        "max_depth": 15,      # Árboles más profundos
        "random_state": 42
    }
}

trainer = ModelTrainer("RandomForestClassifier", config)
```

### Usar GPU/CUDA

Los modelos de PyTorch detectan automáticamente GPU:

```python
import torch

# Verificar si CUDA está disponible
print(f"GPU Available: {torch.cuda.is_available()}")

# Los modelos la usarán automáticamente
trainer = ModelTrainer("ResNet50", config)
# Se verá un log: "GPU available, using CUDA"
```

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'torch'"
```bash
# Instalar dependencias nuevamente
pip install --upgrade -r requirements.txt
```

### Error: "CUDA out of memory"
```python
# Reducir batch size
trainer.train(X, y, batch_size=16)  # En lugar de 32
```

### Error: "No se puede cargar el dataset"
```python
# Asegurate que el archivo es CSV o Excel
# Formatos soportados: .csv, .xlsx, .json, .parquet

# O cargalo manualmente
import pandas as pd
df = pd.read_csv("mi_datos.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
```

---

## 📚 Documentación Completa

Para documentación más detallada, ver:
- `ML_SYSTEM_DOCUMENTATION.md` - Documentación técnica completa
- `example_usage.py` - Ejemplos de código
- `test_ml_system.py` - Tests de integración

---

## 🎓 Pasos Recomendados

**Para principiantes:**
1. Ejecuta `python main.py`
2. Crea un proyecto con LogisticRegression
3. Carga el dataset Iris (viene en scikit-learn)
4. Observa cómo funciona el proceso

**Para usuarios intermedios:**
1. Experimenta con diferentes modelos
2. Ajusta hiperparámetros
3. Compara resultados entre modelos

**Para usuarios avanzados:**
1. Modifica el código de los modelos
2. Añade nuevas métricas
3. Crea modelos personalizados

---

## 🚀 Próximos Pasos

- Exportar modelos a ONNX
- Hacer deploy a Azure/AWS
- Crear pipelines de machine learning
- Automl con búsqueda de hiperparámetros
- Explicabilidad con SHAP

---

¡Ahora estás listo para comenzar! 🎉

Cualquier pregunta o problema, revisa los logs en `logs/` o ejecuta `test_ml_system.py`.
