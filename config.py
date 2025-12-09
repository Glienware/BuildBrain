"""
Configuration file for BuildBrain ML System.

Defines:
1. Logging configuration
2. Color schemes
3. Default hyperparameters
4. Model categories
5. Training defaults
"""

import logging
import os
from datetime import datetime

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"buildbrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# COLOR SCHEME - ANDROID STUDIO DARK
# ============================================================================

COLORS = {
    # Primary backgrounds
    "bg_primary": "#0D0D0D",
    "bg_secondary": "#1A1A1A",
    "bg_tertiary": "#262626",
    
    # Borders
    "border": "#2D2D2D",
    "border_light": "#3D3D3D",
    
    # Accents
    "accent_green": "#3DDC84",      # Primary action
    "accent_blue": "#82B1FF",       # Secondary action
    "accent_orange": "#FFB74D",     # Warnings
    "accent_red": "#FF6B6B",        # Errors
    "accent_purple": "#BB86FC",     # Highlight
    
    # Text colors
    "text_primary": "#CCCCCC",
    "text_secondary": "#AAAAAA",
    "text_tertiary": "#888888",
    "text_disabled": "#666666",
}

# ============================================================================
# MODEL CATEGORIES
# ============================================================================

MODEL_CATEGORIES = {
    "supervised": {
        "name": "Aprendizaje Supervisado",
        "description": "Modelos para regresión y clasificación",
        "icon": "📊",
        "color": COLORS["accent_blue"],
    },
    "unsupervised": {
        "name": "Aprendizaje No Supervisado",
        "description": "Modelos para clustering y reducción dimensional",
        "icon": "🔍",
        "color": COLORS["accent_green"],
    },
    "anomaly_detection": {
        "name": "Detección de Anomalías",
        "description": "Modelos para detectar outliers",
        "icon": "⚠️",
        "color": COLORS["accent_orange"],
    },
    "deep_learning": {
        "name": "Deep Learning",
        "description": "Redes neuronales profundas (PyTorch)",
        "icon": "🧠",
        "color": COLORS["accent_purple"],
    },
}

# ============================================================================
# DEFAULT HYPERPARAMETERS
# ============================================================================

DEFAULT_HYPERPARAMETERS = {
    # Supervised - Regression
    "LinearRegression": {},
    "RandomForestRegressor": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42,
    },
    "XGBoostRegressor": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "random_state": 42,
    },
    "SVR": {
        "kernel": "rbf",
        "C": 1.0,
        "epsilon": 0.1,
    },
    
    # Supervised - Classification
    "LogisticRegression": {
        "max_iter": 200,
        "C": 1.0,
        "random_state": 42,
    },
    "RandomForestClassifier": {
        "n_estimators": 100,
        "max_depth": None,
        "random_state": 42,
    },
    "XGBoostClassifier": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "random_state": 42,
    },
    "KNN": {
        "n_neighbors": 5,
        "weights": "uniform",
    },
    "SVM": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
    },
    "NaiveBayes": {},
    
    # Unsupervised - Clustering
    "KMeans": {
        "n_clusters": 3,
        "init": "k-means++",
        "n_init": 10,
        "random_state": 42,
    },
    "DBSCAN": {
        "eps": 0.5,
        "min_samples": 5,
    },
    "MeanShift": {
        "bandwidth": None,
    },
    "GaussianMixture": {
        "n_components": 3,
        "covariance_type": "full",
        "random_state": 42,
    },
    
    # Unsupervised - Dimensionality Reduction
    "PCA": {
        "n_components": 2,
    },
    "TSNE": {
        "n_components": 2,
        "perplexity": 30,
        "n_iter": 1000,
        "random_state": 42,
    },
    "UMAP": {
        "n_components": 2,
        "n_neighbors": 15,
        "min_dist": 0.1,
    },
    
    # Anomaly Detection
    "IsolationForest": {
        "contamination": 0.1,
        "random_state": 42,
    },
    "OneClassSVM": {
        "kernel": "rbf",
        "gamma": "auto",
        "nu": 0.05,
    },
    
    # Deep Learning
    "ResNet18": {
        "num_classes": 10,
        "pretrained": False,
    },
    "ResNet34": {
        "num_classes": 10,
        "pretrained": False,
    },
    "ResNet50": {
        "num_classes": 10,
        "pretrained": False,
    },
    "PyTorchCNN": {
        "num_classes": 10,
        "num_channels": 3,
    },
    "PyTorchMLP": {
        "num_classes": 10,
        "input_size": 784,
        "hidden_size": 256,
    },
}

# ============================================================================
# TRAINING DEFAULTS
# ============================================================================

TRAINING_DEFAULTS = {
    "supervised": {
        "validation_split": 0.2,
        "test_split": 0.2,
    },
    "unsupervised": {
        "test_split": 0.2,
    },
    "anomaly_detection": {
        "validation_split": 0.2,
        "test_split": 0.2,
    },
    "deep_learning": {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "validation_split": 0.2,
        "optimizer": "adam",
        "loss_function": "cross_entropy",
    },
}

# ============================================================================
# PREPROCESSING DEFAULTS
# ============================================================================

PREPROCESSING_CONFIG = {
    "normalize": True,
    "standardize": True,
    "handle_missing": "mean",  # Options: mean, median, drop
    "outlier_detection": False,
    "feature_scaling": "standard",  # Options: standard, minmax, robust
}

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

PROJECT_STRUCTURE = {
    "models": "Modelos entrenados",
    "data": "Datos del proyecto",
    "logs": "Logs de entrenamiento",
    "plots": "Gráficos y visualizaciones",
    "config": "Archivos de configuración",
}

# ============================================================================
# SUPPORTED FILE FORMATS
# ============================================================================

SUPPORTED_FORMATS = {
    "data": [".csv", ".xlsx", ".json", ".parquet", ".pickle"],
    "models": [".pkl", ".h5", ".pth", ".onnx"],
    "config": [".json", ".yaml", ".txt"],
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_logger(name):
    """Get a logger instance."""
    return logging.getLogger(name)


def log_model_training(model_type, config, metrics):
    """Log model training information."""
    logger.info(f"Model: {model_type}")
    logger.info(f"Config: {config}")
    logger.info(f"Metrics: {metrics}")


def get_color(name):
    """Get color by name."""
    return COLORS.get(name, "#CCCCCC")


def get_model_category_info(category):
    """Get information about a model category."""
    return MODEL_CATEGORIES.get(category, {})


def get_default_hyperparameters(model_type):
    """Get default hyperparameters for a model."""
    return DEFAULT_HYPERPARAMETERS.get(model_type, {})


if __name__ == "__main__":
    print("BuildBrain Configuration Module")
    print(f"Log file: {LOG_FILE}")
    print(f"Available categories: {list(MODEL_CATEGORIES.keys())}")
    print(f"Available models: {list(DEFAULT_HYPERPARAMETERS.keys())}")
