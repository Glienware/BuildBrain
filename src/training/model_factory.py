"""
Model Factory and Integration

Factory pattern for creating and managing ML models.
"""

from typing import Dict, Any, Optional
import numpy as np

# Import all models
from .models.supervised_models import (
    LinearRegressionModel,
    RandomForestRegressorModel,
    XGBoostRegressorModel,
    SVRModel,
    LogisticRegressionModel,
    RandomForestClassifierModel,
    XGBoostClassifierModel,
    KNNModel,
    SVMModel,
    NaiveBayesModel,
)

from .models.unsupervised_models import (
    KMeansModel,
    DBSCANModel,
    MeanShiftModel,
    GaussianMixtureModel,
    PCAModel,
    TSNEModel,
    UMAPModel,
)

from .models.anomaly_detection import (
    IsolationForestModel,
    OneClassSVMModel,
)

from .models.deep_learning import (
    ResNet18Model,
    ResNet34Model,
    ResNet50Model,
    PyTorchCNNModel,
    PyTorchMLPModel,
)


class ModelFactory:
    """Factory for creating ML models."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> Any:
        """
        Create a model instance based on type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments to pass to model constructor
        
        Returns:
            Model instance
        """
        
        # Supervised Learning - Regression
        if model_type == "linear_regression":
            return LinearRegressionModel(**kwargs)
        elif model_type == "random_forest_regressor":
            return RandomForestRegressorModel(**kwargs)
        elif model_type == "xgboost_regressor":
            return XGBoostRegressorModel(**kwargs)
        elif model_type == "svr":
            return SVRModel(**kwargs)
        
        # Supervised Learning - Classification
        elif model_type == "logistic_regression":
            return LogisticRegressionModel(**kwargs)
        elif model_type == "random_forest_classifier":
            return RandomForestClassifierModel(**kwargs)
        elif model_type == "xgboost_classifier":
            return XGBoostClassifierModel(**kwargs)
        elif model_type == "knn":
            return KNNModel(**kwargs)
        elif model_type == "svm":
            return SVMModel(**kwargs)
        elif model_type == "naive_bayes":
            return NaiveBayesModel(**kwargs)
        
        # Unsupervised Learning - Clustering
        elif model_type == "kmeans":
            return KMeansModel(**kwargs)
        elif model_type == "dbscan":
            return DBSCANModel(**kwargs)
        elif model_type == "mean_shift":
            return MeanShiftModel(**kwargs)
        elif model_type == "gaussian_mixture":
            return GaussianMixtureModel(**kwargs)
        
        # Unsupervised Learning - Dimensionality Reduction
        elif model_type == "pca":
            return PCAModel(**kwargs)
        elif model_type == "tsne":
            return TSNEModel(**kwargs)
        elif model_type == "umap":
            return UMAPModel(**kwargs)
        
        # Anomaly Detection
        elif model_type == "isolation_forest":
            return IsolationForestModel(**kwargs)
        elif model_type == "one_class_svm":
            return OneClassSVMModel(**kwargs)
        
        # Deep Learning - ResNets
        elif model_type == "resnet18":
            return ResNet18Model(**kwargs)
        elif model_type == "resnet34":
            return ResNet34Model(**kwargs)
        elif model_type == "resnet50":
            return ResNet50Model(**kwargs)
        
        # Deep Learning - Custom
        elif model_type == "pytorch_cnn":
            return PyTorchCNNModel(**kwargs)
        elif model_type == "pytorch_mlp":
            return PyTorchMLPModel(**kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_model_category(model_type: str) -> str:
        """Get the category of a model."""
        supervised_models = {
            "linear_regression", "random_forest_regressor", "xgboost_regressor", "svr",
            "logistic_regression", "random_forest_classifier", "xgboost_classifier",
            "knn", "svm", "naive_bayes"
        }
        
        unsupervised_models = {
            "kmeans", "dbscan", "mean_shift", "gaussian_mixture",
            "pca", "tsne", "umap"
        }
        
        anomaly_models = {"isolation_forest", "one_class_svm"}
        
        deep_learning_models = {
            "resnet18", "resnet34", "resnet50",
            "pytorch_cnn", "pytorch_mlp"
        }
        
        if model_type in supervised_models:
            return "supervised"
        elif model_type in unsupervised_models:
            return "unsupervised"
        elif model_type in anomaly_models:
            return "anomaly"
        elif model_type in deep_learning_models:
            return "deep_learning"
        else:
            return "unknown"
    
    @staticmethod
    def get_default_hyperparameters(model_type: str) -> Dict[str, Any]:
        """Get default hyperparameters for a model."""
        
        defaults = {
            # Supervised - Regression
            "linear_regression": {},
            "random_forest_regressor": {"n_estimators": 100, "random_state": 42},
            "xgboost_regressor": {"n_estimators": 100, "learning_rate": 0.1},
            "svr": {"kernel": "rbf", "C": 1.0},
            
            # Supervised - Classification
            "logistic_regression": {"max_iter": 1000},
            "random_forest_classifier": {"n_estimators": 100, "random_state": 42},
            "xgboost_classifier": {"n_estimators": 100, "learning_rate": 0.1},
            "knn": {"n_neighbors": 5},
            "svm": {"kernel": "rbf"},
            "naive_bayes": {},
            
            # Unsupervised - Clustering
            "kmeans": {"n_clusters": 3, "random_state": 42},
            "dbscan": {"eps": 0.5, "min_samples": 5},
            "mean_shift": {},
            "gaussian_mixture": {"n_components": 3, "random_state": 42},
            
            # Unsupervised - Dimensionality Reduction
            "pca": {"n_components": 2},
            "tsne": {"n_components": 2, "perplexity": 30},
            "umap": {"n_components": 2},
            
            # Anomaly Detection
            "isolation_forest": {"contamination": 0.1, "random_state": 42},
            "one_class_svm": {"nu": 0.05},
            
            # Deep Learning
            "resnet18": {"input_size": 224, "num_classes": 10},
            "resnet34": {"input_size": 224, "num_classes": 10},
            "resnet50": {"input_size": 224, "num_classes": 10},
            "pytorch_cnn": {"input_size": 224, "num_classes": 10},
            "pytorch_mlp": {"input_size": 784, "num_classes": 10},
        }
        
        return defaults.get(model_type, {})
