"""
Machine Learning Models Package

Contains implementations of various ML models for different tasks.
"""

from .supervised_models import (
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

from .unsupervised_models import (
    KMeansModel,
    DBSCANModel,
    GaussianMixtureModel,
    PCAModel,
    TSNEModel,
    UMAPModel,
)

from .anomaly_detection import (
    IsolationForestModel,
    OneClassSVMModel,
)

from .deep_learning import (
    ResNet18Model,
    ResNet34Model,
    ResNet50Model,
    PyTorchCNNModel,
    PyTorchMLPModel,
)

__all__ = [
    # Supervised
    'LinearRegressionModel',
    'RandomForestRegressorModel',
    'XGBoostRegressorModel',
    'SVRModel',
    'LogisticRegressionModel',
    'RandomForestClassifierModel',
    'XGBoostClassifierModel',
    'KNNModel',
    'SVMModel',
    'NaiveBayesModel',
    # Unsupervised
    'KMeansModel',
    'DBSCANModel',
    'GaussianMixtureModel',
    'PCAModel',
    'TSNEModel',
    'UMAPModel',
    # Anomaly Detection
    'IsolationForestModel',
    'OneClassSVMModel',
    # Deep Learning
    'ResNet18Model',
    'ResNet34Model',
    'ResNet50Model',
    'PyTorchCNNModel',
    'PyTorchMLPModel',
]
