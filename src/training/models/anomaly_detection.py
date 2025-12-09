"""
Anomaly Detection Models

Implementations of outlier and anomaly detection models.
"""

from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


class BaseAnomalyModel(ABC):
    """Base class for anomaly detection models."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.config = kwargs
        self.predictions_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies. Returns 1 for normal, -1 for anomaly."""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        pass


class IsolationForestModel(BaseAnomalyModel):
    """Isolation Forest Anomaly Detection Model."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42, **kwargs):
        super().__init__(contamination=contamination, random_state=random_state, **kwargs)
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            **kwargs
        )
    
    def fit(self, X: np.ndarray) -> None:
        """Train the Isolation Forest model."""
        self.predictions_ = self.model.fit_predict(X)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies. Returns 1 for normal, -1 for anomaly."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores. Lower scores indicate anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decision function")
        return self.model.decision_function(X)
    
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        if not self.is_fitted:
            self.fit(X)
        
        predictions = self.predictions_
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions)
        
        return {
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'contamination': self.model.contamination,
            'n_samples': len(predictions),
        }


class OneClassSVMModel(BaseAnomalyModel):
    """One-Class SVM Anomaly Detection Model."""
    
    def __init__(self, nu: float = 0.05, kernel: str = 'rbf', **kwargs):
        super().__init__(nu=nu, kernel=kernel, **kwargs)
        self.model = OneClassSVM(nu=nu, kernel=kernel, **kwargs)
    
    def fit(self, X: np.ndarray) -> None:
        """Train the One-Class SVM model."""
        self.predictions_ = self.model.fit_predict(X)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Detect anomalies. Returns 1 for normal, -1 for anomaly."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores. Lower scores indicate anomalies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decision function")
        return self.model.decision_function(X)
    
    def evaluate(self, X: np.ndarray) -> dict:
        """Evaluate model performance."""
        if not self.is_fitted:
            self.fit(X)
        
        predictions = self.predictions_
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions)
        
        return {
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'nu': self.model.nu,
            'kernel': self.model.kernel,
            'n_samples': len(predictions),
        }
