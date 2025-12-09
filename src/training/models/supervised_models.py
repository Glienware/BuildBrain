"""
Supervised Learning Models

Implementations of regression and classification models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb


class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, **kwargs):
        self.model = None
        self.is_fitted = False
        self.config = kwargs
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        pass
    
    def get_config(self) -> dict:
        """Get model configuration."""
        return self.config


# REGRESSION MODELS

class LinearRegressionModel(BaseModel):
    """Linear Regression Model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression(**kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the linear regression model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import mean_squared_error, r2_score
        
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
        }


class RandomForestRegressorModel(BaseModel):
    """Random Forest Regressor Model."""
    
    def __init__(self, n_estimators: int = 100, **kwargs):
        super().__init__(n_estimators=n_estimators, **kwargs)
        self.model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the random forest regressor."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
        }


class XGBoostRegressorModel(BaseModel):
    """XGBoost Regressor Model."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, **kwargs):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, **kwargs)
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost regressor."""
        self.model.fit(X, y, verbose=False)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
        }


class SVRModel(BaseModel):
    """Support Vector Regression Model."""
    
    def __init__(self, kernel: str = 'rbf', **kwargs):
        super().__init__(kernel=kernel, **kwargs)
        self.model = SVR(kernel=kernel, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVR model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import mean_squared_error, r2_score
        
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mse)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2_score': float(r2),
        }


# CLASSIFICATION MODELS

class LogisticRegressionModel(BaseModel):
    """Logistic Regression Model."""
    
    def __init__(self, max_iter: int = 1000, **kwargs):
        super().__init__(max_iter=max_iter, **kwargs)
        self.model = LogisticRegression(max_iter=max_iter, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the logistic regression model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        }


class RandomForestClassifierModel(BaseModel):
    """Random Forest Classifier Model."""
    
    def __init__(self, n_estimators: int = 100, **kwargs):
        super().__init__(n_estimators=n_estimators, **kwargs)
        self.model = RandomForestClassifier(n_estimators=n_estimators, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the random forest classifier."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        }


class XGBoostClassifierModel(BaseModel):
    """XGBoost Classifier Model."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, **kwargs):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, **kwargs)
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            **kwargs
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost classifier."""
        self.model.fit(X, y, verbose=False)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        }


class KNNModel(BaseModel):
    """K-Nearest Neighbors Classifier Model."""
    
    def __init__(self, n_neighbors: int = 5, **kwargs):
        super().__init__(n_neighbors=n_neighbors, **kwargs)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the KNN model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        }


class SVMModel(BaseModel):
    """Support Vector Machine Classifier Model."""
    
    def __init__(self, kernel: str = 'rbf', **kwargs):
        super().__init__(kernel=kernel, **kwargs)
        self.model = SVC(kernel=kernel, **kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the SVM model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        }


class NaiveBayesModel(BaseModel):
    """Naive Bayes Classifier Model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = GaussianNB(**kwargs)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Naive Bayes model."""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        }
