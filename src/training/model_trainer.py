"""
Model Trainer Integration

High-level API for training models with real functionality.
"""

from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

from .model_factory import ModelFactory


class ModelTrainer:
    """
    High-level trainer for all ML models.
    
    Handles model creation, training, evaluation, and prediction.
    """
    
    def __init__(self, model_type: str, project_data: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model_type: Type of model to train
            project_data: Project configuration dictionary
        """
        self.model_type = model_type
        self.project_data = project_data
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'metrics': {}
        }
    
    def create_model(self, **hyperparams) -> None:
        """Create model with specified hyperparameters."""
        default_params = ModelFactory.get_default_hyperparameters(self.model_type)
        default_params.update(hyperparams)
        
        self.model = ModelFactory.create_model(self.model_type, **default_params)
    
    def preprocess_data(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                       fit_scaler: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess data (normalize/standardize).
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            fit_scaler: Whether to fit the scaler
        
        Returns:
            Preprocessed data and labels
        """
        if fit_scaler:
            X_processed = self.scaler.fit_transform(X)
        else:
            X_processed = self.scaler.transform(X)
        
        return X_processed, y
    
    def train(self, X: np.ndarray, y: np.ndarray,
             validation_split: float = 0.2,
             epochs: int = 10,
             batch_size: int = 32,
             verbose: bool = True,
             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Training features
            y: Training labels
            validation_split: Proportion of data to use for validation
            epochs: Number of epochs (for deep learning)
            batch_size: Batch size (for deep learning)
            verbose: Whether to print progress
            progress_callback: Callback function for progress updates
        
        Returns:
            Training history
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Preprocess
        X_train, y_train = self.preprocess_data(X_train, y_train, fit_scaler=True)
        X_val, y_val = self.preprocess_data(X_val, y_val, fit_scaler=False)
        
        # Create model if not already created
        if self.model is None:
            self.create_model()
        
        # Train based on model type
        category = ModelFactory.get_model_category(self.model_type)
        
        if category == "deep_learning":
            # Train deep learning models
            self.model.fit(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
            self.training_history = self.model.history
        else:
            # Train traditional ML models
            self.model.fit(X_train, y_train)
            
            # Evaluate on training data
            if hasattr(self.model, 'evaluate'):
                train_metrics = self.model.evaluate(X_train, y_train)
                val_metrics = self.model.evaluate(X_val, y_val)
                self.training_history['metrics'] = {
                    'train': train_metrics,
                    'validation': val_metrics
                }
        
        self.is_fitted = True
        
        if verbose and progress_callback:
            progress_callback("Entrenamiento completado")
        
        return self.training_history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Preprocess test data
        X_test, y_test = self.preprocess_data(X_test, y_test, fit_scaler=False)
        
        # Evaluate
        if hasattr(self.model, 'evaluate'):
            metrics = self.model.evaluate(X_test, y_test)
        else:
            metrics = {'status': 'Model does not support evaluation'}
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess
        X, _ = self.preprocess_data(X, fit_scaler=False)
        
        # Predict
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save model to file."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self, filepath: str) -> None:
        """Load model from file."""
        import pickle
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get model and training summary."""
        return {
            'model_type': self.model_type,
            'category': ModelFactory.get_model_category(self.model_type),
            'is_fitted': self.is_fitted,
            'hyperparameters': self.model.get_config() if hasattr(self.model, 'get_config') else {},
            'training_history': self.training_history,
        }
