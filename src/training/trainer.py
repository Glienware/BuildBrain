"""
Training Module

Handles model training for different algorithms.
"""

import threading
import queue
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from .models import get_model


class Trainer:
    """
    Main trainer class that handles different ML models.
    """

    def __init__(self, model_type, task_type, hyperparameters, dataset_path, log_callback, project_dir=None):
        self.model_type = model_type
        self.task_type = task_type
        self.hyperparameters = hyperparameters
        self.dataset_path = dataset_path
        self.log_callback = log_callback
        self.model = None
        self.is_training = False
        self.project_dir = project_dir or "projects"
        self.saved_model_path = None

    def _save_model(self, model_name):
        """Save the trained model to disk."""
        import os
        os.makedirs(self.project_dir, exist_ok=True)

        if self.model_type.startswith('PyTorch'):
            # Save PyTorch model
            model_path = os.path.join(self.project_dir, f"{model_name}.ptl")
            torch.save(self.model, model_path)
            self.log_callback(f"Model saved as: {model_path}")
            self.saved_model_path = model_path
            return model_path
        else:
            # For sklearn models, we could save with pickle or joblib
            import pickle
            model_path = os.path.join(self.project_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            self.log_callback(f"Model saved as: {model_path}")
            self.saved_model_path = model_path
            return model_path

    def start_training(self):
        """
        Start training in a separate thread.
        """
        self.is_training = True
        thread = threading.Thread(target=self._train)
        thread.start()

    def _train(self):
        """
        Internal training method with real implementation.
        """
        try:
            self.log_callback("Loading dataset...")

            # Check if it's an image dataset or tabular dataset
            if self.dataset_path.endswith('.csv'):
                # Tabular data
                df = pd.read_csv(self.dataset_path)
                # Assume target column is the last one or named 'target'
                if 'target' in df.columns:
                    target_col = 'target'
                else:
                    target_col = df.columns[-1]

                X = df.drop(columns=[target_col]).values
                y = df[target_col].values

                # Handle categorical features
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for i in range(X.shape[1]):
                    if X[:, i].dtype == 'object':
                        X[:, i] = le.fit_transform(X[:, i].astype(str))

                # Convert to numeric
                X = X.astype(float)

                self.log_callback(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

                # Determine if we should use deep learning or traditional ML
                deep_learning_models = ["ResNet-18", "ResNet-50", "ResNet-101", "VGG16", "VGG19",
                                      "EfficientNet-B0", "EfficientNet-B4", "DenseNet-121", "MobileNetV2",
                                      "PyTorch CNN", "PyTorch MLP"]

                if self.model_type in deep_learning_models:
                    self._train_deep_learning_tabular(X, y)
                else:
                    self._train_traditional_ml(X, y)

            else:
                # Image dataset (folder with images)
                self.log_callback("Image datasets not fully implemented yet. Using basic CNN.")
                self._train_image_dataset()

        except Exception as e:
            self.log_callback(f"Error during training: {str(e)}")
            import traceback
            self.log_callback(traceback.format_exc())
        finally:
            self.is_training = False

    def _train_traditional_ml(self, X, y):
        """Train traditional ML models."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.log_callback("Training model...")

        if self.model_type == "LogisticRegression":
            self.model = LogisticRegression(**self.hyperparameters)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.log_callback(f"Accuracy: {acc:.4f}")
            self.log_callback(f"Precision: {report['weighted avg']['precision']:.4f}")
            self.log_callback(f"Recall: {report['weighted avg']['recall']:.4f}")
            self.log_callback(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")

        elif self.model_type == "RandomForest":
            self.model = RandomForestClassifier(**self.hyperparameters)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.log_callback(f"Accuracy: {acc:.4f}")
            self.log_callback(f"Precision: {report['weighted avg']['precision']:.4f}")
            self.log_callback(f"Recall: {report['weighted avg']['recall']:.4f}")
            self.log_callback(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")

        elif self.model_type == "XGBoost":
            self.model = xgb.XGBClassifier(**self.hyperparameters)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            self.log_callback(f"Accuracy: {acc:.4f}")
            self.log_callback(f"Precision: {report['weighted avg']['precision']:.4f}")
            self.log_callback(f"Recall: {report['weighted avg']['recall']:.4f}")
            self.log_callback(f"F1-Score: {report['weighted avg']['f1-score']:.4f}")

        self.log_callback("Training completed successfully!")
        # Save the trained model
        model_path = self._save_model(f"{self.model_type}_model")

    def _train_deep_learning_tabular(self, X, y):
        """Train deep learning models on tabular data."""
        # Get unique classes for classification
        num_classes = len(np.unique(y))

        # Convert to PyTorch tensors
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.hyperparameters.get('batch_size', 32), shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.hyperparameters.get('batch_size', 32))

        # Get model
        self.model = get_model(self.model_type, num_classes=num_classes, input_size=X.shape[1])

        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=self.hyperparameters.get('learning_rate', 0.001))
        criterion = nn.CrossEntropyLoss()

        # Training loop
        epochs = self.hyperparameters.get('epochs', 10)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        self.log_callback(f"Training {self.model_type} on {device}")

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

            train_acc = 100. * correct / total
            self.log_callback(f"Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, Acc={train_acc:.2f}%")

        # Evaluation
        self.model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = self.model(batch_X)
                _, predicted = outputs.max(1)
                test_total += batch_y.size(0)
                test_correct += predicted.eq(batch_y).sum().item()

        test_acc = 100. * test_correct / test_total
        self.log_callback(f"Test Accuracy: {test_acc:.2f}%")
        self.log_callback("Deep learning training completed!")
        # Save the trained model
        model_path = self._save_model(f"{self.model_type}_model")

    def _train_image_dataset(self):
        """Train models on image datasets."""
        self.log_callback("Image dataset training not fully implemented yet.")
        self.log_callback("For now, only basic CNN is supported for image classification.")
        # TODO: Implement image loading and training
        self.log_callback("Please use CSV datasets for current model training.")