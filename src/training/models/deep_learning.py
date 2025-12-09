"""
Deep Learning Models

Implementations of PyTorch-based neural network models including ResNets.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional


class BaseDeepLearningModel(ABC):
    """Base class for deep learning models."""
    
    def __init__(self, input_size: int = 224, num_classes: int = 10, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.input_size = input_size
        self.num_classes = num_classes
        self.is_fitted = False
        self.config = kwargs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.history = {'train_loss': [], 'val_loss': []}
    
    @abstractmethod
    def build_model(self) -> None:
        """Build the neural network model."""
        pass
    
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 10, batch_size: int = 32) -> None:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
        }


# RESNET MODELS

class ResidualBlock(nn.Module):
    """Residual block for ResNet."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet base architecture."""
    
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride):
        layers_list = []
        layers_list.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers_list.append(block(out_channels, out_channels))
        return nn.Sequential(*layers_list)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18Model(BaseDeepLearningModel):
    """ResNet-18 Model."""
    
    def __init__(self, input_size: int = 224, num_classes: int = 10, **kwargs):
        super().__init__(input_size, num_classes, **kwargs)
        self.build_model()
    
    def build_model(self) -> None:
        """Build ResNet-18 model."""
        self.model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes=self.num_classes)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 10, batch_size: int = 32) -> None:
        """Train the ResNet-18 model."""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_x = torch.from_numpy(X_train[i:i+batch_size]).float().to(self.device)
                batch_y = torch.from_numpy(y_train[i:i+batch_size]).long().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X_train) // batch_size)
            self.history['train_loss'].append(avg_loss)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()


class ResNet34Model(BaseDeepLearningModel):
    """ResNet-34 Model."""
    
    def __init__(self, input_size: int = 224, num_classes: int = 10, **kwargs):
        super().__init__(input_size, num_classes, **kwargs)
        self.build_model()
    
    def build_model(self) -> None:
        """Build ResNet-34 model."""
        self.model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=self.num_classes)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 10, batch_size: int = 32) -> None:
        """Train the ResNet-34 model."""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_x = torch.from_numpy(X_train[i:i+batch_size]).float().to(self.device)
                batch_y = torch.from_numpy(y_train[i:i+batch_size]).long().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X_train) // batch_size)
            self.history['train_loss'].append(avg_loss)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()


class ResNet50Model(BaseDeepLearningModel):
    """ResNet-50 Model."""
    
    def __init__(self, input_size: int = 224, num_classes: int = 10, **kwargs):
        super().__init__(input_size, num_classes, **kwargs)
        self.build_model()
    
    def build_model(self) -> None:
        """Build ResNet-50 model."""
        self.model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=self.num_classes)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 10, batch_size: int = 32) -> None:
        """Train the ResNet-50 model."""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_x = torch.from_numpy(X_train[i:i+batch_size]).float().to(self.device)
                batch_y = torch.from_numpy(y_train[i:i+batch_size]).long().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X_train) // batch_size)
            self.history['train_loss'].append(avg_loss)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()


class PyTorchCNNModel(BaseDeepLearningModel):
    """Custom PyTorch CNN Model."""
    
    class CustomCNN(nn.Module):
        def __init__(self, num_classes):
            super(PyTorchCNNModel.CustomCNN, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(256 * 28 * 28, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    def __init__(self, input_size: int = 224, num_classes: int = 10, **kwargs):
        super().__init__(input_size, num_classes, **kwargs)
        self.build_model()
    
    def build_model(self) -> None:
        """Build custom CNN model."""
        self.model = self.CustomCNN(self.num_classes)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 10, batch_size: int = 32) -> None:
        """Train the CNN model."""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_x = torch.from_numpy(X_train[i:i+batch_size]).float().to(self.device)
                batch_y = torch.from_numpy(y_train[i:i+batch_size]).long().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X_train) // batch_size)
            self.history['train_loss'].append(avg_loss)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()


class PyTorchMLPModel(BaseDeepLearningModel):
    """Multi-Layer Perceptron (MLP) Model."""
    
    class MLPNet(nn.Module):
        def __init__(self, input_features, num_classes):
            super(PyTorchMLPModel.MLPNet, self).__init__()
            self.fc1 = nn.Linear(input_features, 512)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(0.3)
            
            self.fc2 = nn.Linear(512, 256)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(0.3)
            
            self.fc3 = nn.Linear(256, 128)
            self.relu3 = nn.ReLU()
            self.dropout3 = nn.Dropout(0.3)
            
            self.fc4 = nn.Linear(128, num_classes)
        
        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.dropout1(x)
            x = self.relu2(self.fc2(x))
            x = self.dropout2(x)
            x = self.relu3(self.fc3(x))
            x = self.dropout3(x)
            x = self.fc4(x)
            return x
    
    def __init__(self, input_size: int = 784, num_classes: int = 10, **kwargs):
        super().__init__(input_size, num_classes, **kwargs)
        self.input_features = input_size
        self.build_model()
    
    def build_model(self) -> None:
        """Build MLP model."""
        self.model = self.MLPNet(self.input_features, self.num_classes)
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            epochs: int = 10, batch_size: int = 32) -> None:
        """Train the MLP model."""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_x = torch.from_numpy(X_train[i:i+batch_size]).float().to(self.device)
                batch_y = torch.from_numpy(y_train[i:i+batch_size]).long().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X_train) // batch_size)
            self.history['train_loss'].append(avg_loss)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
