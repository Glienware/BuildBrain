"""
PyTorch Trainer

Handles model training, data loading, and training workflows.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
import threading
import time
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import json

class ImageDataset(Dataset):
    """Custom dataset for image classification."""

    def __init__(self, dataset_path: str, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        self._load_dataset()

    def _load_dataset(self):
        """Load dataset structure."""
        if not os.path.exists(self.dataset_path):
            return

        # Get class names from directories
        self.classes = sorted([d for d in os.listdir(self.dataset_path)
                              if os.path.isdir(os.path.join(self.dataset_path, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load all image paths and labels
        for class_name in self.classes:
            class_path = os.path.join(self.dataset_path, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Return a placeholder image if loading fails
            placeholder = torch.zeros(3, 224, 224)
            return placeholder, label


class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""

    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class PyTorchTrainer:
    """
    PyTorch-based trainer for image classification models.
    """

    def __init__(self, project_path: str, log_callback: Callable[[str], None]):
        self.project_path = project_path
        self.dataset_path = os.path.join(project_path, "dataset")
        self.log_callback = log_callback

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_training = False
        self.training_thread = None
        self.class_names = []
        self.training_history = {}
        self.final_metrics = None

        self.log_callback(f"Using device: {self.device}")

    def quick_train(self, epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001, optimizer: str = "adam") -> bool:
        """Quick training with default settings."""
        return self._train_model({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": optimizer
        })

    def advanced_train(self, config: Dict[str, Any]) -> bool:
        """Advanced training with custom settings."""
        return self._train_model(config)

    def _train_model(self, config: Dict[str, Any]) -> bool:
        """Internal training method."""
        try:
            if self.is_training:
                self.log_callback("Training already in progress")
                return False

            self.is_training = True

            # Start training in a separate thread
            self.training_thread = threading.Thread(
                target=self._training_worker,
                args=(config,)
            )
            self.training_thread.daemon = True
            self.training_thread.start()

            return True

        except Exception as e:
            self.log_callback(f"Error starting training: {str(e)}")
            self.is_training = False
            return False

    def _training_worker(self, config: Dict[str, Any]):
        """Training worker thread."""
        try:
            self.log_callback("🚀 Starting training...")

            # Setup data transforms
            data_aug = config.get("data_augmentation", True)
            if data_aug:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            # Load dataset
            self.log_callback("📂 Loading dataset...")
            dataset = ImageDataset(self.dataset_path, transform=transform)

            if len(dataset) == 0:
                self.log_callback("❌ No images found in dataset")
                return

            # Store class names for later use
            self.class_names = dataset.classes
            
            self.log_callback(f"📊 Found {len(dataset)} images in {len(dataset.classes)} classes")

            # Split dataset
            val_split = config.get("validation_split", 0.2)
            train_size = int((1 - val_split) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=0
            )

            # Create model
            self.log_callback("🏗️ Building model...")
            num_classes = len(dataset.classes)
            self.model = SimpleCNN(num_classes).to(self.device)

            # Setup loss function
            loss_func = config.get("loss_function", "cross_entropy")
            if loss_func == "cross_entropy":
                criterion = nn.CrossEntropyLoss()
            elif loss_func == "mse":
                criterion = nn.MSELoss()
            elif loss_func == "bce":
                criterion = nn.BCELoss()
            else:
                criterion = nn.CrossEntropyLoss()

            # Setup optimizer
            weight_decay = config.get("weight_decay", 0.0)
            momentum = config.get("momentum", 0.9)
            if config["optimizer"].lower() == "adam":
                optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"], weight_decay=weight_decay)
            elif config["optimizer"].lower() == "sgd":
                optimizer = optim.SGD(self.model.parameters(), lr=config["learning_rate"], momentum=momentum, weight_decay=weight_decay)
            elif config["optimizer"].lower() == "rmsprop":
                optimizer = optim.RMSprop(self.model.parameters(), lr=config["learning_rate"], weight_decay=weight_decay)
            else:
                optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"], weight_decay=weight_decay)

            # Setup scheduler
            scheduler_type = config.get("scheduler", "none")
            if scheduler_type == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            elif scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
            elif scheduler_type == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
            else:
                scheduler = None

            # Training loop
            self.log_callback(f"🎯 Training for {config['epochs']} epochs...")

            best_accuracy = 0.0
            start_time = time.time()
            patience = config.get("patience", 10)
            early_stopping = config.get("early_stopping", False)
            patience_counter = 0
            
            # Initialize history tracking
            self.training_history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': []
            }

            for epoch in range(config["epochs"]):
                if not self.is_training:  # Allow stopping
                    break

                # Train epoch
                train_loss, train_acc = self._train_epoch(train_loader, self.model, criterion, optimizer, self.device)

                # Validate epoch
                val_loss, val_acc = self._validate_epoch(val_loader, self.model, criterion, self.device)

                # Store in history
                self.training_history['train_loss'].append(train_loss)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['train_acc'].append(train_acc)
                self.training_history['val_acc'].append(val_acc)

                # Step scheduler
                if scheduler:
                    scheduler.step()

                # Calculate ETA
                elapsed = time.time() - start_time
                epochs_remaining = config["epochs"] - epoch - 1
                eta_seconds = (elapsed / (epoch + 1)) * epochs_remaining if epoch > 0 else 0

                self.log_callback(
                    f"📈 Epoch {epoch+1}/{config['epochs']} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
                    f"ETA: {self._format_time(eta_seconds)}"
                )

                # Save best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    patience_counter = 0
                    self._save_model()
                else:
                    patience_counter += 1

                # Early stopping
                if early_stopping and patience_counter >= patience:
                    self.log_callback(f"🛑 Early stopping at epoch {epoch+1}")
                    break

            # Final save
            self._save_model()
            total_time = time.time() - start_time

            # Calculate final metrics
            self.log_callback("📊 Calculando métricas finales...")
            self.final_metrics = self.calculate_metrics(val_loader, self.model, self.device)
            
            # Generate report images
            self.log_callback("🎨 Generando gráficas...")
            metrics_dir = os.path.join(self.project_path, "training_reports")
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Generate confusion matrix
            self.generate_confusion_matrix_image(
                self.final_metrics,
                os.path.join(metrics_dir, "confusion_matrix.png")
            )
            
            # Generate metrics report
            self.generate_metrics_report_image(
                self.final_metrics,
                os.path.join(metrics_dir, "metrics_report.png")
            )
            
            # Generate training history
            self.generate_training_history_image(
                self.training_history,
                os.path.join(metrics_dir, "training_history.png")
            )
            
            # Save metrics as JSON
            metrics_json_path = os.path.join(metrics_dir, "metrics.json")
            try:
                # Convert numpy types to native Python types for JSON serialization
                class_report_serializable = {}
                for key, value in self.final_metrics['class_report'].items():
                    if isinstance(value, dict):
                        class_report_serializable[key] = {
                            k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                            for k, v in value.items()
                        }
                    else:
                        class_report_serializable[key] = float(value) if isinstance(value, (np.floating, float)) else int(value) if isinstance(value, (np.integer, int)) else value
                
                with open(metrics_json_path, 'w') as f:
                    json.dump({
                        'accuracy': float(self.final_metrics['accuracy']),
                        'precision': float(self.final_metrics['precision']),
                        'recall': float(self.final_metrics['recall']),
                        'f1': float(self.final_metrics['f1']),
                        'confusion_matrix': self.final_metrics['confusion_matrix'].tolist(),
                        'class_report': class_report_serializable
                    }, f, indent=4)
                self.log_callback(f"💾 Métricas guardadas en: {metrics_json_path}")
            except Exception as e:
                self.log_callback(f"⚠️ Error guardando JSON de métricas: {str(e)}")

            
            self.log_callback(
                f"✅ Training completed! | Best Accuracy: {best_accuracy:.2%} | "
                f"Total Time: {self._format_time(total_time)}"
            )

        except Exception as e:
            self.log_callback(f"❌ Training error: {str(e)}")
        finally:
            self.is_training = False

    def _train_epoch(self, loader: DataLoader, model: nn.Module,
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    device: torch.device) -> Tuple[float, float]:
        """Train for one epoch."""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / len(loader), correct / total

    def _validate_epoch(self, loader: DataLoader, model: nn.Module,
                       criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
        """Validate for one epoch."""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss / len(loader), correct / total

    def calculate_metrics(self, loader: DataLoader, model: nn.Module, device: torch.device) -> Dict[str, Any]:
        """Calculate detailed metrics including confusion matrix."""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Per-class metrics
        class_report = classification_report(all_labels, all_predictions, 
                                            target_names=self.class_names,
                                            output_dict=True,
                                            zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'class_report': class_report,
            'predictions': all_predictions,
            'labels': all_labels
        }

    def generate_confusion_matrix_image(self, metrics: Dict[str, Any], output_path: str) -> bool:
        """Generate confusion matrix image."""
        try:
            cm = metrics['confusion_matrix']
            
            plt.figure(figsize=(12, 10))
            
            # Create heatmap
            im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
            plt.colorbar(im)
            
            # Set ticks
            tick_marks = np.arange(len(self.class_names))
            plt.xticks(tick_marks, self.class_names, rotation=45, ha='right')
            plt.yticks(tick_marks, self.class_names)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=10)
            
            plt.ylabel('Etiqueta Real', fontsize=12)
            plt.xlabel('Predicción', fontsize=12)
            plt.tight_layout()
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_callback(f"✅ Matriz de confusión guardada en: {output_path}")
            return True
            
        except Exception as e:
            self.log_callback(f"❌ Error generando matriz de confusión: {str(e)}")
            return False

    def generate_metrics_report_image(self, metrics: Dict[str, Any], output_path: str) -> bool:
        """Generate metrics report image."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Reporte de Métricas del Modelo', fontsize=16, fontweight='bold')
            
            # 1. Overall metrics
            ax = axes[0, 0]
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
            metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
            colors = ['#3DDC84', '#82B1FF', '#CF6679', '#FFB74D']
            bars = ax.bar(metrics_names, metrics_values, color=colors)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title('Métricas Generales', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
            # 2. Per-class precision
            ax = axes[0, 1]
            class_names = self.class_names
            precisions = [metrics['class_report'][name]['precision'] for name in class_names]
            ax.barh(class_names, precisions, color='#82B1FF')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Precision', fontsize=11)
            ax.set_title('Precisión por Clase', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # 3. Per-class recall
            ax = axes[1, 0]
            recalls = [metrics['class_report'][name]['recall'] for name in class_names]
            ax.barh(class_names, recalls, color='#CF6679')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Recall', fontsize=11)
            ax.set_title('Recall por Clase', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # 4. Per-class F1
            ax = axes[1, 1]
            f1_scores = [metrics['class_report'][name]['f1-score'] for name in class_names]
            ax.barh(class_names, f1_scores, color='#FFB74D')
            ax.set_xlim(0, 1)
            ax.set_xlabel('F1-Score', fontsize=11)
            ax.set_title('F1-Score por Clase', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_callback(f"✅ Reporte de métricas guardado en: {output_path}")
            return True
            
        except Exception as e:
            self.log_callback(f"❌ Error generando reporte de métricas: {str(e)}")
            return False

    def generate_training_history_image(self, history: Dict[str, List], output_path: str) -> bool:
        """Generate training history plot."""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Loss plot
            axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o', markersize=3)
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2, marker='s', markersize=3)
            axes[0].set_xlabel('Epoch', fontsize=11)
            axes[0].set_ylabel('Loss', fontsize=11)
            axes[0].set_title('Pérdida por Época', fontsize=12, fontweight='bold')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Accuracy plot
            axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2, marker='o', markersize=3)
            axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
            axes[1].set_xlabel('Epoch', fontsize=11)
            axes[1].set_ylabel('Accuracy', fontsize=11)
            axes[1].set_title('Exactitud por Época', fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_callback(f"✅ Gráfico de historial guardado en: {output_path}")
            return True
            
        except Exception as e:
            self.log_callback(f"❌ Error generando gráfico de historial: {str(e)}")
            return False

    def _save_model(self):
        """Save the trained model."""
        try:
            model_path = os.path.join(self.project_path, "modelo.plt")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_class': self.model.__class__.__name__,
                'timestamp': datetime.now().isoformat()
            }, model_path)
            self.log_callback(f"💾 Model saved to: {model_path}")
        except Exception as e:
            self.log_callback(f"❌ Error saving model: {str(e)}")

    def stop_training(self):
        """Stop ongoing training."""
        self.is_training = False
        self.log_callback("🛑 Training stopped by user")

    def load_model(self, model_path: str) -> bool:
        """Load a trained model from file."""
        try:
            if not os.path.exists(model_path):
                self.log_callback(f"❌ Model file not found: {model_path}")
                return False
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load dataset to get number of classes and class names
            dataset = ImageDataset(self.dataset_path, transform=self._get_transforms()[0])
            num_classes = len(dataset.classes)
            self.class_names = dataset.classes
            
            # Create and load model
            self.model = SimpleCNN(num_classes).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.log_callback(f"✅ Model loaded successfully from: {model_path}")
            return True
            
        except Exception as e:
            self.log_callback(f"❌ Error loading model: {str(e)}")
            return False

    def predict_image(self, image_path: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict class for a single image."""
        try:
            if self.model is None:
                return "Error", 0.0, {}
            
            if not os.path.exists(image_path):
                return "Error", 0.0, {}
            
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            transform = self._get_transforms()[0]
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
            predicted_label = self.class_names[predicted_class.item()]
            confidence_value = confidence.item()
            
            # Get all class probabilities
            class_probs = {}
            for i, class_name in enumerate(self.class_names):
                class_probs[class_name] = probabilities[0][i].item()
            
            return predicted_label, confidence_value, class_probs
            
        except Exception as e:
            self.log_callback(f"❌ Error predicting image: {str(e)}")
            return "Error", 0.0, {}

    def _get_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Get image transforms."""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human readable time."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"