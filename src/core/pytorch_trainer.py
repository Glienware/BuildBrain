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

            for epoch in range(config["epochs"]):
                if not self.is_training:  # Allow stopping
                    break

                # Train epoch
                train_loss, train_acc = self._train_epoch(train_loader, self.model, criterion, optimizer, self.device)

                # Validate epoch
                val_loss, val_acc = self._validate_epoch(val_loader, self.model, criterion, self.device)

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