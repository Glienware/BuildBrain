"""
Project Configuration Manager

Handles loading, saving, and validating .buildb project files.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any


class ProjectConfig:
    """
    Manages project configuration files (.buildb format).
    """

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.config_file = os.path.join(project_path, f"{os.path.basename(project_path)}.buildb")
        self.config_data = self._load_default_config()
        self.load_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration structure."""
        return {
            "project_name": os.path.basename(self.project_path),
            "task_type": "classification",
            "model_type": "PyTorch_CNN",
            "training_settings": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "loss_function": "cross_entropy",
                "scheduler": "none",
                "weight_decay": 0.0,
                "momentum": 0.9,
                "data_augmentation": True,
                "validation_split": 0.2,
                "early_stopping": False,
                "patience": 10,
                "advanced_enabled": False
            },
            "classes": [],
            "dataset_path": None,
            "dataset_type": "image_folder",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }

    def load_config(self) -> bool:
        """Load configuration from .buildb file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
                return True
            else:
                # Create default config if file doesn't exist
                self.save_config()
                return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False

    def save_config(self) -> bool:
        """Save configuration to .buildb file."""
        try:
            self.config_data["last_modified"] = datetime.now().isoformat()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate required fields
        required_fields = ["project_name", "task_type", "model_type", "training_settings"]
        for field in required_fields:
            if field not in self.config_data:
                errors.append(f"Missing required field: {field}")

        # Validate training settings
        if "training_settings" in self.config_data:
            ts = self.config_data["training_settings"]
            if not isinstance(ts.get("epochs", 0), int) or ts.get("epochs", 0) <= 0:
                errors.append("Epochs must be a positive integer")
            if not isinstance(ts.get("batch_size", 0), int) or ts.get("batch_size", 0) <= 0:
                errors.append("Batch size must be a positive integer")
            if not isinstance(ts.get("learning_rate", 0), (int, float)) or ts.get("learning_rate", 0) <= 0:
                errors.append("Learning rate must be a positive number")

        return errors

    def update_setting(self, key: str, value: Any):
        """Update a specific setting."""
        if key in ["epochs", "batch_size", "learning_rate"]:
            self.config_data["training_settings"][key] = value
        elif key in ["optimizer", "advanced_enabled"]:
            self.config_data["training_settings"][key] = value
        elif key in ["project_name", "task_type", "model_type"]:
            self.config_data[key] = value

    def get_setting(self, key: str, default=None):
        """Get a specific setting value."""
        if key in ["epochs", "batch_size", "learning_rate", "optimizer", "advanced_enabled"]:
            return self.config_data.get("training_settings", {}).get(key, default)
        return self.config_data.get(key, default)

    def add_class(self, class_name: str):
        """Add a new class to the project."""
        if class_name not in self.config_data["classes"]:
            self.config_data["classes"].append(class_name)

    def remove_class(self, class_name: str):
        """Remove a class from the project."""
        if class_name in self.config_data["classes"]:
            self.config_data["classes"].remove(class_name)

    def get_classes(self) -> List[str]:
        """Get list of classes."""
        return self.config_data.get("classes", [])

    def get_project_info(self) -> Dict[str, Any]:
        """Get project information summary."""
        return {
            "name": self.config_data.get("project_name", ""),
            "task_type": self.config_data.get("task_type", ""),
            "model_type": self.config_data.get("model_type", ""),
            "classes_count": len(self.config_data.get("classes", [])),
            "epochs": self.config_data.get("training_settings", {}).get("epochs", 0),
            "created_at": self.config_data.get("created_at", ""),
            "last_modified": self.config_data.get("last_modified", "")
        }