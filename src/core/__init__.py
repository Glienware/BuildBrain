"""
Core module for BuildBrain application.

Contains business logic classes for project management, dataset handling, and training.
"""

from .project_config import ProjectConfig
from .dataset_manager import DatasetManager
from .pytorch_trainer import PyTorchTrainer

__all__ = [
    'ProjectConfig',
    'DatasetManager',
    'PyTorchTrainer'
]