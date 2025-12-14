"""
Módulo de nodos visuales para flujos de machine learning.
Soporta creación, entrenamiento y prueba de modelos mediante interfaces visuales de nodos.
"""

from .base import BaseNode, NodePort, NodeCanvas, ConnectionManager, NodeState
from .types import DatasetNode, ModelNode, TrainingNode, TestNode, OutputNode, NodeFactory
from .executor import FlowExecutor, TopologicalExecutor
from .serializer import FlowSerializer, FlowLoader

__all__ = [
    "BaseNode",
    "NodePort",
    "NodeCanvas",
    "ConnectionManager",
    "NodeState",
    "DatasetNode",
    "ModelNode",
    "TrainingNode",
    "TestNode",
    "OutputNode",
    "NodeFactory",
    "FlowExecutor",
    "TopologicalExecutor",
    "FlowSerializer",
    "FlowLoader",
]
