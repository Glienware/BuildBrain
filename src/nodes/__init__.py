"""
Módulo de nodos visuales para flujos de machine learning y automation.
Soporta creación, configuración y ejecución de workflows n8n-style.
"""

from .automation_nodes import (
    NodeConfig,
    Port,
    DataType,
    NODE_REGISTRY,
    create_node_config,
    get_all_nodes_by_category
)
from .executor import (
    TopologicalExecutor,
    ExecutionContext,
    ExecutionState,
    ExecutionNodeState,
    FlowValidator
)
from .serializer import FlowSerializer, FlowPersistence, FlowDefinition

__all__ = [
    "NodeConfig",
    "Port",
    "DataType",
    "NODE_REGISTRY",
    "create_node_config",
    "get_all_nodes_by_category",
    "TopologicalExecutor",
    "ExecutionContext",
    "ExecutionState",
    "ExecutionNodeState",
    "FlowValidator",
    "FlowSerializer",
    "FlowPersistence",
    "FlowDefinition"
]
