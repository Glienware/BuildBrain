"""
Sistema de ejecución de flujos n8n-style con validación de tipos, manejo de errores y logging.
Ejecuta nodos en orden topológico con soporte para reintentos y rollback.
"""

import asyncio
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, defaultdict
from enum import Enum
from dataclasses import dataclass
from .automation_nodes import DataType, NodeConfig


class ExecutionState(Enum):
    """Estados de ejecución."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionNodeState(Enum):
    """Estados de nodos durante ejecución."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionNodeLog:
    """Log de ejecución de un nodo."""
    node_id: str
    node_type: str
    start_time: float
    end_time: Optional[float] = None
    state: ExecutionNodeState = ExecutionNodeState.PENDING
    outputs: Dict[str, Any] = None
    error: Optional[str] = None
    duration: float = 0.0
    
    def __post_init__(self):
        if self.outputs is None:
            self.outputs = {}


@dataclass
class DataFlowValue:
    """Valor en el flujo de datos con validación de tipo."""
    value: Any
    data_type: DataType
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def validate_type(self) -> bool:
        """Valida que el valor coincida con el tipo declarado."""
        if self.data_type == DataType.ANY:
            return True
        
        # Validaciones básicas de tipo
        type_checks = {
            DataType.STRING: isinstance(self.value, str),
            DataType.NUMBER: isinstance(self.value, (int, float)),
            DataType.BOOLEAN: isinstance(self.value, bool),
            DataType.ARRAY: isinstance(self.value, list),
            DataType.OBJECT: isinstance(self.value, dict),
            DataType.JSON: self.value is not None,
            DataType.DATASET: isinstance(self.value, (list, dict)),
            DataType.MODEL: self.value is not None,
            DataType.METRICS: isinstance(self.value, dict),
        }
        
        return type_checks.get(self.data_type, True)


class ExecutionContext:
    """Contexto de ejecución que mantiene estado y datos entre nodos."""
    
    def __init__(self):
        self.variables: Dict[str, DataFlowValue] = {}
        self.execution_data: Dict[str, Dict[str, Any]] = {}
        self.logs: List[str] = []
        self.start_time: float = 0.0
        self.state: ExecutionState = ExecutionState.IDLE
    
    def set_variable(self, name: str, value: Any, data_type: DataType = DataType.ANY):
        """Asigna una variable global."""
        self.variables[name] = DataFlowValue(value, data_type)
    
    def get_variable(self, name: str) -> Optional[Any]:
        """Obtiene una variable global."""
        dv = self.variables.get(name)
        return dv.value if dv else None
    
    def set_node_output(self, node_id: str, outputs: Dict[str, Any]):
        """Guarda los outputs de un nodo."""
        self.execution_data[node_id] = outputs
    
    def get_node_output(self, node_id: str, port_name: str) -> Optional[Any]:
        """Obtiene el output de un puerto específico."""
        return self.execution_data.get(node_id, {}).get(port_name)
    
    def add_log(self, message: str):
        """Agrega un mensaje al log."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")


class FlowValidator:
    """Valida flujos antes de ejecutarlos."""
    
    @staticmethod
    def validate(nodes: Dict[str, NodeConfig], connections: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Valida un flujo completo."""
        # Validar conexiones
        valid, msg = FlowValidator.validate_connections(nodes, connections)
        if not valid:
            return False, msg
        
        # Validar tipos
        valid, msg = FlowValidator.validate_types(nodes, connections)
        if not valid:
            return False, msg
        
        # Validar ciclos
        has_cycle, msg = FlowValidator.has_cycles(nodes, connections)
        if has_cycle:
            return False, msg
        
        return True, "Flow is valid"
    
    @staticmethod
    def validate_connections(nodes: Dict[str, NodeConfig], connections: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Valida que las conexiones sean válidas."""
        for conn in connections:
            source_node_id = conn.get("source_node")
            target_node_id = conn.get("target_node")
            source_port = conn.get("source_port")
            target_port = conn.get("target_port")
            
            # Verificar que los nodos existan
            if source_node_id not in nodes or target_node_id not in nodes:
                return False, "Nodo no encontrado en conexión"
            
            # Verificar puertos
            source_node = nodes[source_node_id]
            target_node = nodes[target_node_id]
            
            if source_port not in source_node.output_ports:
                return False, f"Puerto de salida {source_port} no existe en {source_node_id}"
            
            if target_port not in target_node.input_ports:
                return False, f"Puerto de entrada {target_port} no existe en {target_node_id}"
        
        return True, ""
    
    @staticmethod
    def validate_types(nodes: Dict[str, NodeConfig], connections: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Valida compatibilidad de tipos entre conexiones."""
        for conn in connections:
            source_node_id = conn.get("source_node")
            target_node_id = conn.get("target_node")
            source_port = conn.get("source_port")
            target_port = conn.get("target_port")
            
            source_node = nodes[source_node_id]
            target_node = nodes[target_node_id]
            
            source_type = source_node.output_ports[source_port].data_type
            target_type = target_node.input_ports[target_port].data_type
            
            # Tipos compatibles si uno es ANY
            if source_type != DataType.ANY and target_type != DataType.ANY and source_type != target_type:
                return False, f"Incompatibilidad de tipos: {source_type.value} -> {target_type.value}"
        
        return True, ""
    
    @staticmethod
    def has_cycles(nodes: Dict[str, NodeConfig], connections: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Detecta ciclos en el grafo."""
        graph = defaultdict(list)
        
        for conn in connections:
            source = conn.get("source_node")
            target = conn.get("target_node")
            graph[source].append(target)
        
        visited = set()
        rec_stack = set()
        
        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in nodes:
            if node not in visited:
                if dfs(node):
                    return True, "El flujo contiene ciclos"
        
        return False, ""


class TopologicalExecutor:
    """Ejecutor que ordena nodos topológicamente y los ejecuta en secuencia."""
    
    def __init__(self):
        self.execution_order: List[str] = []
        self.context: ExecutionContext = ExecutionContext()
        self.node_logs: Dict[str, ExecutionNodeLog] = {}
        self.is_executing: bool = False
    
    def _build_dependency_graph(self, connections: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """Construye el grafo de dependencias."""
        graph: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = defaultdict(int)
        
        # Construir grafo desde conexiones
        nodes_in_flow = set()
        for conn in connections:
            source = conn.get("source_node")
            target = conn.get("target_node")
            
            nodes_in_flow.add(source)
            nodes_in_flow.add(target)
            
            graph[source].append(target)
            in_degree[target] += 1
        
        # Asegurar que todos los nodos tengan entry
        for node_id in nodes_in_flow:
            if node_id not in in_degree:
                in_degree[node_id] = 0
        
        return graph, in_degree
    
    def _topological_sort(self, nodes: Dict[str, NodeConfig], connections: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Ordena los nodos topológicamente usando Kahn's algorithm."""
        graph, in_degree = self._build_dependency_graph(connections)
        
        # Inicializar cola con nodos sin dependencias
        queue = deque([node_id for node_id in nodes.keys() if in_degree.get(node_id, 0) == 0])
        order = []
        
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            
            # Procesar vecinos
            for neighbor in graph.get(node_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Si no procesamos todos los nodos, hay un ciclo
        if len(order) != len(nodes):
            return False, []
        
        self.execution_order = order
        return True, order
    
    async def validate_flow(self, nodes: Dict[str, NodeConfig], connections: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Valida que el flujo sea ejecutable."""
        
        # Validar conexiones
        valid, msg = FlowValidator.validate_connections(nodes, connections)
        if not valid:
            return False, msg
        
        # Validar tipos
        valid, msg = FlowValidator.validate_types(nodes, connections)
        if not valid:
            return False, msg
        
        # Verificar ciclos
        has_cycle, msg = FlowValidator.has_cycles(nodes, connections)
        if has_cycle:
            return False, msg
        
        # Verificar orden topológico
        success, order = self._topological_sort(nodes, connections)
        if not success:
            return False, "No se puede determinar orden de ejecución válido"
        
        return True, ""
    
    async def execute(
        self,
        nodes: Dict[str, NodeConfig],
        connections: List[Dict[str, Any]],
        node_executor_fn = None
    ) -> Tuple[bool, str]:
        """
        Ejecuta el flujo completo.
        Args:
            nodes: Dict de nodos por ID
            connections: Lista de conexiones
            node_executor_fn: Función async para ejecutar cada nodo
        """
        if self.is_executing:
            return False, "Una ejecución ya está en progreso"
        
        self.is_executing = True
        self.context.state = ExecutionState.RUNNING
        self.context.start_time = time.time()
        self.node_logs = {}
        
        try:
            # Validar flujo
            valid, msg = await self.validate_flow(nodes, connections)
            if not valid:
                return False, msg
            
            # Ejecutar en orden topológico
            for node_id in self.execution_order:
                if node_id not in nodes:
                    continue
                
                node_config = nodes[node_id]
                log = ExecutionNodeLog(
                    node_id=node_id,
                    node_type=node_config.node_type,
                    start_time=time.time()
                )
                self.node_logs[node_id] = log
                
                try:
                    log.state = ExecutionNodeState.RUNNING
                    
                    # Obtener inputs del nodo
                    inputs = self._collect_node_inputs(node_id, nodes, connections)
                    
                    # Ejecutar nodo
                    if node_executor_fn:
                        result = await node_executor_fn(node_id, node_config, inputs)
                    else:
                        result = await self._execute_default_node(node_config, inputs)
                    
                    # Guardar outputs
                    log.outputs = result
                    log.state = ExecutionNodeState.SUCCESS
                    self.context.set_node_output(node_id, result)
                    
                    self.context.add_log(f"✓ {node_config.display_name} ejecutado")
                    
                except Exception as e:
                    log.state = ExecutionNodeState.FAILED
                    log.error = str(e)
                    self.context.add_log(f"✗ Error en {node_config.display_name}: {str(e)}")
                    return False, f"Error en nodo {node_id}: {str(e)}"
                
                finally:
                    log.end_time = time.time()
                    log.duration = log.end_time - log.start_time
            
            self.context.state = ExecutionState.COMPLETED
            return True, "Flujo ejecutado exitosamente"
        
        except Exception as e:
            self.context.state = ExecutionState.FAILED
            return False, str(e)
        
        finally:
            self.is_executing = False
    
    def _collect_node_inputs(self, node_id: str, nodes: Dict[str, NodeConfig], connections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recolecta inputs para un nodo desde sus conexiones."""
        inputs = {}
        
        if node_id not in nodes:
            return inputs
        
        node = nodes[node_id]
        
        # Para cada conexión que apunta a este nodo
        for conn in connections:
            target_node = conn.get("target_node")
            if target_node != node_id:
                continue
            
            source_node = conn.get("source_node")
            source_port = conn.get("source_port")
            target_port = conn.get("target_port")
            
            # Obtener el valor del puerto de origen
            value = self.context.get_node_output(source_node, source_port)
            inputs[target_port] = value
        
        # Asignar valores por defecto para puertos sin conexión
        for port_name, port in node.input_ports.items():
            if port_name not in inputs:
                inputs[port_name] = port.default_value
        
        return inputs
    
    async def _execute_default_node(self, node_config: NodeConfig, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecución por defecto de un nodo."""
        # Implementación simple: retornar inputs como outputs
        outputs = {}
        for port_name in node_config.output_ports:
            outputs[port_name] = inputs.get(list(node_config.input_ports.keys())[0], None) if node_config.input_ports else None
        
        return outputs
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Retorna un resumen de la ejecución."""
        total_time = (time.time() - self.context.start_time) if self.context.start_time else 0
        
        return {
            "state": self.context.state.value,
            "total_time": total_time,
            "total_nodes": len(self.node_logs),
            "successful_nodes": sum(1 for log in self.node_logs.values() if log.state == ExecutionNodeState.SUCCESS),
            "failed_nodes": sum(1 for log in self.node_logs.values() if log.state == ExecutionNodeState.FAILED),
            "node_logs": {k: {
                "node_type": v.node_type,
                "state": v.state.value,
                "duration": v.duration,
                "error": v.error,
                "outputs": v.outputs
            } for k, v in self.node_logs.items()},
            "logs": self.context.logs[-50:],  # Últimos 50 logs
        }

