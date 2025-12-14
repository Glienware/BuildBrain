"""
Sistema de ejecución de flujos usando topología de DAG (Directed Acyclic Graph).
Ordena los nodos topológicamente y los ejecuta en secuencia.
"""

import asyncio
from typing import Dict, List, Tuple, Any
from collections import deque, defaultdict
from .base import BaseNode, NodeCanvas, NodeState


class FlowExecutor:
    """Ejecutor base de flujos."""
    
    def __init__(self, canvas: NodeCanvas):
        self.canvas = canvas
        self.execution_log: List[Dict[str, Any]] = []
        self.is_executing = False
    
    async def execute(self) -> Tuple[bool, str]:
        """Ejecuta el flujo. Retorna (éxito, mensaje)."""
        raise NotImplementedError("Subclases deben implementar execute()")


class TopologicalExecutor(FlowExecutor):
    """
    Ejecutor que ordena nodos topológicamente y los ejecuta en secuencia.
    Garantiza que los datos fluyan correctamente de entrada a salida.
    """
    
    def __init__(self, canvas: NodeCanvas):
        super().__init__(canvas)
        self.execution_order: List[str] = []
        self.node_inputs: Dict[str, Dict[str, Any]] = {}
    
    def _build_dependency_graph(self) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Construye el grafo de dependencias.
        Retorna (grafo_adyacencia, in_degree).
        """
        graph: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = {node_id: 0 for node_id in self.canvas.nodes.keys()}
        
        # Construir grafo desde conexiones
        for connection in self.canvas.connections:
            source_node_id = connection.source.node_id
            target_node_id = connection.target.node_id
            
            graph[source_node_id].append(target_node_id)
            in_degree[target_node_id] += 1
        
        return graph, in_degree
    
    def _topological_sort(self) -> Tuple[bool, List[str]]:
        """
        Ordena los nodos topológicamente usando Kahn's algorithm.
        Retorna (éxito, orden_de_ejecución).
        """
        graph, in_degree = self._build_dependency_graph()
        
        # Inicializar cola con nodos sin dependencias
        queue = deque([node_id for node_id in self.canvas.nodes.keys() if in_degree[node_id] == 0])
        order = []
        
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            
            # Procesar vecinos
            for neighbor in graph[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Si no procesamos todos los nodos, hay un ciclo
        if len(order) != len(self.canvas.nodes):
            return False, []
        
        self.execution_order = order
        return True, order
    
    def _validate_flow(self) -> Tuple[bool, str]:
        """Valida que el flujo sea ejecutable."""
        valid, msg = self.canvas.connection_manager.validate_flow()
        if not valid:
            return False, msg
        
        # Verificar que no haya ciclos
        success, order = self._topological_sort()
        if not success:
            return False, "El flujo contiene ciclos (no soportado)"
        
        return True, ""
    
    def _collect_node_inputs(self, node_id: str) -> Dict[str, Any]:
        """Recolecta datos de entrada para un nodo desde sus conexiones."""
        inputs = {}
        node = self.canvas.get_node(node_id)
        
        if not node:
            return inputs
        
        # Para cada puerto de entrada, buscar la conexión
        for port_name, port in node.input_ports.items():
            if port.is_connected:
                # Encontrar la conexión
                for connection in self.canvas.connections:
                    if connection.target == port:
                        source_node_id = connection.source.node_id
                        source_node = self.canvas.get_node(source_node_id)
                        
                        if source_node_id in self.node_inputs:
                            source_output = source_node.output_data.get(connection.source.port_name)
                            inputs[port_name] = source_output
                        break
        
        return inputs
    
    async def execute(self) -> Tuple[bool, str]:
        """
        Ejecuta el flujo en orden topológico.
        Retorna (éxito, mensaje).
        """
        if self.is_executing:
            return False, "Una ejecución ya está en progreso"
        
        self.is_executing = True
        self.execution_log = []
        
        try:
            # Validar
            valid, msg = self._validate_flow()
            if not valid:
                return False, msg
            
            # Ejecutar en orden
            for node_id in self.execution_order:
                node = self.canvas.get_node(node_id)
                if not node:
                    continue
                
                # Recolectar inputs
                inputs = self._collect_node_inputs(node_id)
                
                # Ejecutar nodo
                try:
                    result = await asyncio.wait_for(node.execute(inputs), timeout=300)
                    self.node_inputs[node_id] = result
                    
                    # Log
                    self.execution_log.append({
                        "node_id": node_id,
                        "node_type": node.NODE_TYPE,
                        "state": node.state.value,
                        "execution_time": node.execution_time,
                        "error": node.error_message if node.state == NodeState.ERROR else None,
                    })
                    
                    # Si el nodo falló, detener
                    if node.state == NodeState.ERROR:
                        return False, f"Error en nodo {node_id}: {node.error_message}"
                
                except asyncio.TimeoutError:
                    node.state = NodeState.ERROR
                    node.error_message = "Timeout"
                    return False, f"Timeout ejecutando {node_id}"
                except Exception as e:
                    node.state = NodeState.ERROR
                    node.error_message = str(e)
                    return False, str(e)
            
            return True, "Flujo ejecutado exitosamente"
        
        finally:
            self.is_executing = False
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Retorna un resumen de la ejecución."""
        total_time = sum(log.get("execution_time", 0) for log in self.execution_log)
        
        return {
            "total_nodes": len(self.execution_log),
            "successful_nodes": sum(1 for log in self.execution_log if log.get("state") == "success"),
            "failed_nodes": sum(1 for log in self.execution_log if log.get("state") == "error"),
            "total_time": total_time,
            "logs": self.execution_log,
        }


class ParallelExecutor(FlowExecutor):
    """
    Ejecutor que ejecuta nodos en paralelo cuando es posible.
    Nota: Implementación futura para escalabilidad.
    """
    
    def __init__(self, canvas: NodeCanvas, max_workers: int = 4):
        super().__init__(canvas)
        self.max_workers = max_workers
    
    async def execute(self) -> Tuple[bool, str]:
        # TODO: Implementar ejecución paralela con límite de workers
        return False, "ParallelExecutor aún no está implementado"
