"""
Clases base para el sistema de nodos visuales.
Define la estructura fundamental de nodos, puertos y conexiones.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4
import json


class NodeState(Enum):
    """Estados posibles de un nodo."""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class NodePort:
    """Representa un puerto de entrada/salida en un nodo."""
    node_id: str
    port_name: str
    port_type: str  # "input" o "output"
    data_type: str  # "dataset", "model", "metrics", "parameters", etc.
    is_connected: bool = False
    
    def __hash__(self):
        return hash((self.node_id, self.port_name, self.port_type))
    
    def __eq__(self, other):
        if not isinstance(other, NodePort):
            return False
        return (self.node_id == other.node_id and 
                self.port_name == other.port_name and 
                self.port_type == other.port_type)


@dataclass
class Connection:
    """Representa una conexión entre dos puertos."""
    source: NodePort
    target: NodePort
    connection_id: str = field(default_factory=lambda: str(uuid4())[:8])
    created_at: float = field(default_factory=lambda: __import__('time').time())
    
    def is_valid(self) -> bool:
        """Valida que la conexión sea válida."""
        return self.validation_error() is None

    def validation_error(self) -> Optional[str]:
        """Retorna mensaje si la conexión no es válida."""
        if self.source.port_type != "output" or self.target.port_type != "input":
            return "Las salidas deben conectarse únicamente a entradas."
        if self.source.data_type != self.target.data_type:
            return "El tipo de dato del puerto de salida no coincide con el de la entrada."
        if self.source.node_id == self.target.node_id:
            return "No puedes conectar un nodo consigo mismo."
        return None


@dataclass
class NodeParam:
    """Parámetro configurable de un nodo."""
    name: str
    param_type: str  # "float", "int", "str", "bool", "choice"
    value: Any
    description: str = ""
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    choices: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "param_type": self.param_type,
            "value": self.value,
            "description": self.description,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "choices": self.choices,
        }


class BaseNode:
    """Clase base para todos los nodos del sistema."""
    
    NODE_TYPE = "base"
    DISPLAY_NAME = "Base Node"
    DESCRIPTION = "Nodo base abstracto"
    
    def __init__(self, node_id: Optional[str] = None, x: float = 0, y: float = 0):
        self.node_id = node_id or str(uuid4())[:12]
        self.x = x
        self.y = y
        self.width = 200
        self.height = 120
        
        self.state = NodeState.IDLE
        self.error_message = ""
        self.execution_time = 0.0
        self.output_data: Dict[str, Any] = {}
        
        # Parámetros configurables
        self.params: Dict[str, NodeParam] = {}
        
        # Puertos
        self.input_ports: Dict[str, NodePort] = {}
        self.output_ports: Dict[str, NodePort] = {}
        
        # Conexiones entrantes y salientes
        self.incoming_connections: Dict[str, Connection] = {}  # port_name -> Connection
        self.outgoing_connections: Dict[str, Connection] = {}  # port_name -> Connection
    
    def add_input_port(self, port_name: str, data_type: str) -> NodePort:
        """Agrega un puerto de entrada."""
        port = NodePort(self.node_id, port_name, "input", data_type)
        self.input_ports[port_name] = port
        return port
    
    def add_output_port(self, port_name: str, data_type: str) -> NodePort:
        """Agrega un puerto de salida."""
        port = NodePort(self.node_id, port_name, "output", data_type)
        self.output_ports[port_name] = port
        return port
    
    def add_param(self, param: NodeParam):
        """Agrega un parámetro configurable."""
        self.params[param.name] = param
    
    def set_param(self, param_name: str, value: Any):
        """Actualiza el valor de un parámetro."""
        if param_name in self.params:
            self.params[param_name].value = value
    
    def get_param(self, param_name: str) -> Any:
        """Obtiene el valor de un parámetro."""
        if param_name in self.params:
            return self.params[param_name].value
        return None
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta la lógica del nodo.
        Debe ser implementado por subclases.
        """
        raise NotImplementedError("Subclases deben implementar execute()")
    
    def validate(self) -> tuple[bool, str]:
        """
        Valida que el nodo esté configurado correctamente.
        Retorna (es_válido, mensaje_error).
        """
        return True, ""
    
    def to_dict(self) -> Dict:
        """Serializa el nodo a diccionario."""
        return {
            "node_id": self.node_id,
            "node_type": self.NODE_TYPE,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "params": {name: param.to_dict() for name, param in self.params.items()},
            "state": self.state.value,
            "output_data": self.output_data,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BaseNode":
        """Deserializa un nodo desde diccionario."""
        node = cls(node_id=data.get("node_id"), x=data.get("x", 0), y=data.get("y", 0))
        
        # Restaurar parámetros
        for param_name, param_data in data.get("params", {}).items():
            param = NodeParam(
                name=param_data["name"],
                param_type=param_data["param_type"],
                value=param_data["value"],
                description=param_data.get("description", ""),
                min_val=param_data.get("min_val"),
                max_val=param_data.get("max_val"),
                choices=param_data.get("choices"),
            )
            node.add_param(param)
        
        return node


class NodeCanvas:
    """Gestor del canvas visual con todos los nodos."""
    
    def __init__(self, width: float = 1200, height: float = 800, dataset_path: Optional[str] = None):
        self.width = width
        self.height = height
        self.nodes: Dict[str, BaseNode] = {}
        self.connections: List[Connection] = []
        self.connection_manager = ConnectionManager(self)
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.dataset_path = dataset_path
    
    def add_node(self, node: BaseNode):
        """Agrega un nodo al canvas."""
        self.nodes[node.node_id] = node
    
    def remove_node(self, node_id: str) -> bool:
        """Remueve un nodo del canvas."""
        if node_id in self.nodes:
            # Remover todas las conexiones asociadas
            self.connections = [
                c for c in self.connections
                if c.source.node_id != node_id and c.target.node_id != node_id
            ]
            del self.nodes[node_id]
            return True
        return False
    
    def get_node(self, node_id: str) -> Optional[BaseNode]:
        """Obtiene un nodo por ID."""
        return self.nodes.get(node_id)
    
    def connect(self, source_port: NodePort, target_port: NodePort) -> bool:
        """Conecta dos puertos."""
        return self.connection_manager.connect(source_port, target_port)

    def get_last_connection_error(self) -> str:
        """Devuelve la última razón de fallo al conectar."""
        return self.connection_manager.last_error
    
    def disconnect(self, connection_id: str) -> bool:
        """Desconecta una conexión."""
        self.connections = [c for c in self.connections if c.connection_id != connection_id]
        return True
    
    def get_connections_for_node(self, node_id: str) -> List[Connection]:
        """Obtiene todas las conexiones de un nodo."""
        return [
            c for c in self.connections
            if c.source.node_id == node_id or c.target.node_id == node_id
        ]
    
    def to_dict(self) -> Dict:
        """Serializa el canvas."""
        return {
            "width": self.width,
            "height": self.height,
            "zoom": self.zoom,
            "pan": {"x": self.pan_x, "y": self.pan_y},
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "connections": [
                {
                    "connection_id": c.connection_id,
                    "source": {
                        "node_id": c.source.node_id,
                        "port_name": c.source.port_name,
                    },
                    "target": {
                        "node_id": c.target.node_id,
                        "port_name": c.target.port_name,
                    },
                }
                for c in self.connections
            ],
        }


class ConnectionManager:
    """Gestor de conexiones entre nodos."""
    
    def __init__(self, canvas: NodeCanvas):
        self.canvas = canvas
        self.last_error: str = ""
    
    def connect(self, source_port: NodePort, target_port: NodePort) -> bool:
        """
        Conecta dos puertos con validación.
        Retorna True si la conexión fue exitosa.
        """
        self.last_error = ""
        connection = Connection(source_port, target_port)
        
        validation_error = connection.validation_error()
        if validation_error:
            self.last_error = validation_error
            return False
        
        # Evitar conexiones duplicadas
        for existing in self.canvas.connections:
            if (existing.source == source_port and existing.target == target_port):
                self.last_error = "Esa conexión ya existe."
                return False
        
        # Evitar ciclos (no soportamos ciclos en DAG)
        if self._would_create_cycle(connection):
            self.last_error = "Esa conexión generaría un ciclo en el flujo."
            return False
        
        self.canvas.connections.append(connection)
        source_port.is_connected = True
        target_port.is_connected = True
        self.last_error = ""
        return True
    
    def _would_create_cycle(self, connection: Connection) -> bool:
        """Verifica si una conexión crearía un ciclo."""
        # BFS para detectar ciclos
        from collections import deque
        
        visited = set()
        queue = deque([connection.target.node_id])
        
        while queue:
            node_id = queue.popleft()
            if node_id == connection.source.node_id:
                return True
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            # Agregar nodos destino de este nodo
            for conn in self.canvas.connections:
                if conn.source.node_id == node_id:
                    queue.append(conn.target.node_id)
        
        return False
    
    def validate_flow(self) -> tuple[bool, str]:
        """Valida que el flujo sea válido para ejecutar."""
        if not self.canvas.nodes:
            return False, "El canvas está vacío."
        
        # Validar cada nodo
        for node in self.canvas.nodes.values():
            valid, msg = node.validate()
            if not valid:
                return False, f"Nodo {node.node_id}: {msg}"
        
        return True, "Flujo válido"
