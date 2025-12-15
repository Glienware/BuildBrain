"""Serialización y deserialización de flujos de automation."""

import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from .automation_nodes import NodeConfig, NODE_REGISTRY, DataType


@dataclass
class FlowDefinition:
    """Definición completa de un flujo de automation."""
    id: str
    name: str
    description: str
    version: str = "1.0"
    nodes: Dict[str, Dict[str, Any]] = None
    connections: List[Dict[str, Any]] = None
    variables: Dict[str, Any] = None
    settings: Dict[str, Any] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        if self.nodes is None:
            self.nodes = {}
        if self.connections is None:
            self.connections = []
        if self.variables is None:
            self.variables = {}
        if self.settings is None:
            self.settings = {}
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            self.updated_at = datetime.now().isoformat()


class FlowSerializer:
    """Serializa y deserializa flujos completos."""
    
    @staticmethod
    def serialize_node(node_id: str, node_config: NodeConfig, position: Dict[str, float], data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Serializa un nodo individual."""
        return {
            "id": node_id,
            "type": node_config.node_type,
            "display_name": node_config.display_name,
            "category": node_config.category,
            "position": position,
            "settings": {**node_config.settings},
            "input_ports": {
                name: {
                    "name": port.name,
                    "data_type": port.data_type.value,
                    "description": port.description,
                    "required": port.required,
                    "default_value": port.default_value
                }
                for name, port in node_config.input_ports.items()
            },
            "output_ports": {
                name: {
                    "name": port.name,
                    "data_type": port.data_type.value,
                    "description": port.description
                }
                for name, port in node_config.output_ports.items()
            },
            "data": data or {}
        }
    
    @staticmethod
    def serialize_connection(source_node: str, source_port: str, target_node: str, target_port: str) -> Dict[str, str]:
        """Serializa una conexión entre nodos."""
        return {
            "id": f"{source_node}.{source_port}->{target_node}.{target_port}",
            "source_node": source_node,
            "source_port": source_port,
            "target_node": target_node,
            "target_port": target_port
        }
    
    @staticmethod
    def serialize_flow(
        flow_id: str,
        flow_name: str,
        nodes: Dict[str, Tuple[NodeConfig, Dict[str, float]]],
        connections: List[Dict[str, str]],
        variables: Dict[str, Any] = None,
        settings: Dict[str, Any] = None
    ) -> FlowDefinition:
        """Serializa un flujo completo."""
        
        serialized_nodes = {}
        for node_id, (config, position) in nodes.items():
            serialized_nodes[node_id] = FlowSerializer.serialize_node(node_id, config, position)
        
        return FlowDefinition(
            id=flow_id,
            name=flow_name,
            description=f"Automation flow: {flow_name}",
            nodes=serialized_nodes,
            connections=connections,
            variables=variables or {},
            settings=settings or {},
            updated_at=datetime.now().isoformat()
        )
    
    @staticmethod
    def to_json(flow_def: FlowDefinition) -> str:
        """Convierte a JSON."""
        return json.dumps({
            "id": flow_def.id,
            "name": flow_def.name,
            "description": flow_def.description,
            "version": flow_def.version,
            "created_at": flow_def.created_at,
            "updated_at": flow_def.updated_at,
            "nodes": flow_def.nodes,
            "connections": flow_def.connections,
            "variables": flow_def.variables,
            "settings": flow_def.settings
        }, indent=2)
    
    @staticmethod
    def from_json(json_str: str) -> Tuple[bool, Optional[FlowDefinition], str]:
        """Lee un flujo desde JSON."""
        try:
            data = json.loads(json_str)
            
            return True, FlowDefinition(
                id=data.get("id"),
                name=data.get("name"),
                description=data.get("description"),
                version=data.get("version", "1.0"),
                nodes=data.get("nodes", {}),
                connections=data.get("connections", []),
                variables=data.get("variables", {}),
                settings=data.get("settings", {}),
                created_at=data.get("created_at"),
                updated_at=data.get("updated_at")
            ), ""
        except Exception as e:
            return False, None, str(e)


class FlowPersistence:
    """Maneja persistencia de flujos en disco."""
    
    def __init__(self, flows_dir: str = "projects/flows"):
        self.flows_dir = Path(flows_dir)
        self.flows_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, flow_def: FlowDefinition, overwrite: bool = False) -> Tuple[bool, str]:
        """Alias para save_flow - Guarda un flujo a disco."""
        return self.save_flow(flow_def, overwrite)
    
    def save_flow(self, flow_def: FlowDefinition, overwrite: bool = False) -> Tuple[bool, str]:
        """Guarda un flujo a disco."""
        try:
            flow_path = self.flows_dir / f"{flow_def.id}.json"
            
            if flow_path.exists() and not overwrite:
                return False, f"El flujo {flow_def.id} ya existe"
            
            json_content = FlowSerializer.to_json(flow_def)
            flow_path.write_text(json_content, encoding="utf-8")
            
            return True, f"Flujo guardado en {flow_path}"
        except Exception as e:
            return False, str(e)
    
    def load_flow(self, flow_id: str) -> Tuple[bool, Optional[FlowDefinition], str]:
        """Carga un flujo desde disco."""
        try:
            flow_path = self.flows_dir / f"{flow_id}.json"
            
            if not flow_path.exists():
                return False, None, f"Flujo {flow_id} no encontrado"
            
            json_content = flow_path.read_text(encoding="utf-8")
            return FlowSerializer.from_json(json_content)
        except Exception as e:
            return False, None, str(e)
    
    def list_flows(self) -> List[Dict[str, str]]:
        """Lista todos los flujos guardados."""
        flows = []
        for flow_file in self.flows_dir.glob("*.json"):
            success, flow_def, _ = self.load_flow(flow_file.stem)
            if success and flow_def:
                flows.append({
                    "id": flow_def.id,
                    "name": flow_def.name,
                    "created": flow_def.created_at,
                    "updated": flow_def.updated_at
                })
        return flows
    
    def delete_flow(self, flow_id: str) -> Tuple[bool, str]:
        """Elimina un flujo guardado."""
        try:
            flow_path = self.flows_dir / f"{flow_id}.json"
            
            if not flow_path.exists():
                return False, f"Flujo {flow_id} no encontrado"
            
            flow_path.unlink()
            return True, f"Flujo {flow_id} eliminado"
        except Exception as e:
            return False, str(e)
    
    def export_flow(self, flow_id: str, export_path: str) -> Tuple[bool, str]:
        """Exporta un flujo a un archivo."""
        try:
            success, flow_def, msg = self.load_flow(flow_id)
            if not success:
                return False, msg
            
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            json_content = FlowSerializer.to_json(flow_def)
            export_file.write_text(json_content, encoding="utf-8")
            
            return True, f"Flujo exportado a {export_path}"
        except Exception as e:
            return False, str(e)
    
    def import_flow(self, import_path: str) -> Tuple[bool, Optional[FlowDefinition], str]:
        """Importa un flujo desde un archivo."""
        try:
            import_file = Path(import_path)
            
            if not import_file.exists():
                return False, None, f"Archivo {import_path} no encontrado"
            
            json_content = import_file.read_text(encoding="utf-8")
            return FlowSerializer.from_json(json_content)
        except Exception as e:
            return False, None, str(e)
