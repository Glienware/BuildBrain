"""
Serialización y deserialización de flujos en formato JSON (.buildb).
Permite guardar y cargar configuraciones completas de nodos.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
from .base import BaseNode, NodeCanvas, Connection, NodePort
from .types import NodeFactory


class FlowSerializer:
    """Serializa un flujo a JSON (.buildb)."""
    
    @staticmethod
    def serialize_canvas(canvas: NodeCanvas) -> Dict[str, Any]:
        """Convierte un canvas a diccionario serializable."""
        return {
            "version": "1.0",
            "metadata": {
                "width": canvas.width,
                "height": canvas.height,
                "zoom": canvas.zoom,
                "pan": {"x": canvas.pan_x, "y": canvas.pan_y},
            },
            "nodes": FlowSerializer._serialize_nodes(canvas),
            "connections": FlowSerializer._serialize_connections(canvas),
        }
    
    @staticmethod
    def _serialize_nodes(canvas: NodeCanvas) -> list:
        """Serializa todos los nodos."""
        nodes = []
        for node in canvas.nodes.values():
            nodes.append({
                "node_id": node.node_id,
                "node_type": node.NODE_TYPE,
                "x": node.x,
                "y": node.y,
                "width": node.width,
                "height": node.height,
                "params": {
                    name: param.to_dict()
                    for name, param in node.params.items()
                },
                "state": node.state.value,
                "output_data": node.output_data,
            })
        return nodes
    
    @staticmethod
    def _serialize_connections(canvas: NodeCanvas) -> list:
        """Serializa todas las conexiones."""
        connections = []
        for conn in canvas.connections:
            connections.append({
                "connection_id": conn.connection_id,
                "source_node_id": conn.source.node_id,
                "source_port": conn.source.port_name,
                "target_node_id": conn.target.node_id,
                "target_port": conn.target.port_name,
            })
        return connections
    
    @staticmethod
    def save_to_file(canvas: NodeCanvas, filepath: str) -> tuple[bool, str]:
        """
        Guarda el flujo a un archivo .buildb.
        Retorna (éxito, mensaje).
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            data = FlowSerializer.serialize_canvas(canvas)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return True, f"Flujo guardado en {filepath}"
        
        except Exception as e:
            return False, f"Error guardando flujo: {str(e)}"


class FlowLoader:
    """Carga un flujo desde JSON (.buildb)."""
    
    @staticmethod
    def deserialize_canvas(data: Dict[str, Any]) -> Optional[NodeCanvas]:
        """Convierte un diccionario a un canvas."""
        try:
            # Crear canvas
            metadata = data.get("metadata", {})
            canvas = NodeCanvas(
                width=metadata.get("width", 1200),
                height=metadata.get("height", 800)
            )
            canvas.zoom = metadata.get("zoom", 1.0)
            pan = metadata.get("pan", {})
            canvas.pan_x = pan.get("x", 0)
            canvas.pan_y = pan.get("y", 0)
            
            # Cargar nodos
            nodes_data = data.get("nodes", [])
            node_map: Dict[str, BaseNode] = {}
            
            for node_data in nodes_data:
                node = FlowLoader._deserialize_node(node_data)
                if node:
                    canvas.add_node(node)
                    node_map[node.node_id] = node
            
            # Cargar conexiones
            connections_data = data.get("connections", [])
            for conn_data in connections_data:
                FlowLoader._deserialize_connection(canvas, conn_data, node_map)
            
            return canvas
        
        except Exception as e:
            print(f"Error deserializando canvas: {str(e)}")
            return None
    
    @staticmethod
    def _deserialize_node(node_data: Dict[str, Any]) -> Optional[BaseNode]:
        """Deserializa un nodo individual."""
        try:
            node_type = node_data.get("node_type")
            node_id = node_data.get("node_id")
            x = node_data.get("x", 0)
            y = node_data.get("y", 0)
            
            # Crear nodo usando factory
            node = NodeFactory.create_node(node_type, node_id=node_id, x=x, y=y)
            
            if not node:
                return None
            
            # Restaurar parámetros
            params_data = node_data.get("params", {})
            for param_name, param_data in params_data.items():
                if param_name in node.params:
                    node.set_param(param_name, param_data.get("value"))
            
            return node
        
        except Exception as e:
            print(f"Error deserializando nodo: {str(e)}")
            return None
    
    @staticmethod
    def _deserialize_connection(canvas: NodeCanvas, conn_data: Dict[str, Any],
                               node_map: Dict[str, BaseNode]) -> bool:
        """Deserializa una conexión."""
        try:
            source_node_id = conn_data.get("source_node_id")
            source_port = conn_data.get("source_port")
            target_node_id = conn_data.get("target_node_id")
            target_port = conn_data.get("target_port")
            
            source_node = node_map.get(source_node_id)
            target_node = node_map.get(target_node_id)
            
            if not source_node or not target_node:
                return False
            
            if source_port not in source_node.output_ports:
                return False
            
            if target_port not in target_node.input_ports:
                return False
            
            # Crear conexión
            return canvas.connect(
                source_node.output_ports[source_port],
                target_node.input_ports[target_port]
            )
        
        except Exception as e:
            print(f"Error deserializando conexión: {str(e)}")
            return False
    
    @staticmethod
    def load_from_file(filepath: str) -> tuple[bool, Optional[NodeCanvas], str]:
        """
        Carga un flujo desde un archivo .buildb.
        Retorna (éxito, canvas, mensaje).
        """
        try:
            path = Path(filepath)
            
            if not path.exists():
                return False, None, f"Archivo no encontrado: {filepath}"
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            canvas = FlowLoader.deserialize_canvas(data)
            
            if not canvas:
                return False, None, "Error deserializando canvas"
            
            return True, canvas, f"Flujo cargado desde {filepath}"
        
        except json.JSONDecodeError as e:
            return False, None, f"Error parseando JSON: {str(e)}"
        except Exception as e:
            return False, None, f"Error cargando flujo: {str(e)}"


class BuildBrainFileManager:
    """Gestor de archivos .buildb de BuildBrain."""
    
    FILE_EXTENSION = ".buildb"
    
    @staticmethod
    def save_project(canvas: NodeCanvas, project_name: str, project_dir: str) -> tuple[bool, str]:
        """
        Guarda un proyecto como .buildb.
        Retorna (éxito, mensaje).
        """
        filepath = Path(project_dir) / f"{project_name}{BuildBrainFileManager.FILE_EXTENSION}"
        return FlowSerializer.save_to_file(canvas, str(filepath))
    
    @staticmethod
    def load_project(project_name: str, project_dir: str) -> tuple[bool, Optional[NodeCanvas], str]:
        """
        Carga un proyecto .buildb.
        Retorna (éxito, canvas, mensaje).
        """
        filepath = Path(project_dir) / f"{project_name}{BuildBrainFileManager.FILE_EXTENSION}"
        return FlowLoader.load_from_file(str(filepath))
    
    @staticmethod
    def get_projects_in_directory(project_dir: str) -> list[str]:
        """Lista todos los proyectos .buildb en un directorio."""
        try:
            path = Path(project_dir)
            if not path.exists():
                return []
            
            return [
                f.stem
                for f in path.glob(f"*{BuildBrainFileManager.FILE_EXTENSION}")
            ]
        except Exception:
            return []
