"""
Tipos específicos de nodos para machine learning.
Incluye Dataset, Model, Training, Test y Output nodes.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional, List
from .base import BaseNode, NodeParam, NodeState
import time


class DatasetNode(BaseNode):
    """Nodo para seleccionar y configurar datasets."""

    NODE_TYPE = "dataset"
    DISPLAY_NAME = "Dataset"
    DESCRIPTION = "Carga y configura un dataset para entrenamiento"

    _IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, node_id: Optional[str] = None, x: float = 0, y: float = 0,
                 dataset_path: Optional[str] = None):
        super().__init__(node_id, x, y)
        self.height = 180
        self.default_dataset_path = dataset_path
        self._resolved_dataset_path: Optional[str] = None

        # Puertos
        self.add_output_port("dataset", "dataset")

        # Parámetros
        self.add_param(NodeParam(
            "dataset_path",
            "str",
            dataset_path or "",
            "Ruta del archivo CSV o carpeta de imágenes"
        ))
        self.add_param(NodeParam(
            "train_split",
            "float",
            0.7,
            "Proporción de datos para entrenamiento",
            min_val=0.1,
            max_val=0.9
        ))
        self.add_param(NodeParam(
            "val_split",
            "float",
            0.15,
            "Proporción de datos para validación",
            min_val=0.0,
            max_val=0.5
        ))
        self.add_param(NodeParam(
            "batch_size",
            "int",
            32,
            "Tamaño del batch",
            min_val=1,
            max_val=256
        ))
        self.add_param(NodeParam(
            "normalize",
            "bool",
            True,
            "Normalizar los datos"
        ))

    def _resolve_dataset_path(self) -> Optional[str]:
        value = self.get_param("dataset_path")
        if value:
            return os.path.abspath(value)
        if self.default_dataset_path:
            return os.path.abspath(self.default_dataset_path)
        return None

    def validate(self) -> tuple[bool, str]:
        """Valida la configuración del dataset."""
        dataset_path = self._resolve_dataset_path()
        if not dataset_path:
            return False, "Dataset path requerido"
        if not os.path.exists(dataset_path):
            return False, f"Ruta no encontrada: {dataset_path}"
        self._resolved_dataset_path = dataset_path
        self.params["dataset_path"].value = dataset_path
        return True, ""

    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Carga el dataset y recopila información real del disco."""
        self.state = NodeState.RUNNING
        start_time = time.time()

        try:
            dataset_path = self._resolved_dataset_path or self._resolve_dataset_path()
            if not dataset_path or not os.path.exists(dataset_path):
                raise FileNotFoundError("Dataset path inválido o inexistente")

            await asyncio.sleep(0.5)

            summary = self._summarize_dataset(dataset_path)
            self.output_data = summary

            self.state = NodeState.SUCCESS
            self.execution_time = time.time() - start_time
            return {"dataset": summary}

        except Exception as e:
            self.state = NodeState.ERROR
            self.error_message = str(e)
            self.execution_time = time.time() - start_time
            return {}

    def _summarize_dataset(self, dataset_path: str) -> Dict[str, Any]:
        path_obj = Path(dataset_path)
        summary: Dict[str, Any] = {
            "dataset_path": dataset_path,
            "train_split": self.get_param("train_split"),
            "val_split": self.get_param("val_split"),
            "batch_size": self.get_param("batch_size"),
            "normalize": self.get_param("normalize"),
        }

        if path_obj.is_dir():
            classes = sorted([p.name for p in path_obj.iterdir() if p.is_dir()])
            samples = 0
            for class_dir in path_obj.iterdir():
                if not class_dir.is_dir():
                    continue
                for file in class_dir.iterdir():
                    if file.is_file() and file.suffix.lower() in self._IMAGE_EXTENSIONS:
                        samples += 1
            summary.update({
                "dataset_type": "image_folder",
                "classes": classes,
                "num_samples": samples,
            })
        elif path_obj.is_file():
            suffix = path_obj.suffix.lower()
            summary.update({
                "dataset_type": suffix or "file",
                "file_size": path_obj.stat().st_size,
            })
            if suffix == ".csv":
                with open(path_obj, "r", encoding="utf-8", errors="ignore") as csvfile:
                    header_line = csvfile.readline().strip()
                    columns = [col.strip() for col in header_line.split(",") if col.strip()]
                    row_count = sum(1 for _ in csvfile)
                summary["columns"] = columns
                summary["num_rows"] = row_count
        else:
            summary["dataset_type"] = "unknown"

        return summary


class ModelNode(BaseNode):
    """Nodo que representa un modelo entrenado o arquitectura."""
    
    NODE_TYPE = "model"
    DISPLAY_NAME = "Model"
    DESCRIPTION = "Define o carga una arquitectura de modelo"
    
    def __init__(self, model_type: str = "cnn", node_id: Optional[str] = None, 
                 x: float = 0, y: float = 0):
        super().__init__(node_id, x, y)
        self.height = 200
        self.model_type = model_type
        
        # Puertos
        self.add_input_port("dataset", "dataset")
        self.add_output_port("model", "model")
        
        # Parámetros comunes
        self.add_param(NodeParam(
            "model_type",
            "choice",
            model_type,
            "Tipo de modelo",
            choices=["cnn", "lstm", "transformer", "xgboost", "sklearn_rf", "custom"]
        ))
        self.add_param(NodeParam(
            "learning_rate",
            "float",
            0.001,
            "Learning rate",
            min_val=1e-5,
            max_val=0.1
        ))
        self.add_param(NodeParam(
            "optimizer",
            "choice",
            "adam",
            "Optimizador",
            choices=["adam", "sgd", "rmsprop"]
        ))
        self.add_param(NodeParam(
            "loss_function",
            "choice",
            "cross_entropy",
            "Función de pérdida",
            choices=["cross_entropy", "mse", "l1"]
        ))
    
    def validate(self) -> tuple[bool, str]:
        if "dataset" not in self.input_ports or not self.input_ports["dataset"].is_connected:
            return False, "Debe conectarse un Dataset"
        return True, ""
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.state = NodeState.RUNNING
        start_time = time.time()
        
        try:
            await asyncio.sleep(0.3)
            
            self.output_data = {
                "model_type": self.get_param("model_type"),
                "learning_rate": self.get_param("learning_rate"),
                "optimizer": self.get_param("optimizer"),
                "loss_function": self.get_param("loss_function"),
                "initialized": True,
            }
            
            self.state = NodeState.SUCCESS
            self.execution_time = time.time() - start_time
            return {"model": self.output_data}
        
        except Exception as e:
            self.state = NodeState.ERROR
            self.error_message = str(e)
            self.execution_time = time.time() - start_time
            return {}


class TrainingNode(BaseNode):
    """Nodo que ejecuta el entrenamiento."""
    
    NODE_TYPE = "training"
    DISPLAY_NAME = "Training"
    DESCRIPTION = "Entrena el modelo con el dataset"
    
    def __init__(self, node_id: Optional[str] = None, x: float = 0, y: float = 0):
        super().__init__(node_id, x, y)
        self.height = 180
        
        # Puertos
        self.add_input_port("dataset", "dataset")
        self.add_input_port("model", "model")
        self.add_output_port("trained_model", "model")
        self.add_output_port("metrics", "metrics")
        
        # Parámetros
        self.add_param(NodeParam(
            "epochs",
            "int",
            10,
            "Número de épocas",
            min_val=1,
            max_val=500
        ))
        self.add_param(NodeParam(
            "early_stopping",
            "bool",
            True,
            "Usar early stopping"
        ))
        self.add_param(NodeParam(
            "patience",
            "int",
            5,
            "Paciencia para early stopping",
            min_val=1,
            max_val=50
        ))
    
    def validate(self) -> tuple[bool, str]:
        if not self.input_ports["dataset"].is_connected:
            return False, "Debe conectarse un Dataset"
        if not self.input_ports["model"].is_connected:
            return False, "Debe conectarse un Model"
        return True, ""
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.state = NodeState.RUNNING
        start_time = time.time()
        
        try:
            epochs = self.get_param("epochs")
            
            # Simulación de entrenamiento
            for epoch in range(epochs):
                await asyncio.sleep(0.1)  # Simulación
            
            self.output_data = {
                "epochs_trained": epochs,
                "final_loss": 0.23,
                "final_accuracy": 0.92,
                "training_time": time.time() - start_time,
                "model_saved": True,
            }
            
            self.state = NodeState.SUCCESS
            self.execution_time = time.time() - start_time
            return {
                "trained_model": self.output_data,
                "metrics": {
                    "loss": 0.23,
                    "accuracy": 0.92,
                }
            }
        
        except Exception as e:
            self.state = NodeState.ERROR
            self.error_message = str(e)
            self.execution_time = time.time() - start_time
            return {}


class TestNode(BaseNode):
    """Nodo para evaluar el modelo en datos de prueba."""
    
    NODE_TYPE = "test"
    DISPLAY_NAME = "Test/Evaluation"
    DESCRIPTION = "Evalúa el modelo en datos de prueba"
    
    def __init__(self, node_id: Optional[str] = None, x: float = 0, y: float = 0):
        super().__init__(node_id, x, y)
        self.height = 160
        
        # Puertos
        self.add_input_port("dataset", "dataset")
        self.add_input_port("model", "model")
        self.add_output_port("results", "metrics")
        
        # Parámetros
        self.add_param(NodeParam(
            "metric",
            "choice",
            "accuracy",
            "Métrica principal",
            choices=["accuracy", "f1", "roc_auc", "precision", "recall"]
        ))
    
    def validate(self) -> tuple[bool, str]:
        if not self.input_ports["dataset"].is_connected:
            return False, "Debe conectarse un Dataset"
        if not self.input_ports["model"].is_connected:
            return False, "Debe conectarse un Model"
        return True, ""
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.state = NodeState.RUNNING
        start_time = time.time()
        
        try:
            await asyncio.sleep(0.2)
            
            metric = self.get_param("metric")
            
            self.output_data = {
                "accuracy": 0.89,
                "f1": 0.87,
                "precision": 0.91,
                "recall": 0.85,
                "roc_auc": 0.93,
                "primary_metric": metric,
                "primary_metric_value": 0.89,
            }
            
            self.state = NodeState.SUCCESS
            self.execution_time = time.time() - start_time
            return {"results": self.output_data}
        
        except Exception as e:
            self.state = NodeState.ERROR
            self.error_message = str(e)
            self.execution_time = time.time() - start_time
            return {}


class OutputNode(BaseNode):
    """Nodo final que muestra resultados."""
    
    NODE_TYPE = "output"
    DISPLAY_NAME = "Output/Results"
    DESCRIPTION = "Muestra los resultados finales"
    
    def __init__(self, node_id: Optional[str] = None, x: float = 0, y: float = 0):
        super().__init__(node_id, x, y)
        self.height = 160
        
        # Puertos
        self.add_input_port("metrics", "metrics")
        self.add_input_port("model", "model")
    
    def validate(self) -> tuple[bool, str]:
        if not self.input_ports["metrics"].is_connected:
            return False, "Debe conectarse métricas"
        return True, ""
    
    async def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.state = NodeState.RUNNING
        
        try:
            self.output_data = inputs.get("metrics", {})
            self.state = NodeState.SUCCESS
            return {"completed": True}
        
        except Exception as e:
            self.state = NodeState.ERROR
            self.error_message = str(e)
            return {}


class NodeFactory:
    """Factory para crear instancias de nodos."""
    
    _NODE_REGISTRY: Dict[str, type] = {
        "dataset": DatasetNode,
        "model": ModelNode,
        "training": TrainingNode,
        "test": TestNode,
        "output": OutputNode,
    }
    
    @classmethod
    def register_node(cls, node_type: str, node_class: type):
        """Registra un nuevo tipo de nodo."""
        cls._NODE_REGISTRY[node_type] = node_class
    
    @classmethod
    def create_node(cls, node_type: str, **kwargs) -> Optional[BaseNode]:
        """Crea una instancia de nodo del tipo especificado."""
        node_class = cls._NODE_REGISTRY.get(node_type)
        if node_class:
            init_kwargs = dict(kwargs)
            dataset_path = init_kwargs.pop("dataset_path", None)
            if node_type == "dataset" and dataset_path is not None:
                init_kwargs["dataset_path"] = dataset_path
            return node_class(**init_kwargs)
        return None
    
    @classmethod
    def get_available_nodes(cls) -> List[str]:
        """Retorna lista de tipos de nodos disponibles."""
        return list(cls._NODE_REGISTRY.keys())
    
    @classmethod
    def get_node_info(cls, node_type: str) -> Dict[str, str]:
        """Retorna información sobre un tipo de nodo."""
        node_class = cls._NODE_REGISTRY.get(node_type)
        if node_class:
            return {
                "type": node_type,
                "display_name": node_class.DISPLAY_NAME,
                "description": node_class.DESCRIPTION,
            }
        return {}
