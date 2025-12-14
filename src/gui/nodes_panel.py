"""
Interfaz visual de nodos en Flet.
Implementa un canvas interactivo para diseñar flujos visuales con drag & drop.
"""

import flet as ft
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass
from ..nodes.base import BaseNode, NodeCanvas, Connection, NodePort, NodeState
from ..nodes.types import NodeFactory
from ..nodes.executor import TopologicalExecutor
from ..nodes.serializer import FlowSerializer, FlowLoader


@dataclass
class Vector2:
    """Representa una posición 2D."""
    x: float
    y: float
    
    def __add__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: "Vector2") -> "Vector2":
        return Vector2(self.x - other.x, self.y - other.y)
    
    def distance_to(self, other: "Vector2") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class NodeVisual(ft.Container):
    """Representación visual de un nodo."""
    
    def __init__(self, node: BaseNode, on_select: Callable, on_pan: Callable, on_port_click: Optional[Callable[[str, str], None]] = None):
        self.node = node
        self.on_select = on_select
        self.on_port_click = on_port_click
        self.is_selected = False
        
        super().__init__(
            width=160,
            height=100,
            border_radius=ft.border_radius.all(8),
            padding=0,
            shadow=ft.BoxShadow(blur_radius=8, color="rgba(0,0,0,0.3)"),
        )
        
        # Click para seleccionar
        self.on_tap = self._on_tap
        
        self._update_content()
        self._update_appearance()
    
    def _update_content(self):
        """Actualiza el contenido visual del nodo."""
        icon_map = {
            "dataset": ft.Icons.STORAGE,
            "model": ft.Icons.BUILD,
            "training": ft.Icons.PLAY_CIRCLE,
            "test": ft.Icons.CHECK_CIRCLE,
            "output": ft.Icons.SAVE,
        }
        
        icon = icon_map.get(self.node.NODE_TYPE, ft.Icons.SETTINGS)
        
        port_size = 12
        input_controls = self._build_port_controls(self.node.input_ports, "input", port_size)
        output_controls = self._build_port_controls(self.node.output_ports, "output", port_size)

        content_body = ft.Column(
            spacing=4,
            controls=[
                ft.Row(
                    spacing=5,
                    controls=[
                        ft.Icon(icon, size=16),
                        ft.Text(self.node.DISPLAY_NAME, size=11, weight="bold", expand=True),
                    ]
                ),
                ft.Text(self.node.NODE_TYPE, size=8, color="rgba(255,255,255,0.6)"),
            ]
        )

        self.content = ft.Row(
            spacing=0,
            controls=[
                ft.Container(
                    width=20,
                    height=100,
                    content=ft.Column(
                        spacing=8,
                        controls=input_controls,
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    alignment=ft.alignment.top_center,
                ),
                ft.Container(
                    expand=True,
                    padding=ft.padding.symmetric(8, 6),
                    content=content_body,
                ),
                ft.Container(
                    width=20,
                    height=100,
                    content=ft.Column(
                        spacing=8,
                        controls=output_controls,
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    alignment=ft.alignment.top_center,
                ),
            ]
        )

    def _build_port_controls(self, ports: Dict[str, NodePort], port_type: str, size: int) -> List[ft.Control]:
        controls: List[ft.Control] = []
        if not ports:
            controls.append(ft.Container(width=size, height=size))
            return controls

        color = "#4CAF50" if port_type == "input" else "#FF9800"

        for port_name in ports:
            controls.append(ft.Container(
                width=size,
                height=size,
                border_radius=ft.border_radius.all(size / 2),
                bgcolor=color,
                border=ft.border.all(2, "#fff"),
                tooltip=port_name,
                on_click=lambda e, pt=port_type, pn=port_name: self._on_port_click(pt, pn),
            ))
        return controls
    
    def _update_appearance(self):
        """Actualiza colores según estado."""
        color_map = {
            NodeState.IDLE: "#3a3a5c",
            NodeState.RUNNING: "#1e3a8a",
            NodeState.SUCCESS: "#1b5e20",
            NodeState.ERROR: "#b71c1c",
            NodeState.PENDING: "#e65100",
        }
        
        border_color_map = {
            NodeState.IDLE: "#6366f1",
            NodeState.RUNNING: "#2196f3",
            NodeState.SUCCESS: "#66bb6a",
            NodeState.ERROR: "#ef5350",
            NodeState.PENDING: "#ffa726",
        }
        
        bg_color = color_map.get(self.node.state, "#3a3a5c")
        border_color = border_color_map.get(self.node.state, "#6366f1")
        
        if self.is_selected:
            border_color = "#ffd700"
            border_width = 3
        else:
            border_width = 2
        
        self.bgcolor = bg_color
        self.border = ft.border.all(border_width, border_color)
    
    def _on_tap(self, e):
        """Clic para seleccionar."""
        self.on_select(self.node.node_id)
    
    def _on_port_click(self, port_type: str, port_name: str):
        if self.on_port_click:
            self.on_port_click(self.node.node_id, port_type, port_name)

    def set_selected(self, selected: bool):
        """Marca el nodo como seleccionado."""
        self.is_selected = selected
        self._update_appearance()
        if self.page:
            self.update()


class CanvasView(ft.GestureDetector):
    """Canvas con soporte de drag para nodos."""
    
    def __init__(self):
        self.canvas = None
        self.node_visuals: Dict[str, NodeVisual] = {}
        self.selected_node_id: Optional[str] = None
        self.dragging_node_id: Optional[str] = None
        self.drag_offset = Vector2(0, 0)
        
        # Para conexiones
        self.connecting_from: Optional[Tuple[str, str]] = None
        self.connection_lines: Dict[str, List[ft.Container]] = {}

        # Layers para conexiones y nodos
        self.connections_layer = ft.Stack(expand=True)
        self.nodes_layer = ft.Stack(expand=True)
        self.stack = ft.Stack(expand=True, controls=[self.connections_layer, self.nodes_layer])
        
        # Container base
        container = ft.Container(
            expand=True,
            bgcolor="#0f1419",
            border=ft.border.all(1, "#333"),
            content=self.stack,
        )
        
        super().__init__(
            content=container,
            expand=True,
        )
        
        # Pan events en el canvas
        self.on_pan_start = self._on_pan_start
        self.on_pan_update = self._on_pan_update
        self.on_pan_end = self._on_pan_end
    
    def set_canvas(self, canvas: NodeCanvas):
        """Establece el canvas para renderizar."""
        self.canvas = canvas
        if self.page:
            self.refresh_view()
    
    def refresh_view(self):
        """Actualiza la visualización del canvas."""
        if not self.canvas:
            return
        
        controls = []
        
        # Renderizar nodos
        for node_id, node in self.canvas.nodes.items():
            if node_id not in self.node_visuals:
                visual = NodeVisual(
                    node,
                    on_select=self._on_node_select,
                    on_pan=lambda nid, x, y: None,
                    on_port_click=self._on_port_click,
                )
                self.node_visuals[node_id] = visual
            
            visual = self.node_visuals[node_id]
            visual._update_appearance()
            visual.left = node.x
            visual.top = node.y
            controls.append(visual)
        
        self.nodes_layer.controls = controls
        self._draw_connections()
        self.update()
    
    def _on_node_select(self, node_id: str):
        """Maneja la selección de un nodo."""
        if self.selected_node_id and self.selected_node_id in self.node_visuals:
            self.node_visuals[self.selected_node_id].set_selected(False)
        
        self.selected_node_id = node_id
        if node_id in self.node_visuals:
            self.node_visuals[node_id].set_selected(True)
    
    def _on_pan_start(self, e: ft.DragStartEvent):
        """Inicia el pan en el canvas."""
        # Detectar qué nodo se está arrastando
        for node_id, visual in self.node_visuals.items():
            node_x = visual.left or 0
            node_y = visual.top or 0
            
            # ¿El click está dentro del nodo?
            if (node_x <= e.local_x <= node_x + 160 and
                node_y <= e.local_y <= node_y + 100):
                self.dragging_node_id = node_id
                self.drag_offset = Vector2(
                    e.local_x - node_x,
                    e.local_y - node_y,
                )
                self._on_node_select(node_id)
                break
    
    def _on_pan_update(self, e: ft.DragUpdateEvent):
        """Actualiza posición durante pan."""
        if self.dragging_node_id and self.dragging_node_id in self.node_visuals:
            visual = self.node_visuals[self.dragging_node_id]
            
            # Nueva posición
            new_x = max(0, e.local_x - self.drag_offset.x)
            new_y = max(0, e.local_y - self.drag_offset.y)
            
            visual.left = new_x
            visual.top = new_y
            visual.node.x = new_x
            visual.node.y = new_y
            
            if self.page:
                self._draw_connections()
                self.update()
    
    def _on_pan_end(self, e: ft.DragEndEvent):
        """Termina el pan."""
        self.dragging_node_id = None
    
    def add_visual(self, node: BaseNode):
        """Agrega un nodo visual al canvas."""
        if node.node_id not in self.node_visuals:
            visual = NodeVisual(
                node,
                on_select=self._on_node_select,
                on_pan=lambda nid, x, y: None,
                on_port_click=self._on_port_click,
            )
            self.node_visuals[node.node_id] = visual
            visual.left = node.x
            visual.top = node.y
            self.nodes_layer.controls.append(visual)
            self._draw_connections()
            self.update()
    
    def remove_visual(self, node_id: str):
        """Remueve un nodo visual del canvas."""
        if node_id in self.node_visuals:
            visual = self.node_visuals[node_id]
            self.nodes_layer.controls.remove(visual)
            del self.node_visuals[node_id]
            self.update()

    def _draw_connections(self):
        """Dibuja las conexiones con segmentos simples."""
        if not self.canvas:
            return

        self.connections_layer.controls = []
        self.connection_lines.clear()

        for connection in self.canvas.connections:
            source = connection.source
            target = connection.target
            if source.node_id not in self.node_visuals or target.node_id not in self.node_visuals:
                continue

            from_visual = self.node_visuals[source.node_id]
            to_visual = self.node_visuals[target.node_id]

            start_x = (from_visual.left or 0) + 160
            start_y = (from_visual.top or 0) + 50
            end_x = (to_visual.left or 0)
            end_y = (to_visual.top or 0) + 50

            segments = self._create_l_shaped_line(start_x, start_y, end_x, end_y)
            if segments:
                dot = ft.Container(
                    left=end_x - 4,
                    top=end_y - 4,
                    width=8,
                    height=8,
                    bgcolor="#FF9800",
                    border=ft.border.all(1, "#fff"),
                    border_radius=ft.border_radius.all(4),
                )
                segments.append(dot)
            for segment in segments:
                self.connections_layer.controls.append(segment)
            if segments:
                self.connection_lines[connection.connection_id] = segments

    def _create_l_shaped_line(self, x1: float, y1: float, x2: float, y2: float) -> List[ft.Container]:
        """Construye una conexión en L con segmentos horizontales y verticales."""
        segments: List[ft.Container] = []
        mid_x = x1 + (x2 - x1) / 2
        color = "#FF9800"

        def horizontal_segment(start, end, y):
            width = abs(end - start)
            if width < 1:
                return None
            left = min(start, end)
            return ft.Container(
                left=left,
                top=y - 1,
                width=width,
                height=2,
                bgcolor=color,
                border_radius=ft.border_radius.all(1),
            )

        def vertical_segment(x, start, end):
            height = abs(end - start)
            if height < 1:
                return None
            top = min(start, end)
            return ft.Container(
                left=x - 1,
                top=top,
                width=2,
                height=height,
                bgcolor=color,
                border_radius=ft.border_radius.all(1),
            )

        first = horizontal_segment(x1, mid_x, y1)
        second = vertical_segment(mid_x, y1, y2)
        third = horizontal_segment(mid_x, x2, y2)

        for segment in (first, second, third):
            if segment:
                segments.append(segment)

        return segments

    def _on_port_click(self, node_id: str, port_type: str, port_name: str):
        """Maneja conexiones en orden naranja (salida) → verde (entrada)."""
        if port_type == "output":
            self.connecting_from = (node_id, port_name)
            print(f"📍 Salida {node_id[:6]}:{port_name} seleccionada")
            print("👉 Ahora toca hacer clic en una entrada verde")
            return

        if port_type != "input":
            print("⚠️ Solo se conectan salidas a entradas")
            return

        if not self.connecting_from:
            print("⚠️ Selecciona primero una salida naranja")
            return

        if self.connecting_from[0] == node_id:
            print("❌ Selecciona otra entrada distinta al mismo nodo")
            self.connecting_from = None
            return

        source_node = self.canvas.nodes.get(self.connecting_from[0])
        target_node = self.canvas.nodes.get(node_id)
        if not source_node or not target_node:
            self.connecting_from = None
            return

        source_port = source_node.output_ports.get(self.connecting_from[1])
        target_port = target_node.input_ports.get(port_name)
        if source_port and target_port and self.canvas.connect(source_port, target_port):
            print(f"✓ Conectado {self.connecting_from[:6]} → {node_id[:6]}")
            self._draw_connections()
            if self.page:
                self.update()
        else:
            reason = self.canvas.get_last_connection_error() if self.canvas else ""
            print(f"❌ {reason or 'No se pudo crear la conexión'}")

        self.connecting_from = None


class NodesPanel(ft.Column):
    """Panel principal para la sección de nodos con mejor diseño."""
    
    def __init__(self):
        super().__init__(expand=True, spacing=0)
        
        self.canvas = NodeCanvas(width=1200, height=800)
        self.executor = TopologicalExecutor(self.canvas)
        
        self.canvas_view = CanvasView()
        
        # Panel lateral izquierdo con tipos de nodos
        self.node_types_panel = self._create_node_types_panel()
        
        # Panel inferior con información
        self.info_panel = self._create_info_panel()
        
        # Toolbar superior
        self.toolbar = self._create_toolbar()
        
        # Construir la interfaz
        self._build_ui()
        
        self.canvas_view.set_canvas(self.canvas)

    def did_mount(self):
        super().did_mount()
        self.canvas_view.set_canvas(self.canvas)
    
    def _create_node_types_panel(self) -> ft.Container:
        """Crea el panel lateral con tipos de nodos organizados por categoría."""
        
        # Mapeo de categorías de nodos
        categories = {
            "Entrada": ["dataset"],
            "Modelo": ["model"],
            "Entrenamiento": ["training"],
            "Evaluación": ["test"],
            "Salida": ["output"],
        }
        
        node_buttons = []
        
        for category, node_types in categories.items():
            node_buttons.append(
                ft.Text(category, weight="bold", size=12, color="#6366f1")
            )
            
            for node_type in node_types:
                info = NodeFactory.get_node_info(node_type)
                
                # Botón simple que agrega nodos
                btn = ft.ElevatedButton(
                    text=info.get("display_name", node_type),
                    on_click=lambda e, nt=node_type: self._add_node_simple(nt),
                    width=170,
                    style=ft.ButtonStyle(
                        bgcolor=ft.Colors.INDIGO_900,
                        color=ft.Colors.WHITE,
                    ),
                )
                node_buttons.append(btn)
            
            node_buttons.append(ft.Divider(height=1, color="rgba(255,255,255,0.1)"))
        
        return ft.Container(
            content=ft.ListView(
                spacing=8,
                padding=12,
                controls=node_buttons,
            ),
            width=200,
            bgcolor="#1a1e28",
            border=ft.border.all(1, "#333"),
        )
    
    def _create_toolbar(self) -> ft.Container:
        """Crea la barra de herramientas superior mejorada."""
        self.status_text = ft.Text("Listo para diseñar", size=12, color="#66bb6a")
        
        return ft.Container(
            content=ft.Row(
                spacing=15,
                controls=[
                    ft.IconButton(
                        ft.Icons.PLAY_ARROW,
                        icon_size=24,
                        icon_color="#66bb6a",
                        on_click=self._execute_flow,
                        tooltip="Ejecutar flujo",
                    ),
                    ft.IconButton(
                        ft.Icons.SAVE,
                        icon_size=24,
                        icon_color="#42a5f5",
                        on_click=self._save_flow,
                        tooltip="Guardar flujo",
                    ),
                    ft.IconButton(
                        ft.Icons.FOLDER_OPEN,
                        icon_size=24,
                        icon_color="#ab47bc",
                        on_click=self._load_flow,
                        tooltip="Cargar flujo",
                    ),
                    ft.IconButton(
                        ft.Icons.DELETE,
                        icon_size=24,
                        icon_color="#ef5350",
                        on_click=self._clear_canvas,
                        tooltip="Limpiar canvas",
                    ),
                    ft.VerticalDivider(width=1),
                    self.status_text,
                    ft.Container(expand=True),
                ]
            ),
            padding=12,
            bgcolor="#1a1e28",
            border=ft.border.all(1, "#333"),
        )
    
    def _create_info_panel(self) -> ft.Container:
        """Crea el panel de información inferior mejorado."""
        self.stats_text = ft.Text(
            "Nodos: 0 | Conexiones: 0 | Listo",
            size=10,
            color="#a0aec0",
        )
        self.execution_log_text = ft.Text("", size=9, color="#60a5fa", max_lines=3)
        
        return ft.Container(
            content=ft.Column(
                spacing=8,
                controls=[
                    self.stats_text,
                    self.execution_log_text,
                ]
            ),
            padding=12,
            height=100,
            bgcolor="#1a1e28",
            border=ft.border.all(1, "#333"),
            expand=False,
        )
    
    def _on_node_added(self, node_type: str, x: float, y: float):
        """Se llama cuando un nodo es arrastrado al canvas."""
        node = NodeFactory.create_node(node_type, x=x, y=y)
        if node:
            self.canvas.add_node(node)
            self.canvas_view.add_visual(node)
            self._update_stats()
            self.status_text.value = f"✓ {node.DISPLAY_NAME} agregado"
            self.status_text.color = "#66bb6a"
            self.status_text.update()
    
    def _add_node_simple(self, node_type: str):
        """Agrega un nodo con posición en área visible."""
        # Agregar nodos en cascada desde la esquina superior izquierda
        num_nodes = len(self.canvas.nodes)
        x = 50 + (num_nodes * 30)  # Offset horizontal por número de nodos
        y = 50 + (num_nodes * 30)  # Offset vertical por número de nodos
        self._on_node_added(node_type, x, y)
    
    def _update_stats(self):
        """Actualiza las estadísticas mostradas."""
        self.stats_text.value = (
            f"Nodos: {len(self.canvas.nodes)} | "
            f"Conexiones: {len(self.canvas.connections)} | Listo"
        )
        self.stats_text.update()
    
    async def _execute_flow(self, e):
        """Ejecuta el flujo de nodos."""
        if not self.canvas.nodes:
            self.status_text.value = "⚠ Agrega nodos antes de ejecutar"
            self.status_text.color = "#ffa726"
            self.status_text.update()
            return
        
        self.status_text.value = "⏳ Ejecutando flujo..."
        self.status_text.color = "#42a5f5"
        self.status_text.update()
        
        success, msg = await self.executor.execute()
        
        self.canvas_view.refresh_view()
        
        if success:
            self.status_text.value = "✓ Flujo ejecutado exitosamente"
            self.status_text.color = "#66bb6a"
        else:
            self.status_text.value = f"✗ Error: {msg}"
            self.status_text.color = "#ef5350"
        
        self.status_text.update()
    
    def _save_flow(self, e):
        """Guarda el flujo actual."""
        if not self.canvas.nodes:
            self.status_text.value = "⚠ No hay nodos para guardar"
            self.status_text.color = "#ffa726"
            self.status_text.update()
            return
        
        success, msg = FlowSerializer.save_to_file(
            self.canvas,
            "flows/default_flow.buildb"
        )
        
        if success:
            self.status_text.value = "✓ Flujo guardado"
            self.status_text.color = "#66bb6a"
        else:
            self.status_text.value = f"✗ Error: {msg}"
            self.status_text.color = "#ef5350"
        
        self.status_text.update()
    
    def _load_flow(self, e):
        """Carga un flujo guardado."""
        success, canvas, msg = FlowLoader.load_from_file("flows/default_flow.buildb")
        
        if success and canvas:
            self.canvas = canvas
            self.executor.canvas = canvas
            self.canvas_view.set_canvas(canvas)
            self.status_text.value = "✓ Flujo cargado"
            self.status_text.color = "#66bb6a"
        else:
            self.status_text.value = f"✗ {msg}"
            self.status_text.color = "#ef5350"
        
        self._update_stats()
        self.status_text.update()
    
    def _clear_canvas(self, e):
        """Limpia el canvas."""
        self.canvas = NodeCanvas(width=1200, height=800)
        self.executor = TopologicalExecutor(self.canvas)
        self.canvas_view.set_canvas(self.canvas)
        self.status_text.value = "✓ Canvas limpiado"
        self.status_text.color = "#66bb6a"
        self._update_stats()
        self.status_text.update()
    
    def _build_ui(self):
        """Construye la interfaz del panel de nodos."""
        self.controls = [
            self.toolbar,
            ft.Row(
                spacing=0,
                expand=True,
                controls=[
                    self.node_types_panel,
                    self.canvas_view,
                ]
            ),
            self.info_panel,
        ]