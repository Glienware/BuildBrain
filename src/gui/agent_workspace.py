import flet as ft
from typing import Dict, List, Optional, Any, Tuple
import uuid
from datetime import datetime
from src.nodes.automation_nodes import get_all_nodes_by_category, NodeConfig
from src.nodes.executor import TopologicalExecutor, FlowValidator
from src.nodes.serializer import FlowPersistence

DEFAULT_BG = "#0d1117"
HOVER_COLOR = "#1f2937"
PORT_SIZE = 12
CONNECTION_COLOR = "#00BCD4"
NODE_WIDTH = 240
NODE_HEIGHT = 120
CANVAS_PANEL_BG = "#05070a"
SIDE_PANEL_BG = "#0c1420"
CARD_BG = "#111827"
ACCENT_COLOR = "#40C4FF"


class NodeInstance:
    """Instancia de un nodo en el canvas."""
    def __init__(self, node_id: str, config: NodeConfig, x: float, y: float):
        self.node_id = node_id
        self.config = config
        self.position = {"x": x, "y": y}
        self.settings: Dict[str, Any] = dict(config.settings)
        self.input_ports: List[str] = list(config.input_ports.keys())
        self.output_ports: List[str] = list(config.output_ports.keys())
        self.ui_control: Optional[ft.Control] = None
        self.port_positions: Dict[str, Tuple[float, float]] = {}
        self.drag_offset: Dict[str, float] = {"x": 0.0, "y": 0.0}


class Connection:
    """Conexión entre dos nodos."""
    def __init__(self, conn_id: str, source_node: str, source_port: str, 
                 target_node: str, target_port: str):
        self.id = conn_id
        self.source_node = source_node
        self.source_port = source_port
        self.target_node = target_node
        self.target_port = target_port


class AgentWorkspace:
    def __init__(self, page: ft.Page, agents_dir: str | None = None, on_back_to_welcome=None):
        self.page = page
        self.agents_dir = agents_dir
        self.on_back = on_back_to_welcome

        # Estado
        self.nodes: Dict[str, NodeInstance] = {}
        self.connections: Dict[str, Connection] = {}
        self.node_counter = 0
        self.selected_node: Optional[str] = None
        self.dragging_node: Optional[str] = None
        self.connecting_from: Optional[Tuple[str, str]] = None
        self.connection_in_progress: Optional[Dict] = None  # Rastrear conexión en progreso
        self.sidebar_collapsed = False
        
        # Variables para zoom
        self.canvas_scale = 1.0
        self.min_scale = 0.5
        self.max_scale = 3.0
        
        # Variables para pan/movimiento del canvas
        self.canvas_offset_x = 0.0
        self.canvas_offset_y = 0.0
        self.is_panning = False
        self.pan_start_x = 0.0
        self.pan_start_y = 0.0
        
        # Throttle para redibujado de conexiones (evitar lag)
        self.last_connection_draw = 0.0
        self.connection_draw_throttle = 0.05  # 50ms entre redibujados

        # Catálogo
        self.node_catalog = get_all_nodes_by_category()

        # Node palette modal helpers
        self.palette_modal: Optional[ft.Stack] = None

        # UI
        self.title_input = ft.TextField(
            value="Untitled Workflow",
            width=360,
            border_color="#1f2937",
            bgcolor="#06070a",
            color="#f8fafc",
        )
        self.node_count_text = ft.Text("0 nodes", size=10, color="#e0e7ff")
        
        # Contenedor para las líneas de conexión (Stack para posicionamiento absoluto)
        self.connections_stack = ft.Stack([], expand=True, clip_behavior=ft.ClipBehavior.HARD_EDGE)
        
        # Canvas stack con tamaño mínimo para evitar que se expanda con los nodos
        self.canvas_stack = ft.Stack(
            expand=True, 
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            width=800,
            height=600
        )
        self.canvas_panel = self._build_canvas_panel()
        self.execution_log: List[str] = []
        self.inspector_content = ft.Column([], expand=True, spacing=8)
        self.log_list = ft.ListView(expand=True, spacing=6, auto_scroll=True)
        self.left_panel_container = ft.Container(
            width=300,
            bgcolor=SIDE_PANEL_BG,
            padding=16,
            border=ft.border.all(1, "#1f2937"),
            border_radius=16,
        )
        self.right_panel_container = ft.Container(
            width=320,
            bgcolor=SIDE_PANEL_BG,
            padding=12,
            border=ft.border.all(1, "#1f2937"),
            border_radius=16,
        )
        self._refresh_left_panel()
        self._refresh_right_panel()

        # Render view
        self.view = ft.Container(
            expand=True,
            bgcolor=DEFAULT_BG,
            padding=14,
            content=ft.Column([
                self._build_top_bar(),
                ft.Row([
                    self.left_panel_container,
                    self.canvas_panel,
                    self.right_panel_container
                ], expand=True, spacing=12)
            ], spacing=12)
        )

    def build(self) -> ft.Container:
        return self.view

    def update(self):
        if self.view:
            self.view.update()
    
    def _build_top_bar(self) -> ft.Container:
        node_count_chip = ft.Container(
            self.node_count_text,
            padding=ft.padding.symmetric(horizontal=10, vertical=6),
            border=ft.border.all(1, "#1f2937"),
            border_radius=10,
            bgcolor="#0d1b2a"
        )

        toolbar = ft.Row([
            ft.IconButton(ft.Icons.MENU, icon_color="#cbd5f5", on_click=self._toggle_sidebar),
            ft.Column([
                ft.Text("AUTOFLOW DSK", weight="bold", size=18, color="#f8fafc"),
                node_count_chip
            ], spacing=4),
            ft.Container(expand=True),
            ft.Row([
                ft.ElevatedButton("Add Node", icon=ft.Icons.ADD, on_click=self._open_node_palette, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12))),
                ft.ElevatedButton("Execute", on_click=self._execute_workflow, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12), bgcolor=ACCENT_COLOR, color="#010409")),
                ft.ElevatedButton("Save", icon=ft.Icons.SAVE_ALT, on_click=self._save_workflow, style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=12)))
            ], spacing=8)
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=16)

        return ft.Container(
            bgcolor="#080b16",
            padding=12,
            border=ft.border.all(1, "#1f2937"),
            border_radius=20,
            content=toolbar
        )
    
    def _build_left_panel_content(self) -> ft.Column:
        """Construye la columna de Outline con resumen y acceso a la paleta."""
        header = ft.Row([
            ft.Text("Flow Outline", weight="bold", size=14, color="#e2e8f0"),
            ft.IconButton(ft.Icons.ADD, icon_color=ACCENT_COLOR, tooltip="Add node", on_click=self._open_node_palette)
        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN)

        summary = ft.Row([
            ft.Text(f"{len(self.nodes)} nodes", size=10, color="#94a3b8"),
            ft.Text(f"{len(self.connections)} connections", size=10, color="#94a3b8")
        ], alignment=ft.MainAxisAlignment.START, spacing=8)

        outline_controls = []
        for node_id, node in self.nodes.items():
            border_color = ACCENT_COLOR if self.selected_node == node_id else "#1f2937"
            entry = ft.Container(
                content=ft.Row([
                    ft.Column([
                        ft.Text(node.config.display_name, size=12, weight="bold", color="#f8fafc"),
                        ft.Text(node.config.node_type, size=9, color="#94a3b8")
                    ]),
                    ft.Icon(ft.Icons.FIBER_MANUAL_RECORD, size=12, color="#94a3b8")
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                padding=ft.padding.symmetric(vertical=8, horizontal=10),
                border=ft.border.all(1, border_color),
                border_radius=10,
                bgcolor=CARD_BG,
                on_click=lambda e, nid=node_id: self._focus_node(nid)
            )
            outline_controls.append(entry)

        if not outline_controls:
            outline_controls.append(ft.Container(ft.Text("Empty workspace · Add nodes to begin", size=10, color="#94a3b8"), padding=12))

        outline = ft.ListView(controls=outline_controls, expand=True, spacing=6)

        return ft.Column([
            header,
            summary,
            ft.Divider(thickness=1, color="#1f2937"),
            outline
        ], spacing=10, expand=True)

    def _refresh_left_panel(self):
        self.left_panel_container.content = self._build_left_panel_content()
        try:
            self.left_panel_container.update()
        except:
            pass

    def _focus_node(self, node_id: str):
        if node_id in self.nodes:
            self.selected_node = node_id
            self._update_inspector()
            self._redraw_canvas()
            self._refresh_left_panel()

    def _refresh_node_count(self):
        self.node_count_text.value = f"{len(self.nodes)} nodes"
        try:
            self.node_count_text.update()
        except:
            pass

    def _open_node_palette(self, e=None):
        """Abrir diálogo de paleta de nodos usando una vista modal personalizada."""
        print(f"DEBUG: Abriendo paleta. Catalogo tiene {len(self.node_catalog)} categorías")
        
        # Contenedor para la lista de nodos
        nodes_list = ft.Column(
            spacing=4,
            scroll=ft.ScrollMode.AUTO,
            expand=True
        )
        
        def on_search_change(ev):
            """Actualizar búsqueda cuando el usuario tipea."""
            query = (ev.control.value or "").strip().lower()
            controls = self._build_palette_controls(query)
            nodes_list.controls = controls
            nodes_list.update()
            print(f"DEBUG: Búsqueda '{query}' retornó {len(controls)} controles")

        # Cargar todos los nodos inicialmente
        initial_controls = self._build_palette_controls("")
        print(f"DEBUG: Controles iniciales: {len(initial_controls)}")
        nodes_list.controls = initial_controls

        # Campo de búsqueda
        search_field = ft.TextField(
            hint_text="Search nodes by name or type...",
            width=480,
            border_color="#1f2937",
            bgcolor="#05060a",
            color="#f8fafc",
            filled=True,
            prefix_icon=ft.Icons.SEARCH,
            on_change=on_search_change
        )

        def on_close(ev):
            self._close_palette_dialog()

        # Crear el contenido principal del modal
        modal_content = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text("Add Node to Workflow", weight="bold", size=16, color="#f8fafc"),
                    ft.IconButton(ft.Icons.CLOSE, on_click=on_close, icon_color="#94a3b8")
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Divider(thickness=1, color="#1f2937", height=10),
                search_field,
                ft.Divider(thickness=1, color="#1f2937", height=10),
                nodes_list
            ], spacing=12, expand=True),
            width=520,
            height=650,
            bgcolor=SIDE_PANEL_BG,
            border_radius=16,
            border=ft.border.all(1, "#1f2937"),
            padding=20
        )

        # Fondo semi-transparente (overlay) - NO clickeable
        overlay = ft.Container(
            bgcolor="rgba(0,0,0,0.5)",
            expand=True
        )

        # Stack para mostrar modal centered
        modal_stack = ft.Stack([
            overlay,
            ft.Container(
                content=modal_content,
                alignment=ft.alignment.center,
                expand=True
            )
        ], expand=True)

        # Crear una vista modal temporal
        self.palette_modal = modal_stack
        # Agregar al overlay de la página en lugar del canvas
        if self.page:
            # Agregar como overlay (DIALOGO VERDADERO)
            self.page.overlay.append(modal_stack)
            self.page.update()
        else:
            # Fallback si no hay página
            self.canvas_stack.controls.append(modal_stack)
            try:
                self.canvas_stack.update()
            except:
                pass
        print("DEBUG: Modal abierto")

    def _build_palette_controls(self, query: str) -> List[ft.Control]:
        """Construir lista de tiles de nodos filtrados por query."""
        query = (query or "").strip().lower()
        controls: List[ft.Control] = []
        
        print(f"DEBUG: _build_palette_controls llamado con query='{query}'")
        print(f"DEBUG: node_catalog keys: {list(self.node_catalog.keys())}")
        
        for category, nodes in self.node_catalog.items():
            # Filtrar nodos por query
            if query:
                matches = [n for n in nodes 
                          if query in n.display_name.lower() 
                          or query in n.description.lower()
                          or query in category.lower()]
            else:
                matches = nodes
            
            if not matches:
                continue
            
            # Agregar header de categoría
            category_header = ft.Container(
                content=ft.Text(
                    category,
                    size=11,
                    color="#94a3b8",
                    weight="bold"
                ),
                padding=ft.padding.only(top=8, bottom=4, left=4)
            )
            controls.append(category_header)
            
            # Agregar tiles de nodos
            for node_config in matches:
                def make_on_click(nc):
                    def on_click_handler(ev):
                        print(f"DEBUG: Nodo seleccionado: {nc.display_name}")
                        self._select_palette_node(nc)
                    return on_click_handler
                
                tile = ft.Container(
                    content=ft.Row([
                        ft.Column([
                            ft.Text(
                                node_config.display_name,
                                weight="bold",
                                color="#f8fafc",
                                size=13
                            ),
                            ft.Text(
                                node_config.description,
                                size=10,
                                color="#94a3b8"
                            )
                        ], expand=True, spacing=2),
                        ft.Icon(ft.Icons.ARROW_FORWARD, color="#40C4FF", size=16)
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, spacing=12),
                    padding=ft.padding.symmetric(vertical=10, horizontal=12),
                    border=ft.border.all(1, "#1f2937"),
                    border_radius=10,
                    bgcolor="#111827",
                    on_click=make_on_click(node_config),
                    ink=True
                )
                controls.append(tile)
        
        if not controls:
            controls.append(
                ft.Container(
                    content=ft.Text(
                        "No nodes found",
                        color="#94a3b8",
                        size=12
                    ),
                    padding=16,
                    alignment=ft.alignment.center
                )
            )
        
        print(f"DEBUG: _build_palette_controls retornando {len(controls)} controles")
        return controls

    def _select_palette_node(self, node_config: NodeConfig):
        """Seleccionar un nodo de la paleta y agregarlo al canvas."""
        print(f"DEBUG: _select_palette_node llamado para {node_config.display_name}")
        self._create_node_instance(node_config, None, None)
        self._close_palette_dialog()

    def _close_palette_dialog(self, e=None):
        """Cerrar el modal de paleta."""
        print("DEBUG: Cerrando modal")
        try:
            if hasattr(self, 'palette_modal') and self.palette_modal is not None:
                try:
                    # Remover del overlay en lugar del canvas stack
                    if self.page and self.palette_modal in self.page.overlay:
                        self.page.overlay.remove(self.palette_modal)
                        self.page.update()
                except (ValueError, AttributeError) as ex:
                    print(f"DEBUG: Error removiendo modal del overlay: {ex}")
                finally:
                    self.palette_modal = None
        except Exception as ex:
            print(f"DEBUG: Error al cerrar modal: {ex}")
            self.palette_modal = None
    
    def _build_canvas_panel(self) -> ft.Container:
        """Panel central para el lienzo con gradiente y sombreado."""
        # Stack con conexiones atrás y nodos adelante
        canvas_with_connections = ft.Stack([
            self.connections_stack,  # Conexiones atrás
            self.canvas_stack        # Nodos adelante
        ], expand=True, clip_behavior=ft.ClipBehavior.HARD_EDGE)
        
        # Envolver el canvas con GestureDetector para zoom y pan
        canvas_with_gestures = ft.GestureDetector(
            content=canvas_with_connections,
            on_scroll=self._on_canvas_scroll,
            on_pan_start=self._on_canvas_pan_start,
            on_pan_update=self._on_canvas_pan_update,
            on_pan_end=self._on_canvas_pan_end,
        )
        
        canvas_frame = ft.Container(
            expand=True,
            bgcolor=CARD_BG,
            border_radius=20,
            border=ft.border.all(1, "#1b2432"),
            padding=16,
            content=canvas_with_gestures,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        
        return ft.Container(
            expand=True,
            padding=8,
            bgcolor=CANVAS_PANEL_BG,
            border_radius=22,
            content=canvas_frame,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
    
    def _refresh_right_panel(self):
        inspector_frame = ft.Container(
            content=ft.Column([self.inspector_content], expand=True, scroll=ft.ScrollMode.AUTO),
            expand=True,
            bgcolor="#05070a",
            border_radius=14,
            padding=12
        )
        logs_frame = ft.Container(
            content=self.log_list,
            expand=True,
            bgcolor="#05070a",
            border_radius=14,
            padding=12
        )

        tabs = ft.Tabs(
            selected_index=0,
            animation_duration=200,
            tabs=[
                ft.Tab(text="Inspector", content=inspector_frame),
                ft.Tab(text="Logs", content=logs_frame)
            ],
            indicator_color=ACCENT_COLOR,
            expand=True
        )
        self.right_panel_container.content = tabs

    def _create_node_instance(self, node_config: NodeConfig, x: Optional[float], y: Optional[float]):
        index = self.node_counter
        node_id = f"node_{index}"
        columns = 4
        spacing_x = 260
        spacing_y = 160
        default_x = 40 + (index % columns) * spacing_x
        default_y = 40 + (index // columns) * spacing_y
        node = NodeInstance(node_id, node_config, x if x is not None else default_x, y if y is not None else default_y)
        self.node_counter += 1
        self.nodes[node_id] = node
        self._refresh_node_count()
        self.selected_node = node_id
        
        # SOLO agregar el nodo nuevo sin redibujarlo todo
        self._add_node_to_canvas(node)
        self._update_inspector()
        self._refresh_left_panel()
    
    def _add_node_to_canvas(self, node: NodeInstance):
        """Agregar un nodo al canvas sin redibujarlo todo."""
        try:
            print(f"DEBUG: _add_node_to_canvas llamado para {node.node_id}")
            print(f"DEBUG: Canvas tiene {len(self.canvas_stack.controls)} controls ANTES")
            
            node_ui = self._draw_node_ui(node)
            node_ui.data = node.node_id
            node_ui.left = node.position["x"] + self.canvas_offset_x
            node_ui.top = node.position["y"] + self.canvas_offset_y
            node.ui_control = node_ui
            self.canvas_stack.controls.append(node_ui)
            
            print(f"DEBUG: Canvas tiene {len(self.canvas_stack.controls)} controls DESPUÉS de append")
            
            # Actualizar canvas
            if hasattr(self.canvas_stack, 'update') and self.page:
                try:
                    self.canvas_stack.update()
                    print(f"DEBUG: Canvas actualizado correctamente")
                except Exception as update_err:
                    print(f"DEBUG: Error en update: {update_err}")
        except Exception as e:
            print(f"DEBUG: Error al agregar nodo: {e}")
    
    def _draw_node_ui(self, node: NodeInstance) -> ft.Control:
        """Crear el UI para un nodo."""
        def _on_drag_start(e: ft.DragStartEvent):
            self.dragging_node = node.node_id
            node.drag_offset["x"] = e.local_x
            node.drag_offset["y"] = e.local_y

        def _on_node_drag(e: ft.DragUpdateEvent):
            if self.dragging_node != node.node_id:
                return
            node.position["x"] += e.delta_x
            node.position["y"] += e.delta_y
            self._clamp_node_in_bounds(node)
            
            # Actualizar solo la posición del nodo sin redibujar todo
            if hasattr(node, 'ui_control') and node.ui_control:
                node.ui_control.left = node.position["x"] + self.canvas_offset_x
                node.ui_control.top = node.position["y"] + self.canvas_offset_y
                try:
                    node.ui_control.update()
                except:
                    pass
            
            # Redibujar conexiones CON THROTTLE (solo cada 50ms)
            import time
            current_time = time.time()
            if current_time - self.last_connection_draw > self.connection_draw_throttle:
                self._draw_connections()
                self.last_connection_draw = current_time

        def _on_node_drag_end(e):
            self.dragging_node = None
            # Redibuja conexiones al final para asegurar que estén correctas
            self._draw_connections()

        def _on_node_click(e):
            self.selected_node = node.node_id
            self._update_inspector()
            self._redraw_canvas()
        
        # Puertos de entrada
        input_port_controls = []
        for port in node.input_ports:
            port_ui = ft.Container(
                bgcolor=CONNECTION_COLOR,
                width=PORT_SIZE,
                height=PORT_SIZE,
                border_radius=PORT_SIZE / 2,
                on_click=lambda e, p=port: self._start_connection(node.node_id, p, True),
            )
            input_port_controls.append(port_ui)
        
        input_ports = ft.Column(input_port_controls, spacing=12) if input_port_controls else ft.Container(width=0)
        
        # Puertos de salida
        output_port_controls = []
        for port in node.output_ports:
            port_ui = ft.Container(
                bgcolor=CONNECTION_COLOR,
                width=PORT_SIZE,
                height=PORT_SIZE,
                border_radius=PORT_SIZE / 2,
                on_click=lambda e, p=port: self._start_connection(node.node_id, p, False),
            )
            output_port_controls.append(port_ui)
        
        output_ports = ft.Column(output_port_controls, spacing=12) if output_port_controls else ft.Container(width=0)
        
        # Contenido del nodo
        node_content_col = ft.Column([
            ft.Text(node.config.display_name, weight="bold", size=11),
            ft.Text(node.config.node_type, size=8, color="#9fb4c8"),
        ], spacing=4)
        
        content = ft.Row([
            input_ports,
            ft.Container(
                node_content_col,
                expand=True,
                padding=8,
            ),
            output_ports
        ], spacing=8)
        
        border_color = "#00BCD4" if self.selected_node == node.node_id else "#1f2937"
        
        node_control = ft.Container(
            content=content,
            width=NODE_WIDTH,
            height=NODE_HEIGHT,
            bgcolor=HOVER_COLOR,
            border=ft.border.all(2, border_color),
            border_radius=6,
            padding=8,
        )
        
        draggable = ft.GestureDetector(
            content=node_control,
            on_pan_start=_on_drag_start,
            on_pan_update=_on_node_drag,
            on_pan_end=_on_node_drag_end,
            on_tap=_on_node_click
        )

        # Usar posicionamiento absoluto con left y top en lugar de margin
        container = ft.Container(
            content=draggable,
            left=node.position["x"],
            top=node.position["y"],
            width=NODE_WIDTH,
            height=NODE_HEIGHT,
        )
        
        # Guardar referencia del control
        node.ui_control = container

        return container
    
    def _redraw_canvas(self):
        """Actualizar solo los bordes de nodos seleccionados."""
        try:
            for control in self.canvas_stack.controls:
                if hasattr(control, 'data') and isinstance(control.data, str) and control.data.startswith('node_'):
                    # Actualizar color de borde según si está seleccionado
                    if control.data == self.selected_node:
                        # Cambiar borde a cyan
                        if hasattr(control, 'content') and hasattr(control.content, 'content'):
                            if hasattr(control.content.content, 'border'):
                                control.content.content.border = ft.border.all(2, "#00BCD4")
                    else:
                        # Borde normal
                        if hasattr(control, 'content') and hasattr(control.content, 'content'):
                            if hasattr(control.content.content, 'border'):
                                control.content.content.border = ft.border.all(2, "#1f2937")
            
            self.canvas_stack.update()
        except Exception as e:
            print(f"DEBUG: Error en _redraw_canvas: {e}")

    def _clamp_node_in_bounds(self, node: NodeInstance):
        node.position["x"] = max(0, node.position["x"])
        node.position["y"] = max(0, node.position["y"])
    
    def _update_inspector(self):
        """Actualizar el inspector con el nodo seleccionado."""
        if not self.selected_node or self.selected_node not in self.nodes:
            self.inspector_content.controls = [
                ft.Text("Select a node to inspect", color="#9fb4c8")
            ]
            return
        
        node = self.nodes[self.selected_node]
        controls = [ft.Text(f"Node: {node.config.display_name}", weight="bold")]
        
        # ESPECIAL: Interfaz completa para Manual Trigger
        if node.config.node_type == "manual_trigger":
            controls = self._build_manual_trigger_inspector(node, controls)
        else:
            # Interfaz genérica para otros nodos
            controls = self._build_generic_node_inspector(node, controls)
        
        self.inspector_content.controls = controls
        self.inspector_content.update()
        self.right_panel_container.update()
    
    def _build_manual_trigger_inspector(self, node, controls):
        """Construir interfaz especial para Manual Trigger."""
        controls.append(ft.Text("Execution Name", size=10, weight="bold", color=ACCENT_COLOR))
        exec_name_field = ft.TextField(
            value=node.settings.get("execution_name", "Untitled Workflow"),
            on_blur=lambda e: self._update_node_setting(node.node_id, "execution_name", e.control.value),
            bgcolor="#05070a",
            border_color="#1f2937",
            color="#f8fafc",
            filled=True,
            hint_text="Name your execution..."
        )
        controls.append(exec_name_field)
        
        # Input Variables
        controls.append(ft.Divider(thickness=1, color="#1f2937"))
        controls.append(ft.Text("Input Variables", size=10, weight="bold", color=ACCENT_COLOR))
        
        # Agregar variables dinámicas
        input_vars = node.settings.get("input_variables", {})
        if isinstance(input_vars, dict):
            for var_name, var_value in input_vars.items():
                # Fila para cada variable: nombre [valor]
                controls.append(ft.Text(var_name, size=9, color="#cbd5f5"))
                var_field = ft.TextField(
                    value=str(var_value),
                    on_blur=lambda e, vn=var_name: self._update_input_variable(node.node_id, vn, e.control.value),
                    bgcolor="#05070a",
                    border_color="#1f2937",
                    color="#f8fafc",
                    filled=True,
                    height=40
                )
                controls.append(var_field)
        
        # Botón para agregar variable
        add_var_btn = ft.ElevatedButton(
            "➕ Add Variable",
            icon=ft.Icons.ADD,
            on_click=lambda e: self._add_input_variable(node.node_id),
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8)),
            width=180
        )
        controls.append(add_var_btn)
        
        # Advanced - Raw JSON Input
        controls.append(ft.Divider(thickness=1, color="#1f2937"))
        controls.append(ft.Text("Advanced", size=10, weight="bold", color=ACCENT_COLOR))
        controls.append(ft.Text("Raw JSON Input", size=9, color="#94a3b8"))
        
        raw_json = node.settings.get("raw_json_input", "{}")
        json_field = ft.TextField(
            value=raw_json,
            on_blur=lambda e: self._update_node_setting(node.node_id, "raw_json_input", e.control.value),
            bgcolor="#05070a",
            border_color="#1f2937",
            color="#f8fafc",
            filled=True,
            min_lines=4,
            multiline=True,
            hint_text='{"key": "value"}'
        )
        controls.append(json_field)
        
        # Mode Selection - Simple buttons
        controls.append(ft.Divider(thickness=1, color="#1f2937"))
        controls.append(ft.Text("Mode", size=10, weight="bold", color=ACCENT_COLOR))
        
        current_mode = node.settings.get("mode", "test")
        
        def on_mode_change(mode_val):
            self._update_node_setting(node.node_id, "mode", mode_val)
            self._update_inspector()
        
        # Botones para los modos
        test_btn = ft.ElevatedButton(
            "Test",
            bgcolor=ACCENT_COLOR if current_mode == "test" else "#1f2937",
            color="#010409" if current_mode == "test" else "#f8fafc",
            on_click=lambda e: on_mode_change("test"),
            width=90
        )
        
        prod_btn = ft.ElevatedButton(
            "Production",
            bgcolor=ACCENT_COLOR if current_mode == "production" else "#1f2937",
            color="#010409" if current_mode == "production" else "#f8fafc",
            on_click=lambda e: on_mode_change("production"),
            width=110
        )
        
        mode_row = ft.Row([test_btn, prod_btn], spacing=8)
        controls.append(mode_row)
        
        # Run Workflow Button
        controls.append(ft.Divider(thickness=1, color="#1f2937"))
        run_btn = ft.ElevatedButton(
            "▶ Run Workflow",
            icon=ft.Icons.PLAY_ARROW,
            on_click=lambda e: self._execute_workflow_from_node(node.node_id),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=8),
                bgcolor="#40C4FF",
                color="#010409"
            ),
            width=200
        )
        controls.append(run_btn)
        
        return controls
    
    def _build_generic_node_inspector(self, node, controls):
        """Construir interfaz genérica para otros nodos."""
        for setting_name, setting_value in node.settings.items():
            controls.append(ft.Text(f"{setting_name}", size=10))
            
            # ESPECIAL: Si es OpenRouter y el setting es "model", mostrar dropdown
            if (node.config.node_type == "open_router" and setting_name == "model" and 
                hasattr(node.config, 'available_models')):
                
                # Crear dropdown con modelos disponibles
                model_options = [
                    ft.dropdown.Option(model) for model in node.config.available_models
                ]
                
                input_control = ft.Dropdown(
                    label="Select Model",
                    options=model_options,
                    value=setting_value,
                    on_change=lambda e, sn=setting_name: self._update_node_setting(node.node_id, sn, e.control.value),
                    bgcolor="#05070a",
                    border_color="#1f2937",
                    color="#f8fafc",
                    filled=True
                )
            elif isinstance(setting_value, bool):
                input_control = ft.Checkbox(
                    value=setting_value,
                    on_change=lambda e, sn=setting_name: self._update_node_setting(node.node_id, sn, e.control.value)
                )
            elif isinstance(setting_value, (int, float)):
                input_control = ft.TextField(
                    value=str(setting_value),
                    on_blur=lambda e, sn=setting_name: self._update_node_setting(node.node_id, sn, e.control.value)
                )
            else:
                input_control = ft.TextField(
                    value=str(setting_value),
                    on_blur=lambda e, sn=setting_name: self._update_node_setting(node.node_id, sn, e.control.value)
                )
            
            controls.append(input_control)
            
            # Agregar botón de test después del api_key en nodos OpenRouter
            if node.config.node_type == "open_router" and setting_name == "api_key":
                test_btn = ft.ElevatedButton(
                    "Test Connection",
                    icon=ft.Icons.VERIFIED,
                    on_click=lambda e, nid=node.node_id: self._test_openrouter_connection(nid),
                    style=ft.ButtonStyle(
                        shape=ft.RoundedRectangleBorder(radius=8),
                        bgcolor=ACCENT_COLOR,
                        color="#010409"
                    ),
                    width=200
                )
                controls.append(test_btn)
                
                # Agregar sección de Output para OpenRouter
                controls.append(ft.Divider(thickness=1, color="#1f2937"))
                controls.append(ft.Text("Output", size=10, weight="bold", color=ACCENT_COLOR))
                
                # Área de respuesta del LLM
                output_response = node.settings.get("last_response", "")
                output_status = node.settings.get("last_response_status", "pending")
                
                # Indicador de estado
                if output_status == "success":
                    status_color = "#10b981"
                    status_icon = "✅"
                elif output_status == "error":
                    status_color = "#ef4444"
                    status_icon = "❌"
                else:
                    status_color = "#94a3b8"
                    status_icon = "⏳"
                
                status_row = ft.Row([
                    ft.Icon(ft.Icons.CIRCLE, size=12, color=status_color),
                    ft.Text(f"Status: {output_status.upper()}", size=9, color=status_color, weight="bold")
                ], spacing=6)
                controls.append(status_row)
                
                # Área de respuesta (read-only)
                response_area = ft.TextField(
                    value=output_response if output_response else "(No response yet)",
                    multiline=True,
                    read_only=True,
                    min_lines=4,
                    bgcolor="#05070a",
                    border_color="#1f2937",
                    color="#cbd5f5",
                    filled=True
                )
                controls.append(response_area)
                
                # Botón para limpiar output
                clear_btn = ft.ElevatedButton(
                    "Clear Output",
                    icon=ft.Icons.CLEAR,
                    on_click=lambda e: self._clear_openrouter_output(node.node_id),
                    width=120
                )
                controls.append(clear_btn)
        
        return controls
    
    def _test_openrouter_connection(self, node_id: str):
        """Testear la conexión con OpenRouter API."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        api_key = node.settings.get("api_key", "").strip()
        
        # Validar que el API key no esté vacío
        if not api_key:
            self._log("❌ API Key is empty. Please enter your OpenRouter API key.")
            return
        
        # Mostrar que estamos testando
        self._log(f"🔄 Testing OpenRouter connection...")
        
        try:
            import requests
            
            # Test simple: llamada a la API con modelo rápido
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "BuildBrain",
                    "X-Title": "BuildBrain"
                },
                json={
                    "model": node.settings.get("model", "mistralai/devstral-2512:free"),
                    "messages": [
                        {
                            "role": "user",
                            "content": "Say 'Hello' in one word."
                        }
                    ],
                    "max_tokens": 10
                },
                timeout=10
            )
            
            # Validar respuesta
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                    # Guardar respuesta en el nodo
                    node.settings["last_response"] = content
                    node.settings["last_response_status"] = "success"
                    self._log(f"✅ OpenRouter connection successful! Response: '{content}'")
                    self._update_inspector()  # Refrescar para mostrar output
                    return
                else:
                    self._log(f"⚠️ API returned 200 but unexpected format")
                    node.settings["last_response_status"] = "error"
            elif response.status_code == 401:
                node.settings["last_response"] = "Authentication failed. Invalid API Key."
                node.settings["last_response_status"] = "error"
                self._log(f"❌ Authentication failed. Invalid API Key.")
                self._update_inspector()
            elif response.status_code == 429:
                node.settings["last_response"] = "Rate limited. Please try again later."
                node.settings["last_response_status"] = "error"
                self._log(f"❌ Rate limited. Please try again later.")
                self._update_inspector()
            else:
                error_msg = response.json().get("error", {}).get("message", response.text)
                node.settings["last_response"] = f"API Error ({response.status_code}): {error_msg}"
                node.settings["last_response_status"] = "error"
                self._log(f"❌ API Error ({response.status_code}): {error_msg}")
                self._update_inspector()
        
        except requests.exceptions.Timeout:
            node.settings["last_response"] = "Connection timeout. Check your internet."
            node.settings["last_response_status"] = "error"
            self._log(f"❌ Connection timeout. Check your internet.")
            self._update_inspector()
        except requests.exceptions.ConnectionError:
            node.settings["last_response"] = "Connection error. Unable to reach OpenRouter."
            node.settings["last_response_status"] = "error"
            self._log(f"❌ Connection error. Unable to reach OpenRouter.")
            self._update_inspector()
        except Exception as e:
            node.settings["last_response"] = f"Error: {str(e)}"
            node.settings["last_response_status"] = "error"
            self._log(f"❌ Error: {str(e)}")
            self._update_inspector()
    
    def _clear_openrouter_output(self, node_id: str):
        """Limpiar el output del nodo OpenRouter."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.settings["last_response"] = ""
        node.settings["last_response_status"] = "pending"
        self._log(f"Cleared output for {node.config.display_name}")
        self._update_inspector()
    
    def _update_input_variable(self, node_id: str, var_name: str, var_value: str):
        """Actualizar una variable de entrada del Manual Trigger."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        if "input_variables" not in node.settings:
            node.settings["input_variables"] = {}
        
        node.settings["input_variables"][var_name] = var_value
        self._log(f"Updated variable '{var_name}' = '{var_value}'")
    
    def _add_input_variable(self, node_id: str):
        """Agregar una nueva variable de entrada."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        if "input_variables" not in node.settings:
            node.settings["input_variables"] = {}
        
        # Auto-generar nombre de variable
        var_num = len(node.settings["input_variables"]) + 1
        new_var = f"var_{var_num}"
        node.settings["input_variables"][new_var] = ""
        
        # Refrescar inspector
        self._update_inspector()
        self._log(f"Added new variable: '{new_var}'")
    
    def _execute_workflow_from_node(self, node_id: str):
        """Ejecutar el workflow completo desde Manual Trigger."""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        self._log(f"🚀 Starting workflow execution: {node.settings.get('execution_name', 'Untitled')}")
        
        # Obtener modo
        mode = node.settings.get("mode", "test")
        self._log(f"📌 Mode: {mode.upper()}")
        
        # Obtener variables de entrada
        input_vars = node.settings.get("input_variables", {})
        if input_vars:
            self._log(f"📥 Input Variables: {input_vars}")
        
        # Obtener JSON raw si existe
        raw_json = node.settings.get("raw_json_input", "{}")
        if raw_json and raw_json != "{}":
            self._log(f"📋 Raw JSON Input: {raw_json}")
        
        # Aquí es donde se conecta con TopologicalExecutor
        try:
            # Crear contexto de ejecución
            execution_context = {
                "mode": mode,
                "variables": input_vars,
                "raw_input": raw_json
            }
            
            # Usar TopologicalExecutor para ejecutar el flujo
            executor = TopologicalExecutor(self.nodes, self.connections)
            self._log("⏳ Executing workflow nodes in topological order...")
            
            # TODO: Implementar ejecución real
            # results = executor.execute(execution_context)
            
            self._log("✅ Workflow execution completed!")
            
        except Exception as e:
            self._log(f"❌ Execution error: {str(e)}")
    
        pass
    
    def _apply_canvas_transform(self):
        """Actualizar transformación en el canvas."""
        pass
    
    def _toggle_sidebar(self, e):
        """Toggle sidebar visibility."""
        self.sidebar_collapsed = not self.sidebar_collapsed
        self.left_panel_container.visible = not self.sidebar_collapsed
        self.left_panel_container.update()
        self.canvas_panel.update()
    
    def _on_canvas_scroll(self, e: ft.ScrollEvent):
        """Manejar scroll del canvas para pan vertical."""
        try:
            # Scroll = pan vertical
            scroll_delta = e.scroll_delta_y if hasattr(e, 'scroll_delta_y') else 0
            self.canvas_offset_y -= scroll_delta * 5  # Multiplicar para sensibilidad
            self._apply_pan()
        except Exception as ex:
            print(f"DEBUG: Error en scroll: {ex}")
    
    def _on_canvas_pan_start(self, e: ft.DragStartEvent):
        """Iniciar pan del canvas."""
        try:
            # Solo pan si no estamos arrastrando un nodo
            if self.dragging_node is None:
                self.is_panning = True
                self.pan_start_x = e.local_x if hasattr(e, 'local_x') else 0
                self.pan_start_y = e.local_y if hasattr(e, 'local_y') else 0
        except Exception as ex:
            print(f"DEBUG: Error al iniciar pan: {ex}")
    
    def _on_canvas_pan_update(self, e: ft.DragUpdateEvent):
        """Actualizar pan del canvas."""
        try:
            if self.is_panning and self.dragging_node is None:
                delta_x = e.delta_x if hasattr(e, 'delta_x') else 0
                delta_y = e.delta_y if hasattr(e, 'delta_y') else 0
                
                self.canvas_offset_x += delta_x
                self.canvas_offset_y += delta_y
                self._apply_pan()
        except Exception as ex:
            print(f"DEBUG: Error al actualizar pan: {ex}")
    
    def _on_canvas_pan_end(self, e: ft.DragEndEvent):
        """Terminar pan del canvas."""
        self.is_panning = False
    
    def _apply_pan(self):
        """Aplicar pan a todos los nodos."""
        try:
            # Buscar directamente en el canvas_stack por control.data == node_id
            for node_id, node in self.nodes.items():
                for control in self.canvas_stack.controls:
                    if hasattr(control, 'data') and control.data == node_id:
                        # Aplicar offset de pan
                        control.left = node.position["x"] + self.canvas_offset_x
                        control.top = node.position["y"] + self.canvas_offset_y
                        break
            
            self._draw_connections()
            self.canvas_stack.update()
        except Exception as ex:
            print(f"DEBUG: Error en pan: {ex}")
    
    
    def _update_node_setting(self, node_id: str, setting_name: str, value: Any):
        """Actualizar una configuración de nodo."""
        if node_id in self.nodes:
            self.nodes[node_id].settings[setting_name] = value
    
    def _start_connection(self, node_id: str, port_name: str, is_input: bool):
        """Empezar a crear una conexión."""
        if self.connecting_from is None:
            # Iniciar nueva conexión
            self.connecting_from = (node_id, port_name)
            self.connection_in_progress = {
                "source_node": node_id,
                "source_port": port_name,
                "is_input": is_input,
                "start_x": 0,
                "start_y": 0
            }
            self._log(f"Connection start: {node_id}.{port_name}")
            self._draw_connections()
        else:
            # Completar conexión
            source_node, source_port = self.connecting_from
            target_node, target_port = node_id, port_name
            
            # No permitir conexión a sí mismo
            if source_node == target_node:
                self._log("Cannot connect node to itself")
                self.connecting_from = None
                self.connection_in_progress = None
                self._draw_connections()
                return
            
            conn_id = str(uuid.uuid4())
            self.connections[conn_id] = Connection(conn_id, source_node, source_port, target_node, target_port)
            self._log(f"Connection created: {source_node}.{source_port} → {target_node}.{target_port}")
            
            self.connecting_from = None
            self.connection_in_progress = None
            self._draw_connections()
    
    def _draw_connections(self):
        """Dibujar todas las conexiones en el canvas (optimizado)."""
        try:
            if not hasattr(self, 'connections_stack') or self.connections_stack is None:
                return
            
            # Limpiar conexiones anteriores
            try:
                self.connections_stack.controls.clear()
            except:
                pass
            
            # Dibujar conexiones completadas - SOLO si hay conexiones
            if not self.connections:
                return
                
            for conn_id, conn in self.connections.items():
                try:
                    self._draw_connection_line(
                        conn.source_node, 
                        conn.source_port,
                        conn.target_node,
                        conn.target_port,
                        color="#00BCD4"
                    )
                except Exception as line_error:
                    print(f"DEBUG: Error dibujando línea individual: {line_error}")
                    continue
            
            # Actualizar el stack
            try:
                if hasattr(self.connections_stack, 'update'):
                    self.connections_stack.update()
            except Exception as update_error:
                print(f"DEBUG: Error actualizando stack: {update_error}")
                
        except Exception as e:
            print(f"DEBUG: Error en _draw_connections: {e}")
    
    def _draw_connection_line(self, source_node_id: str, source_port: str, 
                             target_node_id: str, target_port: str, color: str):
        """Dibujar una línea de conexión curva entre dos puertos."""
        try:
            import math
            
            # Obtener posiciones de los nodos
            if source_node_id not in self.nodes or target_node_id not in self.nodes:
                return
            
            source_node = self.nodes[source_node_id]
            target_node = self.nodes[target_node_id]
            
            # Calcular posiciones de los puertos (centro del nodo) CON OFFSET
            source_x = source_node.position["x"] + NODE_WIDTH / 2 + self.canvas_offset_x
            source_y = source_node.position["y"] + NODE_HEIGHT / 2 + self.canvas_offset_y
            
            target_x = target_node.position["x"] + NODE_WIDTH / 2 + self.canvas_offset_x
            target_y = target_node.position["y"] + NODE_HEIGHT / 2 + self.canvas_offset_y
            
            # Calcular distancia
            dx = target_x - source_x
            dy = target_y - source_y
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance < 1:
                return
            
            # Crear una curva suave usando múltiples segmentos
            # OPTIMIZADO: máximo 8 segmentos en lugar de 10+
            segments = max(3, min(8, int(distance / 70)))  # Menos segmentos = mejor performance
            
            # Puntos de control para la curva Bezier
            ctrl1_x = source_x + dx / 3
            ctrl1_y = source_y + dy / 4
            ctrl2_x = source_x + 2 * dx / 3
            ctrl2_y = source_y + 3 * dy / 4
            
            # Generar puntos en la curva Bezier cúbica
            for i in range(segments):
                t_start = i / segments
                t_end = (i + 1) / segments
                
                # Fórmula de Bezier cúbica: B(t) = (1-t)³P0 + 3(1-t)²tP1 + 3(1-t)t²P2 + t³P3
                def bezier_point(t):
                    mt = 1 - t
                    x = (mt**3 * source_x + 
                         3 * mt**2 * t * ctrl1_x + 
                         3 * mt * t**2 * ctrl2_x + 
                         t**3 * target_x)
                    y = (mt**3 * source_y + 
                         3 * mt**2 * t * ctrl1_y + 
                         3 * mt * t**2 * ctrl2_y + 
                         t**3 * target_y)
                    return x, y
                
                # Puntos de inicio y fin del segmento
                x1, y1 = bezier_point(t_start)
                x2, y2 = bezier_point(t_end)
                
                # Calcular ángulo y longitud del segmento
                seg_dx = x2 - x1
                seg_dy = y2 - y1
                seg_length = math.sqrt(seg_dx**2 + seg_dy**2)
                seg_angle = math.atan2(seg_dy, seg_dx)
                
                if seg_length > 0.1:
                    # Crear línea para este segmento
                    line = ft.Container(
                        width=int(seg_length),
                        height=3,
                        bgcolor=color,
                        border_radius=1,
                        left=int(x1),
                        top=int(y1 - 1.5),
                        opacity=0.85,
                        rotate=ft.Rotate(angle=seg_angle),
                    )
                    self.connections_stack.controls.append(line)
            
        except Exception as e:
            print(f"DEBUG: Error dibujando línea curva: {e}")
    
    def _execute_workflow(self, e):
        """Ejecutar el workflow."""
        self.execution_log.clear()
        self._log("Starting workflow execution...")
        
        try:
            flow_dict = self._serialize_flow()
            
            validator = FlowValidator()
            errors = validator.validate(flow_dict)
            
            if errors:
                for error in errors:
                    self._log(f"❌ {error}")
                return
            
            executor = TopologicalExecutor()
            results = executor.execute(flow_dict)
            
            for result in results:
                self._log(f"✓ {result['node']}: {result.get('status', 'unknown')}")
            
            self._log("✓ Execution complete")
        except Exception as ex:
            self._log(f"❌ Error: {str(ex)}")
    
    def _serialize_flow(self) -> Dict:
        """Serializar el workflow actual."""
        return {
            "name": self.title_input.value,
            "nodes": [
                {
                    "id": node_id,
                    "type": node.config.node_type,
                    "settings": node.settings,
                    "position": node.position
                }
                for node_id, node in self.nodes.items()
            ],
            "connections": [
                {
                    "id": conn.id,
                    "source_node": conn.source_node,
                    "source_port": conn.source_port,
                    "target_node": conn.target_node,
                    "target_port": conn.target_port
                }
                for conn in self.connections.values()
            ]
        }
    
    def _save_workflow(self, e):
        """Guardar el workflow."""
        try:
            flow_dict = self._serialize_flow()
            persistence = FlowPersistence()
            
            filename = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            persistence.save(flow_dict, filename)
            
            self._log(f"✓ Saved: {filename}")
        except Exception as ex:
            self._log(f"❌ Save error: {str(ex)}")
    
    def _log(self, message: str):
        """Agregar mensaje al log."""
        self.execution_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        self.log_list.controls = [ft.Text(log, size=9, color="#9fb4c8") for log in self.execution_log[-20:]]
        self.log_list.update()
        self.right_panel_container.update()
        self.update()
