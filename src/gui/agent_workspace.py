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
        self.sidebar_collapsed = False

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
        # Canvas stack con tamaño mínimo para evitar que se expanda con los nodos
        self.canvas_stack = ft.Stack(
            expand=True, 
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
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
        self._add_node_to_canvas(node_config)
        self._close_palette_dialog()

    def _close_palette_dialog(self, e=None):
        """Cerrar el modal de paleta."""
        print("DEBUG: Cerrando modal")
        try:
            if hasattr(self, 'palette_modal') and self.palette_modal is not None:
                if self.palette_modal in self.canvas_stack.controls:
                    self.canvas_stack.controls.remove(self.palette_modal)
                    try:
                        self.canvas_stack.update()
                    except:
                        pass
                self.palette_modal = None
        except Exception as ex:
            print(f"DEBUG: Error al cerrar modal: {ex}")
            self.palette_modal = None
    
    def _build_canvas_panel(self) -> ft.Container:
        """Panel central para el lienzo con gradiente y sombreado."""
        canvas_frame = ft.Container(
            expand=True,
            bgcolor=CARD_BG,
            border_radius=20,
            border=ft.border.all(1, "#1b2432"),
            padding=16,
            content=self.canvas_stack,
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
            content=self.inspector_content,
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
        
    def _add_node_to_canvas(self, node_config: NodeConfig, x: Optional[float] = None, y: Optional[float] = None):
        """Agregar un nodo al canvas con la nueva lógica de paleta."""
        self._create_node_instance(node_config, x, y)

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
        self._update_inspector()
        self._refresh_left_panel()
        self._redraw_canvas()
    
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
                node.ui_control.left = node.position["x"]
                node.ui_control.top = node.position["y"]
                try:
                    node.ui_control.update()
                except:
                    pass

        def _on_node_drag_end(e):
            self.dragging_node = None

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
        )
        
        # Guardar referencia del control
        node.ui_control = container

        return container
    
    def _redraw_canvas(self):
        """Redibujar todo el canvas."""
        try:
            # Mantener solo los nodos actuales, actualizando los existentes
            current_node_ids = set(self.nodes.keys())
            canvas_node_ids = set()
            
            # Identificar qué nodos ya están en el canvas
            for i, control in enumerate(self.canvas_stack.controls):
                # Buscar si es un nodo (tiene data atributo con node_id)
                if hasattr(control, 'data'):
                    canvas_node_ids.add(control.data)
            
            # Limpiar nodos que fueron eliminados
            to_remove = []
            for i, control in enumerate(self.canvas_stack.controls):
                if hasattr(control, 'data') and control.data not in current_node_ids:
                    to_remove.append(i)
            
            # Remover en orden inverso para no cambiar índices
            for i in sorted(to_remove, reverse=True):
                self.canvas_stack.controls.pop(i)
            
            # Agregar nodos nuevos o actualizar existentes
            for node_id, node in self.nodes.items():
                if node_id not in canvas_node_ids:
                    node_ui = self._draw_node_ui(node)
                    node_ui.data = node_id
                    self.canvas_stack.controls.append(node_ui)
                else:
                    # Actualizar posición del nodo existente
                    for control in self.canvas_stack.controls:
                        if hasattr(control, 'data') and control.data == node_id:
                            control.left = node.position["x"]
                            control.top = node.position["y"]
                            break
            
            # Solo actualizar si se agregaron/removieron nodos
            if to_remove or len(current_node_ids) > len(canvas_node_ids):
                if hasattr(self.canvas_stack, 'update') and self.page:
                    try:
                        self.canvas_stack.update()
                    except:
                        pass
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
        
        for setting_name, setting_value in node.settings.items():
            controls.append(ft.Text(f"{setting_name}", size=10))
            
            if isinstance(setting_value, bool):
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
        
        self.inspector_content.controls = controls
        self.inspector_content.update()
        self.right_panel_container.update()
    
    def _toggle_sidebar(self, e):
        """Toggle sidebar visibility."""
        self.sidebar_collapsed = not self.sidebar_collapsed
        self.left_panel_container.visible = not self.sidebar_collapsed
        self.left_panel_container.update()
        self.canvas_panel.update()
    
    def _update_node_setting(self, node_id: str, setting_name: str, value: Any):
        """Actualizar una configuración de nodo."""
        if node_id in self.nodes:
            self.nodes[node_id].settings[setting_name] = value
    
    def _start_connection(self, node_id: str, port_name: str, is_input: bool):
        """Empezar a crear una conexión."""
        if self.connecting_from is None:
            self.connecting_from = (node_id, port_name)
            self._log(f"Connection start: {node_id}.{port_name}")
        else:
            source_node, source_port = self.connecting_from
            target_node, target_port = node_id, port_name
            
            conn_id = str(uuid.uuid4())
            self.connections[conn_id] = Connection(conn_id, source_node, source_port, target_node, target_port)
            self._log(f"Connection created: {source_node}.{source_port} -> {target_node}.{target_port}")
            
            self.connecting_from = None
            self._redraw_canvas()
    
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
