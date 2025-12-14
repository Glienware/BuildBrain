import os
import flet as ft
from typing import Callable, Optional
from .nodes_panel import NodesPanel


class AgentWorkspace:
    """Experiencia dedicada para crear agentes de IA con nodos."""

    def __init__(self, page: ft.Page, agents_dir: str, on_back_to_welcome: Callable):
        self.page = page
        self.agents_dir = agents_dir
        self.on_back = on_back_to_welcome
        os.makedirs(self.agents_dir, exist_ok=True)
        self.nodes_panel = NodesPanel(dataset_path=self.agents_dir)

    def build(self) -> ft.Control:
        header = ft.Container(
            content=ft.Row([
                ft.Icon(ft.Icons.ANDROID, size=32, color="#FF6D00"),
                ft.Container(width=12),
                ft.Column([
                    ft.Text("Agente de IA", size=26, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                    ft.Text("Define agentes inteligentes con nodos estilizados específicamente para automatizaciones",
                            size=14, color="#AAAAAA"),
                ], spacing=0, expand=True),
                    ft.ElevatedButton(
                        "Volver",
                        icon=ft.Icons.ARROW_BACK,
                        on_click=lambda e: self.on_back(),
                        style=ft.ButtonStyle(bgcolor="#3DDC84", color=ft.Colors.BLACK),
                    ),
            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            padding=ft.padding.symmetric(horizontal=28, vertical=18),
            bgcolor="#121620",
            border=ft.border.only(bottom=ft.border.BorderSide(1, "#2D2D2D")),
        )

        return ft.Container(
            content=ft.Column([
                header,
                ft.Container(height=20),
                self.nodes_panel,
            ], spacing=0),
            bgcolor="#0D0D0D",
            expand=True,
        )
