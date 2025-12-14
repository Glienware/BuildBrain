"""Minimal NodesPanel placeholder used by the Main Window.

This file was emptied previously; restore a lightweight `NodesPanel`
so the main window can import and render the 'Nodos' tab. It's
designed to be replaced with the full implementation later.
"""

import flet as ft
from typing import Optional


class NodesPanel(ft.Container):
    def __init__(self, dataset_path: Optional[str] = None):
        content = ft.Column([
            ft.Row([
                ft.Text("Nodes Panel (placeholder)", size=16, weight=ft.FontWeight.W_700),
            ]),
            ft.Divider(height=8),
            ft.Text("The visual automation workspace has been moved to the Agent Builder.")
        ], spacing=12)
        super().__init__(content=content, expand=True, padding=ft.padding.all(12), bgcolor="#0b0f14")


if __name__ == "__main__":
    def _main(page: ft.Page):
        page.title = "NodesPanel Debug"
        page.add(NodesPanel())

    ft.app(target=_main)
