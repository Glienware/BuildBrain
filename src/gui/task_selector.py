"""
Task Selector Component

Selects between Classification and Regression tasks.
"""

import flet as ft


class TaskSelector:
    """
    Radio buttons for task type selection.
    """

    TASKS = ["Classification", "Regression"]

    def __init__(self, page: ft.Page):
        self.page = page
        self.task_type = ft.RadioGroup(
            content=ft.Row([
                ft.Radio(value=task, label=task) for task in self.TASKS
            ]),
            value=self.TASKS[0]
        )

    def build(self):
        """
        Build the task selector UI.
        """
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.CATEGORY, size=24),
                    ft.Text("Task Type", size=16, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.START),
                self.task_type
            ], spacing=10),
            padding=10
        )