"""
Settings Panel Component

Handles presets and manual hyperparameter editing.
"""

import flet as ft


class SettingsPanel:
    """
    Panel for training settings and hyperparameters.
    """

    PRESETS = ["Fast", "Balanced", "Max Performance"]

    def __init__(self, page: ft.Page):
        self.page = page
        self.preset = ft.Dropdown(
            label="Preset",
            options=[ft.dropdown.Option(p) for p in self.PRESETS],
            value=self.PRESETS[0],
            width=200
        )

        # Hyperparameters
        self.learning_rate = ft.TextField(label="Learning Rate", value="0.001", width=150)
        self.batch_size = ft.TextField(label="Batch Size", value="32", width=150)
        self.epochs = ft.TextField(label="Epochs", value="100", width=150)
        self.optimizer = ft.Dropdown(
            label="Optimizer",
            options=[ft.dropdown.Option(opt) for opt in ["Adam", "SGD", "RMSprop"]],
            value="Adam",
            width=150
        )
        self.weight_decay = ft.TextField(label="Weight Decay", value="0.0001", width=150)
        self.max_depth = ft.TextField(label="Max Depth", value="10", width=150)
        self.n_estimators = ft.TextField(label="N Estimators", value="100", width=150)

        self.manual_mode = ft.Switch(label="Manual Hyperparameters", value=False)

    def build(self):
        """
        Build the settings panel UI.
        """
        manual_params = ft.Column([
            ft.Text("Hyperparameters", size=14, weight=ft.FontWeight.BOLD),
            ft.Row([self.learning_rate, self.batch_size]),
            ft.Row([self.epochs, self.optimizer]),
            ft.Row([self.weight_decay, self.max_depth]),
            ft.Row([self.n_estimators])
        ], visible=False, spacing=10)

        def toggle_manual(e):
            manual_params.visible = self.manual_mode.value
            self.page.update()

        self.manual_mode.on_change = toggle_manual

        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.TUNE, size=24),
                    ft.Text("Training Settings", size=16, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.START),
                ft.Row([ft.Text("Preset:"), self.preset]),
                self.manual_mode,
                manual_params
            ], spacing=10),
            padding=10
        )