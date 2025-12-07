"""
Model Selector Component

Allows user to select the ML model to train.
"""

import flet as ft


class ModelSelector:
    """
    Dropdown for selecting ML models.
    """

    MODELS = [
        # Traditional ML Models
        "LogisticRegression",
        "RandomForest", 
        "XGBoost",
        # Deep Learning Models
        "ResNet-18",
        "ResNet-50", 
        "ResNet-101",
        "VGG16",
        "VGG19",
        "EfficientNet-B0",
        "EfficientNet-B4",
        "DenseNet-121",
        "MobileNetV2",
        "PyTorch CNN",
        "PyTorch MLP"
    ]

    def __init__(self, page: ft.Page):
        self.page = page
        self.selected_model = ft.Dropdown(
            label="Select Model",
            options=[ft.dropdown.Option(model) for model in self.MODELS],
            value=self.MODELS[0],
            width=300
        )

    def build(self):
        """
        Build the model selector UI.
        """
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.MODEL_TRAINING, size=24),
                    ft.Text("Select Model", size=16, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.START),
                self.selected_model
            ], spacing=10),
            padding=10
        )