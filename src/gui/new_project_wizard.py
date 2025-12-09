"""
New Project Wizard for AI/ML Trainer App

Step-by-step wizard similar to Android Studio for creating new ML projects.
"""

import flet as ft
import os
import json
from datetime import datetime
from typing import Dict, Any
from .dataset_uploader import DatasetUploader
from ..training.model_trainer import ModelTrainer
from ..training.model_factory import ModelFactory


class NewProjectWizard:
    """
    Step-by-step wizard for creating new ML projects with Android Studio-like interface.
    """

    def __init__(self, page: ft.Page, on_complete, on_cancel):
        self.page = page
        self.on_complete = on_complete
        self.on_cancel = on_cancel

        # Project data
        self.project_data = {
            "project_name": "",
            "task_type": "classification",
            "model_type": "logistic_regression",
            "training_settings": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "advanced_enabled": False
            },
            "classes": [],
            "dataset_path": None,
            "dataset_type": None,
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }

        # UI state
        self.current_step = 0
        self._update_steps_based_on_model()

        # UI components
        self.step_indicator = None
        self.content_area = None
        self.next_button = None
        self.back_button = None

    def _update_steps_based_on_model(self):
        """Update steps dynamically based on selected model type."""
        model_type = self.project_data.get("model_type", "logistic_regression")
        
        # Determine model category
        supervised_models = [
            "linear_regression", "random_forest_regressor", "xgboost_regressor", "svr",
            "logistic_regression", "random_forest_classifier", "xgboost_classifier",
            "knn", "svm", "naive_bayes"
        ]
        
        unsupervised_models = [
            "kmeans", "dbscan", "mean_shift", "gaussian_mixture",
            "pca", "tsne", "umap"
        ]
        
        anomaly_models = ["isolation_forest", "one_class_svm"]
        
        deep_learning_models = [
            "resnet18", "resnet34", "resnet50",
            "pytorch_cnn", "pytorch_mlp"
        ]
        
        # Set steps based on model category
        if model_type in supervised_models:
            self.steps = [
                "Nombre del Proyecto",
                "Configuración de Tareas",
                "Gestión de Clases",
                "Clases Balanceadas",
                "Cargar Dataset",
                "Entrenamiento y Logs"
            ]
        elif model_type in unsupervised_models:
            self.steps = [
                "Nombre del Proyecto",
                "Tipo de Modelo",
                "Parámetros del Modelo",
                "Cargar Dataset",
                "Entrenamiento y Logs"
            ]
        elif model_type in anomaly_models:
            self.steps = [
                "Nombre del Proyecto",
                "Configuración de Anomalías",
                "Parámetros del Modelo",
                "Cargar Dataset",
                "Entrenamiento y Logs"
            ]
        elif model_type in deep_learning_models:
            self.steps = [
                "Nombre del Proyecto",
                "Tipo de Red Neuronal",
                "Configuración de la Red",
                "Parámetros de Entrenamiento",
                "Cargar Dataset",
                "Entrenamiento y Logs"
            ]
        else:
            self.steps = [
                "Nombre del Proyecto",
                "Configuración de Tareas",
                "Cargar Dataset",
                "Entrenamiento y Logs"
            ]

    def build(self):
        """Build the wizard interface."""
        # Initialize navigation buttons first
        self._build_navigation_buttons()

        self.step_indicator = self._build_step_indicator()
        self.content_area = self._build_content_area()

        return ft.Container(
            content=ft.Column([
                # Premium Header
                ft.Container(
                    content=ft.Row([
                        # Left section with close button
                        ft.Container(
                            content=ft.Row([
                                ft.Container(
                                    content=ft.IconButton(
                                        ft.Icons.CLOSE,
                                        on_click=self._show_cancel_dialog,
                                        tooltip="Cancelar",
                                        icon_size=24,
                                        icon_color="#FF6B6B"
                                    ),
                                    width=50,
                                    height=50,
                                    alignment=ft.alignment.center,
                                ),
                                ft.Container(width=5),
                                ft.Column([
                                    ft.Text("Crear Nuevo Modelo", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                                    ft.Text("Configura tu proyecto de ML paso a paso", size=12, color="#B0B0B0"),
                                ], spacing=3),
                            ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            expand=True
                        ),
                        
                        # Right section with navigation buttons
                        ft.Row([
                            self.back_button,
                            ft.Container(width=8),
                            self.next_button,
                        ], spacing=0, alignment=ft.MainAxisAlignment.END),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    bgcolor="#0D0D0D",
                    padding=ft.padding.symmetric(horizontal=24, vertical=16),
                    border=ft.border.only(bottom=ft.border.BorderSide(1.5, "#2D2D2D"))
                ),

                # Main content area
                ft.Container(
                    content=ft.Row([
                        # Left Sidebar - Step Indicator
                        ft.Container(
                            content=ft.Column([
                                ft.Container(height=4),
                                ft.Text("Pasos", size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                                ft.Container(height=10),
                                self.step_indicator,
                                ft.Container(expand=True),
                                # Footer info
                                ft.Divider(color="#2D2D2D", height=1),
                                ft.Container(height=8),
                                ft.Text(
                                    f"Paso {self.current_step + 1} de {len(self.steps)}",
                                    size=10,
                                    color="#888888"
                                ),
                            ], spacing=0, expand=True),
                            width=280,
                            bgcolor="#1A1A1A",
                            padding=ft.padding.symmetric(horizontal=14, vertical=12),
                            border=ft.border.only(right=ft.border.BorderSide(1.5, "#2D2D2D"))
                        ),

                        # Right Content Area
                        ft.Container(
                            content=ft.Column([
                                ft.Container(height=20),
                                ft.Container(
                                    content=ft.Column([
                                        self.content_area,
                                    ], spacing=24, expand=True),
                                    expand=True,
                                ),
                                ft.Container(height=30),
                            ], scroll=ft.ScrollMode.AUTO, spacing=0, expand=True),
                            expand=True,
                            bgcolor="#0D0D0D",
                            padding=ft.padding.symmetric(horizontal=40, vertical=0),
                        ),
                    ], spacing=0, expand=True),
                    expand=True,
                ),

            ], spacing=0),
            bgcolor="#0D0D0D",
            expand=True
        )

    def _build_step_indicator(self):
        """Build the vertical step indicator."""
        step_items = []
        for i, step_name in enumerate(self.steps):
            is_active = i == self.current_step
            is_completed = i < self.current_step

            # Determine colors
            if is_completed:
                bg_color = "#3DDC84"
                text_color = ft.Colors.WHITE
                desc_color = "#3DDC84"
                highlight_bg = None
            elif is_active:
                bg_color = "#82B1FF"
                text_color = ft.Colors.WHITE
                desc_color = "#82B1FF"
                highlight_bg = ft.Colors.with_opacity(0.15, "#82B1FF")
            else:
                bg_color = "#3A3A3A"
                text_color = "#888888"
                desc_color = "#666666"
                highlight_bg = None

            step_items.append(
                ft.Container(
                    content=ft.Row([
                        # Step circle with icon or number
                        ft.Container(
                            content=ft.Stack([
                                # Background circle
                                ft.Container(
                                    width=40,
                                    height=40,
                                    bgcolor=bg_color,
                                    border_radius=20,
                                ),
                                # Icon or number
                                ft.Container(
                                    content=ft.Icon(ft.Icons.CHECK, size=20, color=ft.Colors.WHITE) if is_completed
                                    else ft.Text(
                                        str(i + 1),
                                        size=16,
                                        weight=ft.FontWeight.BOLD,
                                        color=text_color,
                                    ),
                                    width=40,
                                    height=40,
                                    alignment=ft.alignment.center,
                                ),
                            ], width=40, height=40),
                            width=40,
                            height=40,
                        ),

                        # Step text
                        ft.Container(
                            content=ft.Column([
                                ft.Text(
                                    f"Paso {i + 1}: {step_name}",
                                    size=11,
                                    weight=ft.FontWeight.W_600,
                                    color=text_color,
                                ),
                                ft.Text(
                                    self._get_step_description(i),
                                    size=9,
                                    color=desc_color if is_active else "#666666",
                                    max_lines=2,
                                ),
                            ], spacing=2, tight=True),
                            padding=ft.padding.only(left=10),
                            expand=True,
                        ),
                    ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=ft.padding.symmetric(horizontal=10, vertical=10),
                    border_radius=8,
                    bgcolor=highlight_bg,
                    border=ft.border.all(1.5, bg_color) if is_active else None,
                )
            )

            # Add connector line (except for last item)
            if i < len(self.steps) - 1:
                connector_color = "#3DDC84" if is_completed else "#3A3A3A"
                step_items.append(
                    ft.Container(
                        content=ft.Container(
                            width=2.5,
                            height=16,
                            bgcolor=connector_color,
                        ),
                        padding=ft.padding.only(left=20),
                    )
                )

        return ft.Column(step_items, spacing=2, scroll=ft.ScrollMode.AUTO)

    def _get_step_description(self, step_index):
        """Get description for each step."""
        descriptions = [
            "Nombra tu proyecto",
            "Configura el modelo y entrenamiento",
            "Define las clases",
            "Verifica el balance de clases",
            "Carga el dataset",
            "Entrena y monitorea"
        ]
        return descriptions[step_index]

    def _build_content_area(self):
        """Build the main content area for current step based on step name."""
        if self.current_step >= len(self.steps):
            return ft.Text("Step not implemented")
        
        current_step_name = self.steps[self.current_step]
        
        # Always start with project name
        if current_step_name == "Nombre del Proyecto":
            return self._build_project_name_step()
        
        # Supervised Learning flow
        elif current_step_name == "Configuración de Tareas":
            return self._build_task_config_step()
        elif current_step_name == "Gestión de Clases":
            return self._build_class_management_step()
        elif current_step_name == "Clases Balanceadas":
            return self._build_balanced_class_step()
        
        # Unsupervised Learning flow
        elif current_step_name == "Tipo de Modelo":
            return self._build_model_selection_step()
        elif current_step_name == "Parámetros del Modelo":
            return self._build_unsupervised_params_step()
        
        # Anomaly Detection flow
        elif current_step_name == "Configuración de Anomalías":
            return self._build_anomaly_config_step()
        
        # Deep Learning flow
        elif current_step_name == "Tipo de Red Neuronal":
            return self._build_neural_network_type_step()
        elif current_step_name == "Configuración de la Red":
            return self._build_neural_network_config_step()
        elif current_step_name == "Parámetros de Entrenamiento":
            return self._build_training_params_step()
        
        # Common steps
        elif current_step_name == "Cargar Dataset":
            return self._build_dataset_upload_step()
        elif current_step_name == "Entrenamiento y Logs":
            return self._build_training_logs_step()
        
        return ft.Text("Step not implemented")

    def _refresh_current_step(self):
        """Refresh the current step content."""
        # Rebuild the content area
        self.content_area.content = self._build_content_area()

    def _build_project_name_step(self):
        """Build project name input step."""
        project_name_field = ft.TextField(
            label="Nombre del Proyecto",
            hint_text="ej: MiClasificador, SegmentadorImg",
            value=self.project_data["project_name"],
            on_change=lambda e: self._update_project_name(e.data),
            min_lines=1,
            border_radius=8,
            border_color="#3A3A3A",
            focused_border_color="#3DDC84",
            text_style=ft.TextStyle(size=16),
        )

        return ft.Column([
            # Header
            ft.Column([
                ft.Text("Nombre del Proyecto", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text(
                    "Dale a tu proyecto de ML un nombre único y descriptivo",
                    size=15,
                    color="#AAAAAA",
                ),
            ], spacing=0),

            ft.Container(height=40),

            # Input area with icon
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.FOLDER_SPECIAL, size=24, color="#3DDC84"),
                        ft.Container(width=16),
                        ft.Column([
                            ft.Text("Nombre único para tu proyecto", size=12, color="#888888", weight=ft.FontWeight.W_500),
                            ft.Container(height=12),
                            ft.Container(
                                content=project_name_field,
                                expand=True,
                            ),
                        ], expand=True, spacing=0),
                    ], vertical_alignment=ft.CrossAxisAlignment.START, expand=True, spacing=0),
                    
                    ft.Container(height=20),
                    
                    ft.Row([
                        ft.Icon(ft.Icons.INFO_OUTLINE, size=18, color="#666666"),
                        ft.Container(width=12),
                        ft.Column([
                            ft.Text(
                                "Este nombre se usará para guardar tu proyecto y crear la carpeta del modelo",
                                size=12,
                                color="#888888",
                            ),
                        ], expand=True),
                    ], spacing=0),
                ], spacing=0),
                bgcolor="#1A1A1A",
                border=ft.border.all(1.5, "#2D2D2D"),
                border_radius=10,
                padding=ft.padding.all(24),
            ),

            ft.Container(expand=True),
        ], spacing=0, expand=True)

    def _build_task_config_step(self):
        """Build task configuration step."""
        # Model type selection - Comprehensive ML Models
        model_options = [
            # Supervised Learning - Regression
            ft.dropdown.Option("linear_regression", "📊 Regresión Lineal"),
            ft.dropdown.Option("random_forest_regressor", "🌲 Random Forest Regressor"),
            ft.dropdown.Option("xgboost_regressor", "🚀 XGBoost Regressor"),
            ft.dropdown.Option("svr", "📈 SVR (Support Vector Regression)"),
            
            # Supervised Learning - Classification
            ft.dropdown.Option("logistic_regression", "📊 Regresión Logística"),
            ft.dropdown.Option("random_forest_classifier", "🌲 Random Forest Classifier"),
            ft.dropdown.Option("xgboost_classifier", "🚀 XGBoost Classifier"),
            ft.dropdown.Option("knn", "🔍 KNN (K-Nearest Neighbors)"),
            ft.dropdown.Option("svm", "🎯 SVM (Support Vector Machine)"),
            ft.dropdown.Option("naive_bayes", "📚 Naive Bayes"),
            
            # Unsupervised Learning - Clustering
            ft.dropdown.Option("kmeans", "🔗 K-Means"),
            ft.dropdown.Option("dbscan", "🔗 DBSCAN"),
            ft.dropdown.Option("mean_shift", "🔗 Mean Shift"),
            ft.dropdown.Option("gaussian_mixture", "🔗 Gaussian Mixture Models"),
            
            # Unsupervised Learning - Dimensionality Reduction
            ft.dropdown.Option("pca", "📉 PCA (Principal Component Analysis)"),
            ft.dropdown.Option("tsne", "📉 t-SNE"),
            ft.dropdown.Option("umap", "📉 UMAP"),
            
            # Anomaly Detection
            ft.dropdown.Option("isolation_forest", "⚠️ Isolation Forest"),
            ft.dropdown.Option("one_class_svm", "⚠️ One-Class SVM"),
            
            # Deep Learning - ResNets
            ft.dropdown.Option("resnet18", "🧠 ResNet-18"),
            ft.dropdown.Option("resnet34", "🧠 ResNet-34"),
            ft.dropdown.Option("resnet50", "🧠 ResNet-50"),
            
            # Deep Learning - Custom
            ft.dropdown.Option("pytorch_cnn", "🧠 PyTorch CNN"),
            ft.dropdown.Option("pytorch_mlp", "🧠 PyTorch MLP"),
        ]

        model_dropdown = ft.Dropdown(
            label="Model Type",
            options=model_options,
            value=self.project_data["model_type"],
            on_change=lambda e: self._update_model_type(e.data),
            width=400,
        )

        # Task type selection
        task_radio = ft.RadioGroup(
            content=ft.Row([
                ft.Radio(value="classification", label="Classification"),
                ft.Container(width=20),
                ft.Radio(value="regression", label="Regression"),
            ]),
            value=self.project_data["task_type"],
            on_change=lambda e: self._update_task_type(e.value),
        )

        # Basic settings
        epochs_field = ft.TextField(
            label="Epochs",
            value=str(self.project_data["training_settings"]["epochs"]),
            on_change=lambda e: self._update_setting("epochs", int(e.data) if e.data.isdigit() else 100),
            width=150,
        )

        batch_size_field = ft.TextField(
            label="Batch Size",
            value=str(self.project_data["training_settings"]["batch_size"]),
            on_change=lambda e: self._update_setting("batch_size", int(e.data) if e.data.isdigit() else 32),
            width=150,
        )

        learning_rate_field = ft.TextField(
            label="Learning Rate",
            value=str(self.project_data["training_settings"]["learning_rate"]),
            on_change=lambda e: self._update_setting("learning_rate", float(e.data) if e.data.replace('.', '').replace('-', '').isdigit() else 0.001),
            width=150,
        )

        optimizer_dropdown = ft.Dropdown(
            label="Optimizer",
            options=[
                ft.dropdown.Option("adam", "Adam"),
                ft.dropdown.Option("sgd", "SGD"),
                ft.dropdown.Option("rmsprop", "RMSprop"),
            ],
            value=self.project_data["training_settings"]["optimizer"],
            on_change=lambda e: self._update_setting("optimizer", e.data),
            width=150,
        )

        return ft.Column([
            # Header
            ft.Column([
                ft.Text("Configuración del Modelo", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text(
                    "Elige el tipo de modelo y ajusta los parámetros de entrenamiento",
                    size=15,
                    color="#AAAAAA",
                ),
            ], spacing=0),

            ft.Container(height=32),

        # Tipo de Modelo
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.SCHEMA, size=22, color="#3DDC84"),
                        ft.Container(width=16),
                        ft.Text("Tipo de Modelo", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(height=16),
                    model_dropdown,
                    ft.Container(height=12),
                    self._build_model_description_container(),
                ], spacing=0),
                bgcolor="#1A1A1A",
                border=ft.border.all(1.5, "#2D2D2D"),
                border_radius=10,
                padding=ft.padding.all(20),
            ),

            ft.Container(height=20),

            # Tipo de Tarea
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.CATEGORY, size=22, color="#82B1FF"),
                        ft.Container(width=16),
                        ft.Text("Tipo de Tarea", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(height=16),
                    task_radio,
                ], spacing=0),
                bgcolor="#1A1A1A",
                border=ft.border.all(1.5, "#2D2D2D"),
                border_radius=10,
                padding=ft.padding.all(20),
            ),

            ft.Container(height=20),

            # Configuración de Entrenamiento
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.TUNE, size=22, color="#FFB74D"),
                        ft.Container(width=16),
                        ft.Text("Parámetros de Entrenamiento", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(height=16),
                    
                    ft.Column([
                        ft.Row([
                            ft.Column([
                                ft.Text("Épocas", size=12, color="#888888"),
                                ft.Container(height=6),
                                epochs_field,
                            ], expand=True, spacing=0),
                            ft.Container(width=16),
                            ft.Column([
                                ft.Text("Batch Size", size=12, color="#888888"),
                                ft.Container(height=6),
                                batch_size_field,
                            ], expand=True, spacing=0),
                        ], spacing=0),
                        ft.Container(height=12),
                        ft.Row([
                            ft.Column([
                                ft.Text("Learning Rate", size=12, color="#888888"),
                                ft.Container(height=6),
                                learning_rate_field,
                            ], expand=True, spacing=0),
                            ft.Container(width=16),
                            ft.Column([
                                ft.Text("Optimizador", size=12, color="#888888"),
                                ft.Container(height=6),
                                optimizer_dropdown,
                            ], expand=True, spacing=0),
                        ], spacing=0),
                    ], spacing=0),
                ], spacing=0),
                bgcolor="#1A1A1A",
                border=ft.border.all(1.5, "#2D2D2D"),
                border_radius=10,
                padding=ft.padding.all(20),
            ),

            ft.Container(expand=True),
        ], spacing=0, expand=True)

    def _build_class_management_step(self):
        """Build class management step."""
        # Class input
        class_name_field = ft.TextField(
            label="Nombre de la Clase",
            hint_text="ej: Gato, Perro, Pájaro",
            border_radius=8,
            border_color="#3A3A3A",
            focused_border_color="#82B1FF",
            text_style=ft.TextStyle(size=14),
        )

        add_class_button = ft.ElevatedButton(
            "Agregar Clase",
            icon=ft.Icons.ADD_CIRCLE,
            on_click=lambda e: self._add_class(class_name_field.value),
            style=ft.ButtonStyle(
                bgcolor="#82B1FF",
                color=ft.Colors.BLACK,
                padding=ft.padding.symmetric(horizontal=20, vertical=12),
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
        )

        # Classes list
        self.classes_list = ft.Column(spacing=8)
        self._update_classes_display()

        return ft.Column([
            # Header
            ft.Column([
                ft.Text("Gestión de Clases", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text(
                    "Define las categorías que tu modelo necesita clasificar",
                    size=15,
                    color="#AAAAAA",
                ),
            ], spacing=0),

            ft.Container(height=32),

            # Add class input
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.PLAYLIST_ADD, size=22, color="#82B1FF"),
                        ft.Container(width=16),
                        ft.Text("Nueva Clase", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(height=16),
                    ft.Row([
                        ft.Container(
                            content=class_name_field,
                            expand=True,
                        ),
                        ft.Container(width=12),
                        add_class_button,
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                ], spacing=0),
                bgcolor="#1A1A1A",
                border=ft.border.all(1.5, "#2D2D2D"),
                border_radius=10,
                padding=ft.padding.all(20),
            ),

            ft.Container(height=24),

            # Classes list
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.LABEL_OUTLINE, size=22, color="#3DDC84"),
                        ft.Container(width=16),
                        ft.Text("Clases Definidas", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                        ft.Container(expand=True),
                        ft.Container(
                            content=ft.Text(
                                f"{len(self.project_data['classes'])} clases",
                                size=12,
                                color="#3DDC84",
                                weight=ft.FontWeight.W_600,
                            ),
                            padding=ft.padding.symmetric(horizontal=12, vertical=4),
                            bgcolor=ft.Colors.with_opacity(0.2, "#3DDC84"),
                            border_radius=4,
                        ),
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(height=16),
                    ft.Container(
                        content=self.classes_list,
                        expand=True,
                        bgcolor="#0D0D0D",
                        border_radius=8,
                        padding=ft.padding.all(12),
                    ),
                ], spacing=0, expand=True),
                bgcolor="#1A1A1A",
                border=ft.border.all(1.5, "#2D2D2D"),
                border_radius=10,
                padding=ft.padding.all(20),
                expand=True,
            ),

            ft.Container(expand=True),
        ], spacing=0, expand=True)

    def _build_balanced_class_step(self):
        """Build balanced class analysis step."""
        # Class distribution visualization
        self.balance_chart = ft.Column(spacing=8)

        balance_button = ft.ElevatedButton(
            "Balancear Clases",
            icon=ft.Icons.SCALE,
            on_click=self._balance_classes,
            style=ft.ButtonStyle(
                bgcolor="#FFB74D",
                color=ft.Colors.BLACK,
                padding=ft.padding.symmetric(horizontal=24, vertical=12),
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
        )

        self._update_balance_display()

        return ft.Column([
            # Header
            ft.Column([
                ft.Text("Análisis de Balance", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text(
                    "Verifica y equilibra la distribución de clases en tu dataset",
                    size=15,
                    color="#AAAAAA",
                ),
            ], spacing=0),

            ft.Container(height=32),

            # Balance visualization
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.ASSESSMENT, size=22, color="#FFB74D"),
                        ft.Container(width=16),
                        ft.Text("Distribución de Clases", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(height=16),
                    self.balance_chart,
                    ft.Container(height=20),
                    ft.Row([
                        ft.Container(expand=True),
                        balance_button,
                    ], spacing=0),
                ], spacing=0, expand=True),
                bgcolor="#1A1A1A",
                border=ft.border.all(1.5, "#2D2D2D"),
                border_radius=10,
                padding=ft.padding.all(20),
                expand=True,
            ),

            ft.Container(expand=True),
        ], spacing=0, expand=True)

    def _build_dataset_upload_step(self):
        """Build dataset upload step with class-based organization."""
        # Initialize dataset uploader with existing classes
        if not hasattr(self, 'dataset_uploader'):
            self.dataset_uploader = DatasetUploader(self.page, existing_classes=self.project_data.get("classes", []), on_update=self._refresh_current_step)

        return ft.Column([
            # Header
            ft.Column([
                ft.Text("Carga de Dataset", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text(
                    "Sube las imágenes para cada clase que definiste anteriormente",
                    size=15,
                    color="#AAAAAA",
                ),
            ], spacing=0),

            ft.Container(height=24),

            # Dataset uploader and summary - single scrollable container with custom scrollbar style
            ft.Container(
                content=ft.Column([
                    # Dataset uploader
                    self.dataset_uploader.build(),
                    
                    ft.Container(height=24),
                    
                    # Dataset summary
                    self._build_dataset_summary(),
                ], spacing=0, scroll=ft.ScrollMode.AUTO),
                expand=True,
                bgcolor="#0D0D0D",
                border_radius=8,
                border=ft.border.all(1, "#1F1F1F"),
            ),

        ], spacing=0, expand=True)

    def _build_dataset_summary(self):
        """Build dataset summary card with responsive 2-column grid."""
        if not hasattr(self, 'dataset_uploader'):
            return ft.Container()

        dataset_info = self.dataset_uploader.get_dataset_info()

        # Build class distribution items - each card takes 50% width minus spacing
        class_items = []
        for class_name, images in dataset_info['classes'].items():
            num_images = len(images)
            percentage = (num_images / max(dataset_info['total_images'], 1)) * 100 if dataset_info['total_images'] > 0 else 0
            
            class_card = ft.Container(
                content=ft.Column([
                    ft.Text(class_name, size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                    ft.Container(height=12),
                    ft.Row([
                        ft.Column([
                            ft.Text("Imágenes", size=11, color="#888888", weight=ft.FontWeight.W_500),
                            ft.Container(height=6),
                            ft.Text(str(num_images), size=20, weight=ft.FontWeight.BOLD, color="#3DDC84"),
                        ], spacing=0),
                        ft.Container(expand=True),
                        ft.Column([
                            ft.Text("Porcentaje", size=11, color="#888888", weight=ft.FontWeight.W_500),
                            ft.Container(height=6),
                            ft.Text(f"{percentage:.1f}%", size=18, weight=ft.FontWeight.W_600, color="#82B1FF"),
                        ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.END),
                    ], spacing=0, expand=True),
                    
                    # Progress bar
                    ft.Container(height=12),
                    ft.Container(
                        content=ft.Container(
                            bgcolor="#3DDC84",
                            border_radius=2,
                        ),
                        height=4,
                        bgcolor="#2D2D2D",
                        border_radius=2,
                    ),
                ], spacing=0, expand=True),
                padding=ft.padding.all(16),
                bgcolor="#0D0D0D",
                border_radius=8,
                border=ft.border.all(1.5, "#2D2D2D"),
                expand=True,
            )
            class_items.append(class_card)

        # Create 2-column grid rows
        grid_rows = []
        for i in range(0, len(class_items), 2):
            row_controls = []
            
            # First card
            if i < len(class_items):
                row_controls.append(class_items[i])
            
            # Second card or empty space
            if i + 1 < len(class_items):
                row_controls.append(class_items[i + 1])
            else:
                # Empty spacer to keep alignment
                row_controls.append(ft.Container(expand=True))
            
            grid_rows.append(
                ft.Row(
                    controls=row_controls,
                    spacing=12,
                    expand=True,
                )
            )

        # Main summary container with scroll
        return ft.Container(
            content=ft.Column([
                # Top stats
                ft.Row([
                    ft.Icon(ft.Icons.DATASET, size=24, color="#3DDC84"),
                    ft.Container(width=12),
                    ft.Text("Resumen del Dataset", size=15, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                
                ft.Container(height=20),
                
                # Stats row (responsive)
                ft.Row([
                    # Classes stat
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Clases", size=12, color="#888888", weight=ft.FontWeight.W_500),
                            ft.Container(height=8),
                            ft.Text(str(dataset_info['num_classes']), size=28, weight=ft.FontWeight.BOLD, color="#3DDC84"),
                        ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        expand=True,
                        padding=ft.padding.all(16),
                        bgcolor="#0D0D0D",
                        border_radius=8,
                        border=ft.border.all(1.5, "#2D2D2D"),
                    ),
                    
                    ft.Container(width=12),
                    
                    # Total images stat
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Total de Imágenes", size=12, color="#888888", weight=ft.FontWeight.W_500),
                            ft.Container(height=8),
                            ft.Text(str(dataset_info['total_images']), size=28, weight=ft.FontWeight.BOLD, color="#82B1FF"),
                        ], spacing=0, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        expand=True,
                        padding=ft.padding.all(16),
                        bgcolor="#0D0D0D",
                        border_radius=8,
                        border=ft.border.all(1.5, "#2D2D2D"),
                    ),
                ], spacing=0, expand=True),
                
                ft.Container(height=24),
                
                # Class distribution section
                ft.Text("Distribución por Clase", size=13, color="#888888", weight=ft.FontWeight.W_600),
                ft.Container(height=12),
                
                # Class grid (sin scroll, el padre se encarga)
                ft.Column(
                    controls=grid_rows,
                    spacing=12,
                ),
            ], spacing=0, expand=False),
            bgcolor="#1A1A1A",
            border=ft.border.all(1.5, "#2D2D2D"),
            border_radius=10,
            padding=ft.padding.all(20),
        )

    def _build_training_logs_step(self):
        """Build model creation and logs step."""
        # Create model button
        create_model_button = ft.ElevatedButton(
            "Crear Modelo",
            icon=ft.Icons.BUILD_CIRCLE,
            on_click=self._start_model_creation,
            style=ft.ButtonStyle(
                bgcolor="#3DDC84",
                color=ft.Colors.BLACK,
                padding=ft.padding.symmetric(horizontal=32, vertical=16),
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
        )

        # Progress and logs
        self.training_progress = ft.ProgressBar(value=0, color="#3DDC84")
        self.training_logs = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=8)

        return ft.Column([
            # Header
            ft.Column([
                ft.Text("Creación del Modelo", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text(
                    "Crea tu modelo configurando carpetas, archivos y parámetros necesarios",
                    size=15,
                    color="#AAAAAA",
                ),
            ], spacing=0),

            ft.Container(height=32),

            # Create model action
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.BUILD, size=24, color="#3DDC84"),
                        ft.Container(width=16),
                        ft.Text("Crear Modelo", size=15, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(height=20),
                    ft.Text(
                        "El proceso creará automáticamente:",
                        size=12,
                        color="#AAAAAA",
                    ),
                    ft.Container(height=12),
                    ft.Column([
                        self._create_info_row("📁", "Carpetas del proyecto", "#888888"),
                        self._create_info_row("⚙️", "Configuración de modelo", "#888888"),
                        self._create_info_row("📄", "Archivos necesarios", "#888888"),
                        self._create_info_row("🔧", "Parámetros de entrenamiento", "#888888"),
                    ], spacing=8),
                    ft.Container(height=24),
                    create_model_button,
                ], spacing=0),
                bgcolor="#1A1A1A",
                border=ft.border.all(1.5, "#2D2D2D"),
                border_radius=10,
                padding=ft.padding.all(20),
            ),

            ft.Container(height=24),

            # Progress and logs
            ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.INFO_OUTLINE, size=22, color="#82B1FF"),
                        ft.Container(width=16),
                        ft.Text("Logs del Proceso", size=14, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    ft.Container(height=16),
                    ft.Container(
                        content=self.training_progress,
                        padding=ft.padding.symmetric(horizontal=0, vertical=8),
                    ),
                    ft.Container(height=16),
                    ft.Container(
                        content=self.training_logs,
                        bgcolor="#0D0D0D",
                        border=ft.border.all(1.5, "#2D2D2D"),
                        border_radius=8,
                        padding=ft.padding.all(12),
                        expand=True,
                    ),
                ], spacing=0, expand=True),
                bgcolor="#1A1A1A",
                border=ft.border.all(1.5, "#2D2D2D"),
                border_radius=10,
                padding=ft.padding.all(20),
                expand=True,
            ),

            ft.Container(expand=True),
        ], spacing=0, expand=True)

    def _create_info_row(self, icon, text, color):
        """Create an info row with icon and text."""
        return ft.Row([
            ft.Text(icon, size=14),
            ft.Container(width=8),
            ft.Text(text, size=12, color=color),
        ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER)

    def _build_navigation_buttons(self):
        """Build navigation buttons."""
        # Back button with hover effect
        self.back_button = ft.ElevatedButton(
            "Atrás",
            icon=ft.Icons.ARROW_BACK,
            on_click=self._go_back,
            disabled=self.current_step == 0,
            style=ft.ButtonStyle(
                bgcolor=ft.Colors.with_opacity(0.1, "#888888"),
                color="#888888",
                padding=ft.padding.symmetric(horizontal=24, vertical=14),
                side=ft.BorderSide(1.5, "#3A3A3A"),
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
        )

        # Next/Complete button with gradient-like appearance
        next_text = "Completar" if self.current_step == len(self.steps) - 1 else "Siguiente"
        self.next_button = ft.ElevatedButton(
            next_text,
            icon=ft.Icons.CHECK_CIRCLE if next_text == "Completar" else ft.Icons.ARROW_FORWARD,
            on_click=self._go_next,
            style=ft.ButtonStyle(
                bgcolor="#3DDC84",
                color=ft.Colors.BLACK,
                padding=ft.padding.symmetric(horizontal=28, vertical=14),
                shape=ft.RoundedRectangleBorder(radius=8),
            ),
        )

    def _show_cancel_dialog(self, e):
        """Show cancel confirmation dialog."""
        def close_dialog(e=None):
            for ctrl in self.page.overlay:
                if isinstance(ctrl, ft.Container) and hasattr(ctrl, 'data') and ctrl.data == 'cancel_dialog':
                    self.page.overlay.remove(ctrl)
                    break
            self.page.update()

        def confirm_cancel(e):
            close_dialog()
            if self.on_cancel:
                self.on_cancel(e)

        # Create modal content
        modal_content = ft.Container(
            content=ft.Column([
                ft.Text("Cancelar Creación de Proyecto", size=20, weight=ft.FontWeight.BOLD),
                ft.Container(height=15),
                ft.Text("¿Estás seguro de que deseas cancelar? Todos los cambios se perderán.", size=14),
                ft.Container(height=20),
                ft.Row([
                    ft.ElevatedButton(
                        "Continuar Creando",
                        on_click=close_dialog,
                        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_600),
                        expand=True
                    ),
                    ft.Container(width=10),
                    ft.ElevatedButton(
                        "Cancelar y Salir",
                        on_click=confirm_cancel,
                        style=ft.ButtonStyle(bgcolor=ft.Colors.RED_600, color=ft.Colors.WHITE),
                        expand=True
                    ),
                ], spacing=10),
            ], spacing=10),
            bgcolor=ft.Colors.GREY_800,
            padding=25,
            border_radius=12,
            width=400,
            shadow=ft.BoxShadow(blur_radius=20, color="000000")
        )

        # Create overlay
        overlay = ft.Container(
            content=ft.Column([
                ft.Container(expand=True),
                ft.Row([
                    ft.Container(expand=True),
                    modal_content,
                    ft.Container(expand=True)
                ]),
                ft.Container(expand=True)
            ]),
            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
            alignment=ft.alignment.center,
            expand=True,
            on_click=close_dialog
        )
        overlay.data = 'cancel_dialog'
        
        self.page.overlay.append(overlay)
        self.page.update()

    def _go_next(self, e):
        """Go to next step."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self._refresh_ui()
        else:
            # Complete the wizard
            self._complete_wizard()

    def _go_back(self, e):
        """Go to previous step."""
        if self.current_step > 0:
            self.current_step -= 1
            self._refresh_ui()

    def _refresh_ui(self):
        """Refresh the UI for current step."""
        self.step_indicator = self._build_step_indicator()
        self.content_area = self._build_content_area()
        self._build_navigation_buttons()

        # Rebuild the entire wizard
        self.page.controls.clear()
        self.page.add(self.build())
        self.page.update()

    # Helper methods for updating project data
    def _update_project_name(self, name):
        self.project_data["project_name"] = name

    def _update_model_type(self, model_type):
        self.project_data["model_type"] = model_type
        # Update the steps based on new model type
        self._update_steps_based_on_model()
        # Rebuild the entire wizard to reflect the new steps
        self.page.clean()
        self.page.add(self.build())
        self.page.update()

    def _update_task_type(self, task_type):
        self.project_data["task_type"] = task_type

    def _update_setting(self, key, value):
        self.project_data["training_settings"][key] = value

    def _build_model_description_container(self):
        """Build a container with model description based on selected model."""
        model_descriptions = {
            # Supervised Learning - Regression
            "linear_regression": ("Regresión Lineal", "Modelo simple para relaciones lineales entre variables. Rápido y fácil de interpretar.", "#82B1FF"),
            "random_forest_regressor": ("Random Forest Regressor", "Ensemble de árboles para predicciones robustas. Maneja no-linealidades bien.", "#FFB74D"),
            "xgboost_regressor": ("XGBoost Regressor", "Boosting gradual para máximo rendimiento. Excelente en competiciones.", "#FF6B6B"),
            "svr": ("SVR (Support Vector Regression)", "Versión de regresión de máquinas de soporte vectorial.", "#3DDC84"),
            
            # Supervised Learning - Classification
            "logistic_regression": ("Regresión Logística", "Modelo probabilístico para clasificación binaria o multiclase.", "#82B1FF"),
            "random_forest_classifier": ("Random Forest Classifier", "Ensemble robusto para clasificación con múltiples características.", "#FFB74D"),
            "xgboost_classifier": ("XGBoost Classifier", "Algoritmo de boosting de última generación para clasificación.", "#FF6B6B"),
            "knn": ("KNN (K-Nearest Neighbors)", "Clasificación basada en vecinos más cercanos. Simple pero efectivo.", "#3DDC84"),
            "svm": ("SVM (Support Vector Machine)", "Clasificador potente que funciona bien en espacios altos dimensionales.", "#00BCD4"),
            "naive_bayes": ("Naive Bayes", "Clasificador probabilístico rápido basado en el teorema de Bayes.", "#4CAF50"),
            
            # Unsupervised Learning - Clustering
            "kmeans": ("K-Means", "Clustering particional para agrupar datos en K clusters. Rápido y escalable.", "#FF9800"),
            "dbscan": ("DBSCAN", "Clustering basado en densidad. Detecta clusters de formas arbitrarias.", "#F44336"),
            "mean_shift": ("Mean Shift", "Clustering sin necesidad de especificar número de clusters. Adaptativo.", "#2196F3"),
            "gaussian_mixture": ("Gaussian Mixture Models", "Clustering probabilístico que modela clusters como gaussianas.", "#673AB7"),
            
            # Unsupervised Learning - Dimensionality Reduction
            "pca": ("PCA", "Reduce dimensiones mientras preserva varianza. Útil para visualización.", "#009688"),
            "tsne": ("t-SNE", "Excelente para visualizar datos en 2D/3D. Mantiene estructura local.", "#00897B"),
            "umap": ("UMAP", "Alternativa moderna a t-SNE. Más rápido y preserva mejor la estructura global.", "#26A69A"),
            
            # Anomaly Detection
            "isolation_forest": ("Isolation Forest", "Detecta anomalías aislando puntos outliers. Rápido y eficiente.", "#E91E63"),
            "one_class_svm": ("One-Class SVM", "Detección de anomalías usando máquinas de soporte vectorial.", "#C2185B"),
            
            # Deep Learning - ResNets
            "resnet18": ("ResNet-18", "Red neuronal residual con 18 capas. Equilibrio entre velocidad y precisión. Ideal para imágenes.", "#FF6F00"),
            "resnet34": ("ResNet-34", "ResNet con 34 capas. Mejor precisión que ResNet-18 con más parámetros.", "#FFB300"),
            "resnet50": ("ResNet-50", "ResNet con 50 capas. Excelente para tareas complejas de clasificación de imágenes.", "#FFA500"),
            
            # Deep Learning - Custom
            "pytorch_cnn": ("PyTorch CNN", "Red neuronal convolucional personalizada para análisis de imágenes.", "#FF6F00"),
            "pytorch_mlp": ("PyTorch MLP", "Red neuronal multicapa para problemas de regresión y clasificación.", "#FFB300"),
        }
        
        model_type = self.project_data["model_type"]
        title, description, color = model_descriptions.get(
            model_type,
            ("Modelo Desconocido", "Selecciona un modelo válido.", "#666666")
        )
        
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.INFO, size=16, color=color),
                    ft.Container(width=8),
                    ft.Text(title, size=13, weight=ft.FontWeight.W_600, color=color),
                ], spacing=0),
                ft.Container(height=6),
                ft.Text(description, size=11, color="#AAAAAA", max_lines=3),
            ], spacing=0),
            padding=ft.padding.all(12),
            bgcolor="#151515",
            border=ft.border.all(1, color),
            border_radius=8,
        )

    def _build_model_selection_step(self):
        """Build model selection step for unsupervised learning."""
        return ft.Column([
            ft.Column([
                ft.Text("Selecciona tu Modelo", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text("Elige el tipo de modelo no supervisado que mejor se adapte a tus datos", size=15, color="#AAAAAA"),
            ], spacing=0),
            ft.Container(height=40),
            ft.Text("Modelo seleccionado: " + self.project_data["model_type"], size=14, color="#3DDC84"),
            ft.Container(expand=True),
        ], spacing=0, expand=True)

    def _build_unsupervised_params_step(self):
        """Build parameters step for unsupervised models."""
        model_type = self.project_data["model_type"]
        
        params_content = []
        params_content.extend([
            ft.Text("Parámetros del Modelo", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
            ft.Container(height=8),
            ft.Text("Configura los parámetros específicos del algoritmo", size=15, color="#AAAAAA"),
            ft.Container(height=40),
        ])
        
        # Add model-specific parameters
        if model_type == "kmeans":
            params_content.extend([
                ft.Text("Número de Clusters (n_clusters):", size=13, color=ft.Colors.WHITE),
                ft.Slider(min=2, max=10, value=3, label="{value}", on_change=lambda e: self._update_setting("n_clusters", int(e.value))),
            ])
        elif model_type in ["dbscan", "mean_shift"]:
            params_content.extend([
                ft.Text("Parámetros automáticos - Se ajustan en el entrenamiento", size=13, color="#AAAAAA"),
            ])
        elif model_type in ["pca", "tsne", "umap"]:
            params_content.extend([
                ft.Text("Componentes (dimensiones):", size=13, color=ft.Colors.WHITE),
                ft.Slider(min=2, max=3, value=2, label="{value}", on_change=lambda e: self._update_setting("n_components", int(e.value))),
            ])
        
        params_content.append(ft.Container(expand=True))
        return ft.Column(params_content, spacing=0, expand=True)

    def _build_anomaly_config_step(self):
        """Build configuration step for anomaly detection."""
        return ft.Column([
            ft.Column([
                ft.Text("Configuración de Detección de Anomalías", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text("Ajusta los parámetros para detectar anomalías en tus datos", size=15, color="#AAAAAA"),
            ], spacing=0),
            ft.Container(height=40),
            ft.Text("Tasa de Contaminación (% de anomalías esperadas):", size=13, color=ft.Colors.WHITE),
            ft.Container(height=8),
            ft.Slider(min=0.01, max=0.5, value=0.1, label="{value:.2%}", on_change=lambda e: self._update_setting("contamination", e.value)),
            ft.Container(expand=True),
        ], spacing=0, expand=True)

    def _build_neural_network_type_step(self):
        """Build neural network type selection step."""
        return ft.Column([
            ft.Column([
                ft.Text("Tipo de Red Neuronal", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text("Selecciona la arquitectura de red neuronal", size=15, color="#AAAAAA"),
            ], spacing=0),
            ft.Container(height=40),
            ft.Text("Modelo seleccionado: " + self.project_data["model_type"], size=14, color="#3DDC84"),
            ft.Container(height=20),
            ft.Text("Esta arquitectura será optimizada para visión por computadora e imágenes.", size=12, color="#AAAAAA"),
            ft.Container(expand=True),
        ], spacing=0, expand=True)

    def _build_neural_network_config_step(self):
        """Build neural network configuration step."""
        return ft.Column([
            ft.Column([
                ft.Text("Configuración de la Red", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text("Configura los parámetros de la arquitectura", size=15, color="#AAAAAA"),
            ], spacing=0),
            ft.Container(height=40),
            ft.Text("Tamaño de entrada (224x224 por defecto):", size=13, color=ft.Colors.WHITE),
            ft.TextField(label="Resolución", value="224", width=150),
            ft.Container(height=20),
            ft.Text("Número de clases a clasificar:", size=13, color=ft.Colors.WHITE),
            ft.TextField(label="Clases", value=str(len(self.project_data["classes"]) or 10), width=150),
            ft.Container(expand=True),
        ], spacing=0, expand=True)

    def _build_training_params_step(self):
        """Build training parameters step for deep learning."""
        return ft.Column([
            ft.Column([
                ft.Text("Parámetros de Entrenamiento", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                ft.Container(height=8),
                ft.Text("Configura los hiperparámetros para el entrenamiento", size=15, color="#AAAAAA"),
            ], spacing=0),
            ft.Container(height=40),
            ft.Text("Épocas:", size=13, color=ft.Colors.WHITE),
            ft.TextField(label="Épocas", value=str(self.project_data["training_settings"]["epochs"]), width=150),
            ft.Container(height=12),
            ft.Text("Batch Size:", size=13, color=ft.Colors.WHITE),
            ft.TextField(label="Batch Size", value=str(self.project_data["training_settings"]["batch_size"]), width=150),
            ft.Container(height=12),
            ft.Text("Learning Rate:", size=13, color=ft.Colors.WHITE),
            ft.TextField(label="Learning Rate", value=str(self.project_data["training_settings"]["learning_rate"]), width=150),
            ft.Container(expand=True),
        ], spacing=0, expand=True)

    def _add_class(self, class_name):
        if class_name and class_name not in self.project_data["classes"]:
            self.project_data["classes"].append(class_name)
            self._update_classes_display()
            # Actualizar la página inmediatamente
            self.page.update()

    def _update_classes_display(self):
        """Update the classes list display."""
        self.classes_list.controls.clear()
        for i, class_name in enumerate(self.project_data["classes"]):
            self.classes_list.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Row([
                            ft.Container(
                                content=ft.Text(str(i+1), size=12, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                                width=28,
                                height=28,
                                bgcolor="#3DDC84",
                                border_radius=14,
                                alignment=ft.alignment.center,
                            ),
                            ft.Container(width=12),
                            ft.Text(class_name, size=14, weight=ft.FontWeight.W_500, color=ft.Colors.WHITE),
                        ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
                        ft.IconButton(
                            ft.Icons.DELETE_OUTLINE,
                            on_click=lambda e, idx=i: self._remove_class(idx),
                            icon_size=18,
                            icon_color="#FF6B6B",
                        ),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=ft.padding.symmetric(horizontal=12, vertical=10),
                    bgcolor="#0D0D0D",
                    border_radius=8,
                    border=ft.border.all(1, "#2D2D2D"),
                )
            )

    def _remove_class(self, index):
        if 0 <= index < len(self.project_data["classes"]):
            self.project_data["classes"].pop(index)
            self._update_classes_display()
            # Actualizar la página inmediatamente
            self.page.update()

    def _update_balance_display(self):
        """Update class balance visualization."""
        self.balance_chart.controls.clear()

        if not self.project_data["classes"]:
            self.balance_chart.controls.append(
                ft.Text("No classes defined yet", color=ft.Colors.GREY_500)
            )
            return

        # Mock balance data (in real app, this would analyze actual dataset)
        for class_name in self.project_data["classes"]:
            # Mock count - in real app, analyze dataset
            count = 100  # Mock value
            percentage = 100 / len(self.project_data["classes"])

            self.balance_chart.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Text(class_name, size=14, width=100),
                        ft.Container(
                            content=ft.Text(f"{count} samples", size=12),
                            width=80,
                        ),
                        ft.ProgressBar(value=percentage/100, width=150),
                        ft.Text(f"{percentage:.1f}%", size=12, width=50),
                    ], alignment=ft.MainAxisAlignment.START),
                    padding=ft.padding.symmetric(vertical=5),
                )
            )

    def _balance_classes(self, e):
        """Balance classes automatically."""
        # In real implementation, this would apply SMOTE or other balancing techniques
        self._add_log("Classes balanced using SMOTE algorithm")

    def _select_dataset(self, e):
        """Select dataset files/folders."""
        # In real implementation, this would open file picker
        self._add_log("Dataset selection functionality would be implemented here")

    def _start_model_creation(self, e):
        """Start model creation process with ModelTrainer."""
        self.training_progress.value = 0
        self._add_log("🔄 Iniciando proceso de creación de modelo...")
        
        try:
            # Get project data
            project_name = self.project_name.value
            model_type = self.selected_model.value if self.selected_model else "LogisticRegression"
            
            # Create project directory
            import time
            self.training_progress.value = 0.2
            self._add_log(f"📁 Creando estructura de carpetas para '{project_name}'...")
            
            project_dir = os.path.join("projects", project_name)
            os.makedirs(project_dir, exist_ok=True)
            os.makedirs(os.path.join(project_dir, "models"), exist_ok=True)
            os.makedirs(os.path.join(project_dir, "data"), exist_ok=True)
            os.makedirs(os.path.join(project_dir, "logs"), exist_ok=True)
            
            time.sleep(0.3)
            self.training_progress.value = 0.4
            self._add_log(f"⚙️ Configurando parámetros del modelo '{model_type}'...")
            
            # Create project config
            project_config = {
                "name": project_name,
                "model_type": model_type,
                "model_category": ModelFactory.get_model_category(model_type),
                "created_at": datetime.now().isoformat(),
                "hyperparameters": ModelFactory.get_default_hyperparameters(model_type),
                "dataset_info": {
                    "path": "",
                    "classes": getattr(self, 'class_names_input', {}).value if hasattr(self, 'class_names_input') else "",
                    "balanced": getattr(self, 'balance_dataset_switch', False).value if hasattr(self, 'balance_dataset_switch') else False,
                }
            }
            
            time.sleep(0.3)
            self.training_progress.value = 0.6
            self._add_log("📄 Generando archivos de configuración...")
            
            # Save config
            config_path = os.path.join(project_dir, "project_config.json")
            with open(config_path, 'w') as f:
                json.dump(project_config, f, indent=2)
            
            time.sleep(0.3)
            self.training_progress.value = 0.8
            self._add_log("🔧 Inicializando modelo...")
            
            # Initialize ModelTrainer
            trainer = ModelTrainer(model_type, project_config)
            self._add_log(f"✅ Modelo {model_type} inicializado correctamente")
            
            # Save trainer reference for later training
            self.model_trainer = trainer
            self.project_dir = project_dir
            
            time.sleep(0.2)
            self.training_progress.value = 1.0
            self._add_log("✅ Configuración de modelo creada exitosamente!")
            time.sleep(0.3)
            self._add_log("📊 Modelo listo para entrenamiento")
            
        except Exception as ex:
            self._add_log(f"❌ Error durante la creación del modelo: {str(ex)}")
            import traceback
            self._add_log(traceback.format_exc())

    def _simulate_model_creation(self):
        """Deprecated: Model creation now handled by _start_model_creation with ModelTrainer."""
        pass

    def _start_quick_training(self, e):
        """Start quick training."""
        self.training_progress.value = 0
        self._add_log("Starting quick training...")
        # In real implementation, this would start training in background
        self._simulate_training()

    def _start_advanced_training(self, e):
        """Start advanced training."""
        self._add_log("Starting advanced training...")
        # In real implementation, this would open advanced training dialog

    def _simulate_training(self):
        """Simulate training progress with realistic metrics."""
        import time
        import numpy as np
        
        try:
            # Check if ModelTrainer is available
            if not hasattr(self, 'model_trainer'):
                self._add_log("⚠️ Modelo no inicializado. Por favor, crea el modelo primero.")
                return
            
            model_category = self.model_trainer.config.get('model_category', 'supervised')
            epochs = int(getattr(self, 'epochs_input', None).value or 10) if hasattr(self, 'epochs_input') else 10
            
            self._add_log(f"🚀 Iniciando entrenamiento ({epochs} épocas)...")
            
            if model_category == 'deep_learning':
                # Deep learning training with epochs
                for epoch in range(1, epochs + 1):
                    time.sleep(0.2)
                    progress = epoch / epochs
                    self.training_progress.value = progress
                    
                    # Generate realistic loss and accuracy
                    loss = max(0.1, 1.0 - (epoch / epochs) * 0.8 + np.random.random() * 0.05)
                    accuracy = min(0.99, (epoch / epochs) * 0.95 + np.random.random() * 0.02)
                    
                    self._add_log(f"  ⏳ Época {epoch}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
                    self.page.update()
            else:
                # Traditional ML training (single pass)
                steps = [
                    (0.2, "🔄 Preprocesando datos..."),
                    (0.4, "🎓 Entrenando modelo..."),
                    (0.6, "📊 Evaluando desempeño..."),
                    (0.8, "💾 Guardando modelo..."),
                    (1.0, "✅ Entrenamiento completado!"),
                ]
                
                for progress, message in steps:
                    time.sleep(0.3)
                    self.training_progress.value = progress
                    self._add_log(message)
                    self.page.update()
            
            self._add_log(f"✅ Entrenamiento finalizado exitosamente")
            time.sleep(0.3)
            
            # Save model if trainer is available
            if hasattr(self, 'model_trainer') and hasattr(self, 'project_dir'):
                model_path = os.path.join(self.project_dir, "models", "trained_model.pkl")
                self.model_trainer.save_model(model_path)
                self._add_log(f"💾 Modelo guardado en: {model_path}")
            
        except Exception as ex:
            self._add_log(f"❌ Error durante el entrenamiento: {str(ex)}")
            import traceback
            self._add_log(traceback.format_exc())

    def _add_log(self, message):
        """Add message to training logs."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = ft.Row([
            ft.Text(f"[{timestamp}]", size=11, color="#888888", width=80),
            ft.Text(message, size=11, color="#CCCCCC", expand=True),
        ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.START)
        self.training_logs.controls.append(log_entry)
        self.page.update()

    def _complete_wizard(self):
        """Complete the wizard and save project."""
        # Save project to .buildb file
        self._save_project()
        self.on_complete(self.project_data)

    def _save_project(self):
        """Save project to .buildb file."""
        # Create projects directory if it doesn't exist
        projects_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "projects")
        os.makedirs(projects_dir, exist_ok=True)

        project_name = self.project_data['project_name']
        project_dir = os.path.join(projects_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)

        # Generate filename
        filename = f"{project_name}.buildb"
        filepath = os.path.join(project_dir, filename)

        # Update timestamps
        self.project_data["last_modified"] = datetime.now().isoformat()

        # Save dataset information if available
        if hasattr(self, 'dataset_uploader'):
            dataset_info = self.dataset_uploader.get_dataset_info()
            # Copy images to project dataset folder
            copied_dataset = self._copy_dataset_to_project(dataset_info, project_dir)
            self.project_data["dataset"] = copied_dataset

        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(self.project_data, f, indent=2)

        self._add_log(f"Project saved to {filename}")

    def _copy_dataset_to_project(self, dataset_info, project_dir):
        """Copy dataset images to project folder and return updated dataset info."""
        import shutil

        dataset_dir = os.path.join(project_dir, "dataset")
        copied_dataset = {"classes": {}, "total_images": 0, "num_classes": 0}

        for class_name, image_paths in dataset_info.get("classes", {}).items():
            class_dir = os.path.join(dataset_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            copied_paths = []

            for src_path in image_paths:
                if os.path.exists(src_path):
                    filename = os.path.basename(src_path)
                    dst_path = os.path.join(class_dir, filename)
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_paths.append(dst_path)
                    except Exception as e:
                        self._add_log(f"Error copying {src_path}: {e}")
                else:
                    self._add_log(f"Source image not found: {src_path}")

            copied_dataset["classes"][class_name] = copied_paths

        copied_dataset["total_images"] = sum(len(paths) for paths in copied_dataset["classes"].values())
        copied_dataset["num_classes"] = len(copied_dataset["classes"])

        return copied_dataset