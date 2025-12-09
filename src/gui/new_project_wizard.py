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
            "model_type": "LogisticRegression",
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
        self.steps = [
            "Nombre del Proyecto",
            "Configuración de Tareas",
            "Gestión de Clases",
            "Clases Balanceadas",
            "Cargar Dataset",
            "Entrenamiento y Logs"
        ]

        # UI components
        self.step_indicator = None
        self.content_area = None
        self.next_button = None
        self.back_button = None

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
                                ft.Container(height=8),
                                ft.Text("Pasos", size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                                ft.Container(height=16),
                                self.step_indicator,
                                ft.Container(expand=True),
                                # Footer info
                                ft.Divider(color="#2D2D2D", height=1),
                                ft.Container(height=12),
                                ft.Text(
                                    f"Paso {self.current_step + 1} de {len(self.steps)}",
                                    size=11,
                                    color="#888888"
                                ),
                            ], spacing=0, expand=True),
                            width=300,
                            bgcolor="#1A1A1A",
                            padding=ft.padding.symmetric(horizontal=20, vertical=20),
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
                                    width=44,
                                    height=44,
                                    bgcolor=bg_color,
                                    border_radius=22,
                                ),
                                # Icon or number
                                ft.Container(
                                    content=ft.Icon(ft.Icons.CHECK, size=22, color=ft.Colors.WHITE) if is_completed
                                    else ft.Text(
                                        str(i + 1),
                                        size=18,
                                        weight=ft.FontWeight.BOLD,
                                        color=text_color,
                                    ),
                                    width=44,
                                    height=44,
                                    alignment=ft.alignment.center,
                                ),
                            ], width=44, height=44),
                            width=44,
                            height=44,
                        ),

                        # Step text
                        ft.Container(
                            content=ft.Column([
                                ft.Text(
                                    f"Paso {i + 1}: {step_name}",
                                    size=13,
                                    weight=ft.FontWeight.W_600,
                                    color=text_color,
                                ),
                                ft.Text(
                                    self._get_step_description(i),
                                    size=10,
                                    color=desc_color if is_active else "#666666",
                                    max_lines=2,
                                ),
                            ], spacing=3, tight=True),
                            padding=ft.padding.only(left=12),
                            expand=True,
                        ),
                    ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    padding=ft.padding.symmetric(horizontal=12, vertical=14),
                    border_radius=10,
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
        """Build the main content area for current step."""
        if self.current_step == 0:
            return self._build_project_name_step()
        elif self.current_step == 1:
            return self._build_task_config_step()
        elif self.current_step == 2:
            return self._build_class_management_step()
        elif self.current_step == 3:
            return self._build_balanced_class_step()
        elif self.current_step == 4:
            return self._build_dataset_upload_step()
        elif self.current_step == 5:
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
        # Model type selection
        model_options = [
            ft.dropdown.Option("LogisticRegression", "Logistic Regression"),
            ft.dropdown.Option("RandomForest", "Random Forest"),
            ft.dropdown.Option("XGBoost", "XGBoost"),
            ft.dropdown.Option("PyTorch_CNN", "PyTorch CNN"),
            ft.dropdown.Option("PyTorch_MLP", "PyTorch MLP"),
        ]

        model_dropdown = ft.Dropdown(
            label="Model Type",
            options=model_options,
            value=self.project_data["model_type"],
            on_change=lambda e: self._update_model_type(e.data),
            width=300,
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

    def _update_task_type(self, task_type):
        self.project_data["task_type"] = task_type

    def _update_setting(self, key, value):
        self.project_data["training_settings"][key] = value

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
        """Start model creation process."""
        self.training_progress.value = 0
        self._add_log("🔄 Iniciando proceso de creación de modelo...")
        # In real implementation, this would create the model in background
        self._simulate_model_creation()

    def _simulate_model_creation(self):
        """Simulate model creation process."""
        import time
        
        steps = [
            (20, "📁 Creando estructura de carpetas del proyecto..."),
            (40, "⚙️ Configurando parámetros del modelo..."),
            (60, "📄 Generando archivos de configuración..."),
            (80, "🔧 Inicializando parámetros de entrenamiento..."),
            (100, "✅ Configuración de modelo creada exitosamente!"),
        ]
        
        for progress, message in steps:
            time.sleep(0.4)  # Simulate creation time
            self.training_progress.value = progress / 100
            self._add_log(message)
            self.page.update()
        
        # Final success message
        self._add_log("📊 Modelo listo para entrenamiento")

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
        """Simulate training progress."""
        import time
        for i in range(101):
            time.sleep(0.05)  # Simulate training time
            self.training_progress.value = i / 100
            if i % 10 == 0:
                self._add_log(f"Epoch {i//10}/10 completed - Loss: {1.0 - i/200:.3f}")
            self.page.update()

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