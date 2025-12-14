"""
Main Window - Android Studio Style Interface

Complete UI implementation with modern dark theme and tabbed interface.
"""

import flet as ft
import os
import time
import json
from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
from ..core.project_config import ProjectConfig
from ..core.dataset_manager import DatasetManager
from ..core.pytorch_trainer import PyTorchTrainer


class AndroidStyleMainWindow:
    """
    Main application window with Android Studio-style dark theme and tabbed interface.
    """

    # Android Studio Color Palette
    PRIMARY_COLOR = "#3DDC84"  # Android Green
    PRIMARY_VARIANT = "#00BFA5"
    SECONDARY_COLOR = "#82B1FF"  # Android Blue
    BACKGROUND_DARK = "#1E1E1E"
    BACKGROUND_LIGHT = "#2D2D2D"
    SURFACE_COLOR = "#3C3C3C"
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#B0B0B0"
    ERROR_COLOR = "#CF6679"
    SUCCESS_COLOR = "#4CAF50"

    def __init__(self, page: ft.Page, project_path: str, on_back_to_welcome: Optional[Callable] = None):
        self.page = page
        self.project_path = project_path
        self.on_back_to_welcome = on_back_to_welcome

        # UI State
        self.current_tab = 0
        self.logs = []
        self.logs_display = ft.Column(height=300, scroll=ft.ScrollMode.AUTO, spacing=2)
        self.training_active = False
        self.current_view_class = None
        self.images_dialog = None  # Store dialog reference
        self.upload_modal = None   # Store upload modal reference

        # Initialize core components
        self.project_config = ProjectConfig(project_path)
        self.dataset_manager = DatasetManager(project_path, self.project_config)
        self.trainer = PyTorchTrainer(project_path, self._log_message)

        # Setup page theme
        self._setup_theme()

        # Initialize UI components
        self._init_ui_components()

    def _setup_theme(self):
        """Setup Android Studio dark theme."""
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.bgcolor = self.BACKGROUND_DARK
        self.page.theme = ft.Theme(
            color_scheme=ft.ColorScheme(
                primary=self.PRIMARY_COLOR,
                secondary=self.SECONDARY_COLOR,
                surface=self.SURFACE_COLOR,
                background=self.BACKGROUND_DARK,
                on_primary=self.TEXT_PRIMARY,
                on_secondary=self.TEXT_PRIMARY,
                on_surface=self.TEXT_PRIMARY,
                on_background=self.TEXT_PRIMARY,
            )
        )

    def _init_ui_components(self):
        """Initialize all UI components."""
        # Initialize file picker for uploads
        def on_upload_result(e: ft.FilePickerResultEvent):
            if e.files and hasattr(self, 'current_upload_class'):
                file_paths = [f.path for f in e.files]
                success_count, error_count = self.dataset_manager.add_images_to_class(self.current_upload_class, file_paths)
                
                if success_count > 0:
                    self._update_classes_display()
                    self._show_snackbar(f"✅ {success_count} imágenes agregadas", self.SUCCESS_COLOR)
                if error_count > 0:
                    self._show_snackbar(f"⚠️ {error_count} archivos no válidos", self.ERROR_COLOR)
        
        self.file_picker = ft.FilePicker(on_result=on_upload_result)
        self.page.overlay.append(self.file_picker)
        
        # Initialize folder picker for uploads
        def on_folder_result(e: ft.FilePickerResultEvent):
            if e.path and hasattr(self, 'current_upload_class'):
                # Get all image files from the selected folder
                folder_path = e.path
                image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
                
                try:
                    image_files = [
                        os.path.join(folder_path, f) 
                        for f in os.listdir(folder_path) 
                        if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(image_extensions)
                    ]
                    
                    if not image_files:
                        self._show_snackbar(f"⚠️ No se encontraron imágenes en la carpeta", self.ERROR_COLOR)
                        return
                    
                    success_count, error_count = self.dataset_manager.add_images_to_class(self.current_upload_class, image_files)
                    
                    if success_count > 0:
                        self._update_classes_display()
                        self._show_snackbar(f"✅ {success_count} imágenes agregadas desde carpeta", self.SUCCESS_COLOR)
                    if error_count > 0:
                        self._show_snackbar(f"⚠️ {error_count} archivos no válidos", self.ERROR_COLOR)
                except Exception as ex:
                    self._show_snackbar(f"❌ Error al leer carpeta: {str(ex)}", self.ERROR_COLOR)
        
        self.folder_picker = ft.FilePicker(on_result=on_folder_result)
        self.page.overlay.append(self.folder_picker)
        
        # Initialize model picker for test tab
        def on_model_result(e: ft.FilePickerResultEvent):
            if e.files and len(e.files) > 0:
                model_path = e.files[0].path
                if model_path.endswith('.plt'):
                    self._load_model_for_testing(model_path)
                else:
                    self._show_snackbar("❌ Solo se aceptan archivos .plt", self.ERROR_COLOR)
        
        self.model_picker = ft.FilePicker(on_result=on_model_result)
        self.page.overlay.append(self.model_picker)
        
        # Initialize image picker for test tab predictions
        def on_test_image_result(e: ft.FilePickerResultEvent):
            if e.files and len(e.files) > 0:
                image_path = e.files[0].path
                self._test_model_with_image(image_path)
        
        self.test_image_picker = ft.FilePicker(on_result=on_test_image_result)
        self.page.overlay.append(self.test_image_picker)
        
        # Initialize form fields for add class
        self.class_name_field = ft.TextField(
            label="Nombre de la clase",
            autofocus=True,
            width=300,
            border_color=self.SURFACE_COLOR,
            focused_border_color=self.PRIMARY_COLOR,
        )
        self.add_class_form = ft.Container(
            visible=False,
            content=ft.Column([
                ft.Text("Agregar Nueva Clase", size=16, weight=ft.FontWeight.BOLD, color=self.TEXT_PRIMARY),
                self.class_name_field,
                ft.Row([
                    ft.ElevatedButton(
                        "Agregar",
                        on_click=self._add_class_from_form,
                        style=ft.ButtonStyle(bgcolor=self.PRIMARY_COLOR, color=self.TEXT_PRIMARY)
                    ),
                    ft.ElevatedButton(
                        "Cancelar",
                        on_click=self._toggle_add_class_form,
                        style=ft.ButtonStyle(bgcolor=self.SECONDARY_COLOR, color=self.TEXT_PRIMARY)
                    ),
                ], spacing=10),
            ]),
            bgcolor=self.SURFACE_COLOR,
            padding=20,
            border_radius=8,
            margin=ft.margin.only(bottom=20)
        )

        # App Bar
        self.app_bar = self._create_app_bar()

        # Tab Bar
        self.tabs = self._create_tabs()

        # Status Bar
        self.status_bar = self._create_status_bar()

    def _create_upload_modal(self):
        """Create upload images modal."""
        def on_upload_files(e):
            if e.files:
                for file in e.files:
                    success = self.dataset_manager.add_images_to_class(self.current_upload_class, [file.path])
                    if success[0] > 0:
                        self._show_snackbar(f"✅ {success[0]} imagen(es) subida(s)", self.SUCCESS_COLOR)
                    if success[1] > 0:
                        self._show_snackbar(f"⚠️ {success[1]} imagen(es) rechazada(s)", self.SECONDARY_COLOR)
                self._update_classes_display()
                self.upload_modal.open = False
                self.page.update()
        
        file_picker = ft.FilePicker(on_result=on_upload_files)
        self.page.overlay.append(file_picker)
        
        return ft.AlertDialog(
            modal=True,
            title=ft.Text("Subir Imágenes"),
            content=ft.Column([
                ft.Text("¿Qué deseas hacer?", size=14, color=self.TEXT_SECONDARY),
            ]),
            actions=[
                ft.TextButton(
                    "Cancelar",
                    on_click=lambda e: self._close_upload_modal()
                ),
                ft.ElevatedButton(
                    "Seleccionar Imágenes",
                    on_click=lambda e: file_picker.pick_files(allowed_extensions=["jpg", "jpeg", "png", "bmp"]),
                    style=ft.ButtonStyle(bgcolor=self.PRIMARY_COLOR, color=self.TEXT_PRIMARY)
                ),
            ],
        )

    def _close_upload_modal(self, e=None):
        """Close upload modal."""
        if hasattr(self, 'upload_modal'):
            self.upload_modal.open = False
            self.page.update()

    def _create_app_bar(self) -> ft.Container:
        """Create Android Studio-style app bar."""
        return ft.Container(
            content=ft.Row([
                ft.IconButton(
                    ft.Icons.ARROW_BACK,
                    icon_color=self.TEXT_SECONDARY,
                    tooltip="Back to Welcome Screen",
                    on_click=self._on_back_clicked
                ),
                ft.Text(
                    f"BuildBrain - {os.path.basename(self.project_path)}",
                    size=18,
                    weight=ft.FontWeight.BOLD,
                    color=self.TEXT_PRIMARY
                ),
                ft.Container(expand=True),
                ft.Icon(ft.Icons.BUILD, color=self.PRIMARY_COLOR, size=24),
            ], alignment=ft.MainAxisAlignment.START),
            bgcolor=self.BACKGROUND_LIGHT,
            padding=ft.padding.symmetric(horizontal=20, vertical=12),
            border_radius=ft.border_radius.only(bottom_left=8, bottom_right=8),
        )

    def _create_tabs(self) -> ft.Tabs:
        """Create main tab interface."""
        return ft.Tabs(
            tabs=[
                ft.Tab(
                    text="Configuración",
                    icon=ft.Icons.SETTINGS,
                    content=self._create_config_tab()
                ),
                ft.Tab(
                    text="Dataset",
                    icon=ft.Icons.DATA_USAGE,
                    content=self._create_dataset_tab()
                ),
                ft.Tab(
                    text="Entrenamiento",
                    icon=ft.Icons.PLAY_ARROW,
                    content=self._create_training_tab()
                ),
                ft.Tab(
                    text="Test",
                    icon=ft.Icons.CODE,
                    content=self._create_test_tab()
                ),
            ],
            expand=True,
            selected_index=self.current_tab,
            on_change=self._on_tab_change,
            indicator_color=self.PRIMARY_COLOR,
            label_color=self.TEXT_SECONDARY,
            unselected_label_color=self.TEXT_SECONDARY,
        )

    def _create_status_bar(self) -> ft.Container:
        """Create status bar with project info."""
        project_info = self.project_config.get_project_info()

        return ft.Container(
            content=ft.Row([
                ft.Text(
                    f"Proyecto: {project_info['name']} | "
                    f"Clases: {project_info['classes_count']} | "
                    f"Épocas: {project_info['epochs']}",
                    size=12,
                    color=self.TEXT_SECONDARY
                ),
                ft.Container(expand=True),
                ft.Text(
                    f"Última modificación: {project_info['last_modified'][:19]}",
                    size=12,
                    color=self.TEXT_SECONDARY
                ),
            ]),
            bgcolor=self.BACKGROUND_LIGHT,
            padding=ft.padding.symmetric(horizontal=20, vertical=8),
            border_radius=ft.border_radius.only(top_left=8, top_right=8),
        )

    def _create_config_tab(self) -> ft.Container:
        """Create configuration tab content."""
        # Load current values
        project_name = self.project_config.get_setting("project_name", "")
        model_type = self.project_config.get_setting("model_type", "PyTorch_CNN")
        epochs = self.project_config.get_setting("epochs", 50)
        batch_size = self.project_config.get_setting("batch_size", 32)
        learning_rate = self.project_config.get_setting("learning_rate", 0.001)
        optimizer = self.project_config.get_setting("optimizer", "adam")
        loss_function = self.project_config.get_setting("loss_function", "cross_entropy")
        scheduler = self.project_config.get_setting("scheduler", "none")
        weight_decay = self.project_config.get_setting("weight_decay", 0.0)
        momentum = self.project_config.get_setting("momentum", 0.9)
        data_augmentation = self.project_config.get_setting("data_augmentation", True)
        validation_split = self.project_config.get_setting("validation_split", 0.2)
        early_stopping = self.project_config.get_setting("early_stopping", False)
        patience = self.project_config.get_setting("patience", 10)

        # Create form fields
        self.config_fields = {
            "project_name": ft.TextField(
                label="Nombre del Proyecto",
                value=project_name,
                width=300,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "model_type": ft.Dropdown(
                label="Tipo de Modelo",
                value=model_type,
                options=[
                    ft.dropdown.Option("PyTorch_CNN", "PyTorch CNN"),
                    ft.dropdown.Option("ResNet18", "ResNet-18"),
                    ft.dropdown.Option("ResNet50", "ResNet-50"),
                ],
                width=300,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "epochs": ft.TextField(
                label="Épocas",
                value=str(epochs),
                width=150,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "batch_size": ft.TextField(
                label="Tamaño de Lote",
                value=str(batch_size),
                width=150,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "learning_rate": ft.TextField(
                label="Tasa de Aprendizaje",
                value=str(learning_rate),
                width=150,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "optimizer": ft.Dropdown(
                label="Optimizador",
                value=optimizer,
                options=[
                    ft.dropdown.Option("adam", "Adam"),
                    ft.dropdown.Option("sgd", "SGD"),
                    ft.dropdown.Option("rmsprop", "RMSprop"),
                ],
                width=200,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "loss_function": ft.Dropdown(
                label="Función de Pérdida",
                value=loss_function,
                options=[
                    ft.dropdown.Option("cross_entropy", "Cross Entropy"),
                    ft.dropdown.Option("mse", "MSE"),
                    ft.dropdown.Option("bce", "Binary Cross Entropy"),
                ],
                width=200,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "scheduler": ft.Dropdown(
                label="Scheduler de LR",
                value=scheduler,
                options=[
                    ft.dropdown.Option("none", "Ninguno"),
                    ft.dropdown.Option("step", "Step LR"),
                    ft.dropdown.Option("cosine", "Cosine Annealing"),
                    ft.dropdown.Option("exponential", "Exponential"),
                ],
                width=200,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "weight_decay": ft.TextField(
                label="Weight Decay",
                value=str(weight_decay),
                width=150,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "momentum": ft.TextField(
                label="Momentum (SGD)",
                value=str(momentum),
                width=150,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "validation_split": ft.TextField(
                label="Split de Validación",
                value=str(validation_split),
                width=150,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "patience": ft.TextField(
                label="Paciencia Early Stopping",
                value=str(patience),
                width=150,
                border_color=self.SURFACE_COLOR,
                focused_border_color=self.PRIMARY_COLOR,
            ),
            "data_augmentation": ft.Checkbox(
                label="Data Augmentation",
                value=data_augmentation,
            ),
            "early_stopping": ft.Checkbox(
                label="Early Stopping",
                value=early_stopping,
            ),
        }

        # Save button
        save_button = ft.ElevatedButton(
            "Guardar Cambios",
            icon=ft.Icons.SAVE,
            on_click=self._save_config,
            style=ft.ButtonStyle(
                bgcolor=self.PRIMARY_COLOR,
                color=self.TEXT_PRIMARY,
            )
        )

        return ft.Container(
            content=ft.Column([
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.SETTINGS, color=self.PRIMARY_COLOR, size=28),
                        ft.Text("Configuración del Proyecto", size=20, weight=ft.FontWeight.BOLD),
                    ]),
                    margin=ft.margin.only(bottom=20)
                ),

                # Form sections
                ft.Container(
                    content=ft.Column([
                        ft.Text("Información Básica", size=16, weight=ft.FontWeight.BOLD, color=self.TEXT_PRIMARY),
                        ft.Row([
                            self.config_fields["project_name"],
                            self.config_fields["model_type"],
                        ], spacing=20),
                    ]),
                    bgcolor=self.SURFACE_COLOR,
                    padding=20,
                    border_radius=8,
                    margin=ft.margin.only(bottom=20)
                ),

                ft.Container(
                    content=ft.Column([
                        ft.Text("Parámetros de Entrenamiento", size=16, weight=ft.FontWeight.BOLD, color=self.TEXT_PRIMARY),
                        ft.Row([
                            self.config_fields["epochs"],
                            self.config_fields["batch_size"],
                            self.config_fields["learning_rate"],
                        ], spacing=20),
                        ft.Container(height=10),
                        ft.Row([
                            self.config_fields["optimizer"],
                            self.config_fields["loss_function"],
                            self.config_fields["scheduler"],
                        ], spacing=20),
                        ft.Container(height=10),
                        ft.Row([
                            self.config_fields["weight_decay"],
                            self.config_fields["momentum"],
                        ], spacing=20),
                    ]),
                    bgcolor=self.SURFACE_COLOR,
                    padding=20,
                    border_radius=8,
                    margin=ft.margin.only(bottom=20)
                ),

                ft.Container(
                    content=ft.Column([
                        ft.Text("Opciones Avanzadas", size=16, weight=ft.FontWeight.BOLD, color=self.TEXT_PRIMARY),
                        ft.Row([
                            self.config_fields["validation_split"],
                            self.config_fields["patience"],
                        ], spacing=20),
                        ft.Container(height=10),
                        ft.Row([
                            self.config_fields["data_augmentation"],
                            self.config_fields["early_stopping"],
                        ], spacing=20),
                    ]),
                    bgcolor=self.SURFACE_COLOR,
                    padding=20,
                    border_radius=8,
                    margin=ft.margin.only(bottom=20)
                ),

                # Save button
                ft.Container(
                    content=save_button,
                    alignment=ft.alignment.center,
                ),

            ], scroll=ft.ScrollMode.AUTO, spacing=20),
            padding=20
        )

    def _create_dataset_tab(self) -> ft.Container:
        """Create dataset management tab content."""
        # Classes list container
        self.classes_container = ft.Column(spacing=10)

        # Update classes display
        self._update_classes_display()

        return ft.Container(
            content=ft.Column([
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.DATA_USAGE, color=self.SECONDARY_COLOR, size=28),
                        ft.Text("Gestión de Dataset", size=20, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),
                        ft.ElevatedButton(
                            "Agregar Clase",
                            icon=ft.Icons.ADD,
                            on_click=self._toggle_add_class_form,
                            style=ft.ButtonStyle(bgcolor=self.PRIMARY_COLOR, color=self.TEXT_PRIMARY)
                        ),
                    ]),
                    margin=ft.margin.only(bottom=20)
                ),

                # Add class form
                self.add_class_form,

                # Classes list
                self.classes_container,

            ], scroll=ft.ScrollMode.AUTO, spacing=20),
            padding=20
        )

    def _load_saved_metrics(self) -> Optional[Dict[str, Any]]:
        """Load saved metrics from the last training."""
        try:
            metrics_json_path = os.path.join(self.project_path, "training_reports", "metrics.json")
            if os.path.exists(metrics_json_path):
                with open(metrics_json_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading saved metrics: {str(e)}")
        return None

    def _create_training_tab(self) -> ft.Container:
        """Create training tab content."""
        # Logs display
        self.logs_display = ft.Column(
            height=300,
            scroll=ft.ScrollMode.AUTO,
            spacing=2
        )

        # Metrics display - cargar métricas guardadas si existen
        self.metrics_display = ft.Column(spacing=10, visible=False)
        saved_metrics = self._load_saved_metrics()
        if saved_metrics:
            self._populate_metrics_display(saved_metrics)
            self.metrics_display.visible = True
            self.download_metrics_btn_visible = True
        else:
            self.download_metrics_btn_visible = False

        # Progress indicators
        self.progress_bar = ft.ProgressBar(width=400, color=self.PRIMARY_COLOR, visible=False)
        self.progress_text = ft.Text("", size=14, color=self.TEXT_SECONDARY, visible=False)

        # Training buttons
        quick_train_btn = ft.ElevatedButton(
            "Entrenamiento Rápido",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._start_quick_training,
            style=ft.ButtonStyle(bgcolor=self.SUCCESS_COLOR, color=self.TEXT_PRIMARY),
            disabled=self.training_active
        )

        advanced_train_btn = ft.ElevatedButton(
            "Entrenamiento Avanzado",
            icon=ft.Icons.PLAY_CIRCLE_FILLED,
            on_click=self._start_advanced_training,
            style=ft.ButtonStyle(bgcolor=self.PRIMARY_COLOR, color=self.TEXT_PRIMARY),
            disabled=self.training_active
        )

        stop_train_btn = ft.ElevatedButton(
            "Detener Entrenamiento",
            icon=ft.Icons.STOP,
            on_click=self._stop_training,
            style=ft.ButtonStyle(bgcolor=self.ERROR_COLOR, color=self.TEXT_PRIMARY),
            visible=self.training_active
        )
        
        self.download_metrics_btn = ft.ElevatedButton(
            "Descargar Gráficas",
            icon=ft.Icons.DOWNLOAD,
            on_click=self._download_metrics,
            style=ft.ButtonStyle(bgcolor=self.SECONDARY_COLOR, color=self.TEXT_PRIMARY),
            visible=self.download_metrics_btn_visible
        )

        return ft.Container(
            content=ft.Column([
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.PLAY_ARROW, color=self.PRIMARY_COLOR, size=28),
                        ft.Text("Entrenamiento del Modelo", size=20, weight=ft.FontWeight.BOLD),
                    ]),
                    margin=ft.margin.only(bottom=20)
                ),

                # Training controls
                ft.Container(
                    content=ft.Column([
                        ft.Text("Controles de Entrenamiento", size=16, weight=ft.FontWeight.BOLD),
                        ft.Container(height=10),
                        ft.Row([
                            quick_train_btn,
                            advanced_train_btn,
                            stop_train_btn,
                        ], spacing=20),
                        ft.Container(height=20),
                        self.progress_bar,
                        self.progress_text,
                    ]),
                    bgcolor=self.SURFACE_COLOR,
                    padding=20,
                    border_radius=8,
                    margin=ft.margin.only(bottom=20)
                ),

                # Metrics section
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.SHOW_CHART, color=self.PRIMARY_COLOR),
                            ft.Text("Métricas del Modelo", size=16, weight=ft.FontWeight.BOLD),
                        ]),
                        ft.Container(height=10),
                        self.metrics_display,
                        ft.Container(height=10),
                        self.download_metrics_btn,
                    ]),
                    bgcolor=self.SURFACE_COLOR,
                    padding=20,
                    border_radius=8,
                    margin=ft.margin.only(bottom=20)
                ),

                # Logs console
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.TERMINAL, color=self.TEXT_SECONDARY),
                            ft.Text("Consola de Logs", size=16, weight=ft.FontWeight.BOLD),
                            ft.Container(expand=True),
                            ft.IconButton(
                                ft.Icons.COPY,
                                icon_color=self.PRIMARY_COLOR,
                                tooltip="Copiar logs",
                                on_click=self._copy_logs_to_clipboard
                            ),
                        ]),
                        ft.Container(
                            content=self.logs_display,
                            bgcolor=self.BACKGROUND_LIGHT,
                            padding=10,
                            border_radius=4,
                            height=300
                        ),
                    ]),
                    bgcolor=self.SURFACE_COLOR,
                    padding=20,
                    border_radius=8
                ),

            ], scroll=ft.ScrollMode.AUTO, spacing=20),
            padding=20
        )

    def _update_classes_display(self):
        """Update the classes display in dataset tab."""
        self.classes_container.controls.clear()

        # Load from both filesystem and config
        classes_info = self.dataset_manager.get_classes_info()
        
        # También mostrar las clases del config aunque no tengan imágenes
        config_classes = self.project_config.get_setting("classes", [])
        for class_name in config_classes:
            if not any(c["name"] == class_name for c in classes_info):
                classes_info.append({
                    "name": class_name,
                    "image_count": 0,
                    "path": os.path.join(self.dataset_manager.dataset_path, class_name)
                })

        if not classes_info:
            self.classes_container.controls.append(
                ft.Container(
                    content=ft.Text(
                        "No hay clases definidas. Agrega una clase para comenzar.",
                        color=self.TEXT_SECONDARY,
                        text_align=ft.TextAlign.CENTER
                    ),
                    height=100,
                    alignment=ft.alignment.center
                )
            )
        else:
            for class_info in classes_info:
                class_card = self._create_class_card(class_info)
                self.classes_container.controls.append(class_card)

        self.page.update()

    def _create_class_card(self, class_info: Dict[str, Any]) -> ft.Card:
        """Create a card for a class."""
        class_name = class_info["name"]
        
        def on_view_images(e):
            self._view_class_images(class_name)
        
        def on_upload_images(e):
            self._upload_images_to_class(class_name)
        
        def on_delete_class(e):
            self._delete_class(class_name)
        
        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Row([
                        ft.Icon(ft.Icons.FOLDER, color=self.SECONDARY_COLOR),
                        ft.Text(class_info["name"], size=16, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),
                        ft.Text(f"{class_info['image_count']} imágenes", color=self.TEXT_SECONDARY),
                    ]),
                    ft.Container(height=10),
                    ft.Row([
                        ft.ElevatedButton(
                            "Subir Imágenes",
                            icon=ft.Icons.CLOUD_UPLOAD,
                            on_click=on_upload_images,
                            style=ft.ButtonStyle(bgcolor=self.PRIMARY_COLOR, color=self.TEXT_PRIMARY)
                        ),
                        ft.ElevatedButton(
                            "Ver Imágenes",
                            icon=ft.Icons.VISIBILITY,
                            on_click=on_view_images,
                            style=ft.ButtonStyle(bgcolor=self.PRIMARY_COLOR, color=self.TEXT_PRIMARY)
                        ),
                        ft.IconButton(
                            ft.Icons.DELETE,
                            icon_color=self.ERROR_COLOR,
                            tooltip="Eliminar clase",
                            on_click=on_delete_class
                        ),
                    ], spacing=10),
                ]),
                padding=20
            ),
            elevation=2
        )

    def _create_test_tab(self) -> ft.Container:
        """Create test tab content for model inference."""
        # Store references for test tab
        self.test_model_loaded = False
        self.test_model_path = None
        self.test_model_dropdown = ft.Dropdown(
            label="Selecciona modelo",
            width=300,
            on_change=self._on_model_selected,
            options=[]
        )
        self.test_model_label = ft.Text("No hay modelo cargado", size=12, color=self.TEXT_SECONDARY)
        self.test_status_label = ft.Text("⏹️ Inactivo", size=11, color=self.TEXT_SECONDARY)
        self.test_image_label = ft.Text("Selecciona una imagen", size=12, color=self.TEXT_SECONDARY)
        self.test_result_container = ft.Column(spacing=10)
        self.test_image_display = ft.Container(
            height=300,
            bgcolor=self.SURFACE_COLOR,
            border_radius=12,
            alignment=ft.alignment.center,
            content=ft.Text("📷 Imagen no cargada", color=self.TEXT_SECONDARY)
        )
        
        # Refresh available models
        self._refresh_model_list()
        
        def on_refresh_models(e):
            """Refresh the list of available models."""
            self._refresh_model_list()
            self.page.update()
            self._show_snackbar("✅ Lista de modelos actualizada", self.SUCCESS_COLOR)
        
        def on_select_test_image(e):
            if not self.test_model_loaded:
                self._show_snackbar("⚠️ Carga un modelo primero", self.ERROR_COLOR)
                return
            self.test_image_picker.pick_files(
                allowed_extensions=["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
            )
        
        def on_clear_test(e):
            self.test_model_loaded = False
            self.test_model_path = None
            self.test_model_dropdown.value = None
            self.test_model_label.value = "No hay modelo cargado"
            self.test_model_label.color = self.TEXT_SECONDARY
            self.test_status_label.value = "⏹️ Inactivo"
            self.test_status_label.color = self.TEXT_SECONDARY
            self.test_image_label.value = "Selecciona una imagen"
            self.test_result_container.controls.clear()
            self.test_image_display.content = ft.Text("📷 Imagen no cargada", color=self.TEXT_SECONDARY)
            self.page.update()
        
        # Left panel - Model & Controls
        left_panel = ft.Container(
            content=ft.Column([
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.CODE, color=self.PRIMARY_COLOR, size=28),
                        ft.Text("Prueba del Modelo", size=18, weight=ft.FontWeight.BOLD),
                    ]),
                    margin=ft.margin.only(bottom=15)
                ),
                
                # Model selector
                ft.Container(
                    content=ft.Column([
                        ft.Text("1️⃣ Cargar Modelo", size=13, weight=ft.FontWeight.BOLD),
                        ft.Container(height=8),
                        ft.Row([
                            self.test_model_dropdown,
                            ft.IconButton(
                                ft.Icons.REFRESH,
                                tooltip="Refrescar modelos",
                                on_click=on_refresh_models,
                                icon_color=self.SECONDARY_COLOR
                            )
                        ], spacing=5),
                        ft.Container(height=8),
                        self.test_model_label,
                    ]),
                    bgcolor=self.SURFACE_COLOR,
                    padding=15,
                    border_radius=8,
                    margin=ft.margin.only(bottom=15)
                ),
                
                # Image selector
                ft.Container(
                    content=ft.Column([
                        ft.Text("2️⃣ Seleccionar Imagen", size=13, weight=ft.FontWeight.BOLD),
                        ft.Container(height=8),
                        ft.ElevatedButton(
                            "Cargar Imagen",
                            icon=ft.Icons.IMAGE_SEARCH,
                            on_click=on_select_test_image,
                            style=ft.ButtonStyle(bgcolor=self.SECONDARY_COLOR, color=self.TEXT_PRIMARY),
                            expand=True
                        ),
                        ft.Container(height=8),
                        self.test_image_label,
                    ]),
                    bgcolor=self.SURFACE_COLOR,
                    padding=15,
                    border_radius=8,
                    margin=ft.margin.only(bottom=15)
                ),
                
                # Statistics
                ft.Container(
                    content=ft.Column([
                        ft.Text("📊 Información", size=13, weight=ft.FontWeight.BOLD),
                        ft.Container(height=10),
                        ft.Row([
                            ft.Text("Estado:", size=11, weight=ft.FontWeight.BOLD, width=80),
                            self.test_status_label,
                        ]),
                    ], spacing=5),
                    bgcolor=self.BACKGROUND_LIGHT,
                    padding=15,
                    border_radius=8,
                    margin=ft.margin.only(bottom=15)
                ),
                
                # Action buttons
                ft.Row([
                    ft.ElevatedButton(
                        "Limpiar",
                        icon=ft.Icons.DELETE,
                        on_click=on_clear_test,
                        style=ft.ButtonStyle(bgcolor=self.ERROR_COLOR, color=self.TEXT_PRIMARY),
                        expand=True
                    ),
                ], spacing=10),
                
                ft.Container(expand=True),
            ], spacing=0, scroll=ft.ScrollMode.AUTO),
            width=320,
            padding=15
        )
        
        # Right panel - Image and Results
        right_panel = ft.Container(
            content=ft.Column([
                # Image display
                ft.Container(
                    content=ft.Column([
                        ft.Text("📷 Vista Previa", size=13, weight=ft.FontWeight.BOLD),
                        ft.Container(height=10),
                        self.test_image_display,
                    ]),
                    bgcolor=self.SURFACE_COLOR,
                    padding=15,
                    border_radius=8,
                    margin=ft.margin.only(bottom=15),
                    expand=True
                ),
                
                # Results section
                ft.Container(
                    content=ft.Column([
                        ft.Text("🎯 Resultados", size=13, weight=ft.FontWeight.BOLD),
                        ft.Container(height=10),
                        ft.Column([
                            self.test_result_container
                        ], scroll=ft.ScrollMode.AUTO, expand=True),
                    ], expand=True),
                    bgcolor=self.SURFACE_COLOR,
                    padding=15,
                    border_radius=8,
                    expand=True
                ),
            ], spacing=0, expand=True),
            expand=True,
            padding=15
        )
        
        return ft.Container(
            content=ft.Row([
                left_panel,
                ft.VerticalDivider(width=1, color=self.BACKGROUND_LIGHT),
                right_panel,
            ], spacing=0, expand=True),
            bgcolor=self.BACKGROUND_DARK
        )

    def build(self) -> ft.Container:
        """Build the complete main window."""
        return ft.Container(
            content=ft.Column([
                self.app_bar,
                ft.Container(
                    content=self.tabs,
                    expand=True,
                    bgcolor=self.BACKGROUND_DARK
                ),
                self.status_bar,
            ]),
            expand=True,
            bgcolor=self.BACKGROUND_DARK
        )

    # Event handlers
    def _on_back_clicked(self, e):
        """Handle back button click."""
        if self.on_back_to_welcome:
            self.on_back_to_welcome()

    def _on_tab_change(self, e):
        """Handle tab change."""
        self.current_tab = e.control.selected_index

    def _save_config(self, e):
        """Save configuration changes."""
        try:
            # Get the old project name to check if it changed
            old_project_name = self.project_config.get_setting("project_name", "")
            new_project_name = self.config_fields["project_name"].value.strip()
            
            # Update config values
            self.project_config.update_setting("project_name", new_project_name)
            self.project_config.update_setting("model_type", self.config_fields["model_type"].value)
            self.project_config.update_setting("epochs", int(self.config_fields["epochs"].value))
            self.project_config.update_setting("batch_size", int(self.config_fields["batch_size"].value))
            self.project_config.update_setting("learning_rate", float(self.config_fields["learning_rate"].value))
            self.project_config.update_setting("optimizer", self.config_fields["optimizer"].value)
            self.project_config.update_setting("loss_function", self.config_fields["loss_function"].value)
            self.project_config.update_setting("scheduler", self.config_fields["scheduler"].value)
            self.project_config.update_setting("weight_decay", float(self.config_fields["weight_decay"].value))
            self.project_config.update_setting("momentum", float(self.config_fields["momentum"].value))
            self.project_config.update_setting("validation_split", float(self.config_fields["validation_split"].value))
            self.project_config.update_setting("patience", int(self.config_fields["patience"].value))
            self.project_config.update_setting("data_augmentation", self.config_fields["data_augmentation"].value)
            self.project_config.update_setting("early_stopping", self.config_fields["early_stopping"].value)

            # Save to file
            if self.project_config.save_config():
                # If project name changed, rename the project folder
                if old_project_name != new_project_name and new_project_name:
                    self._rename_project_folder(old_project_name, new_project_name)
                
                self._show_snackbar("✅ Configuración guardada exitosamente", self.SUCCESS_COLOR)
                # Update status bar
                self._update_status_bar()
            else:
                self._show_snackbar("❌ Error al guardar la configuración", self.ERROR_COLOR)

        except ValueError as ve:
            self._show_snackbar(f"❌ Error en los valores: {str(ve)}", self.ERROR_COLOR)
        except Exception as ex:
            self._show_snackbar(f"❌ Error inesperado: {str(ex)}", self.ERROR_COLOR)

    def _rename_project_folder(self, old_name: str, new_name: str):
        """Rename the project folder."""
        try:
            import shutil
            
            # Get the parent directory of the project
            projects_parent = os.path.dirname(self.project_path)
            old_project_path = os.path.join(projects_parent, old_name)
            new_project_path = os.path.join(projects_parent, new_name)
            
            # Check if old folder exists
            if not os.path.exists(old_project_path):
                # The folder might have already been renamed or doesn't exist
                self._show_snackbar(f"⚠️ Carpeta original no encontrada", self.ERROR_COLOR)
                return
            
            # Check if new folder already exists
            if os.path.exists(new_project_path):
                self._show_snackbar(f"❌ Ya existe una carpeta con ese nombre", self.ERROR_COLOR)
                return
            
            # Rename the folder
            shutil.move(old_project_path, new_project_path)
            
            # Update the project path in the current instance
            self.project_path = new_project_path
            self.project_config.project_path = new_project_path
            self.dataset_manager.dataset_path = os.path.join(new_project_path, "dataset")
            self.trainer.project_dir = new_project_path
            
            # Update the window title
            self.app_bar.title = ft.Text(
                f"BuildBrain - {new_name}",
                size=16,
                weight=ft.FontWeight.BOLD,
                color=self.TEXT_PRIMARY
            )
            
            self._show_snackbar(f"✅ Carpeta renombrada a '{new_name}'", self.SUCCESS_COLOR)
            self.page.update()
            
        except Exception as ex:
            self._show_snackbar(f"❌ Error renombrando carpeta: {str(ex)}", self.ERROR_COLOR)
            print(f"Error renaming project folder: {str(ex)}")

    def _toggle_add_class_form(self, e=None):
        """Toggle the add class form visibility."""
        self.add_class_form.visible = not self.add_class_form.visible
        if self.add_class_form.visible:
            self.class_name_field.value = ""
            self.class_name_field.focus()
        self.page.update()

    def _add_class_from_form(self, e):
        """Add a new class from the form."""
        class_name = self.class_name_field.value.strip()
        if class_name:
            if self.dataset_manager.add_class(class_name):
                self.project_config.add_class(class_name)
                self.project_config.save_config()
                self._update_classes_display()
                self._show_snackbar(f"✅ Clase '{class_name}' agregada", self.SUCCESS_COLOR)
                self._toggle_add_class_form()
            else:
                self._show_snackbar(f"❌ La clase '{class_name}' ya existe", self.ERROR_COLOR)
        else:
            self._show_snackbar("❌ Ingresa un nombre para la clase", self.ERROR_COLOR)



    def _view_class_images(self, class_name: str):
        """View images - show in overlay modal."""
        images = self.dataset_manager.get_class_images(class_name)
        if not images:
            self._show_snackbar(f"Sin imágenes", self.SECONDARY_COLOR)
            return
        
        # Create close function
        def close_modal(e=None):
            for ctrl in self.page.overlay:
                if isinstance(ctrl, ft.Container) and hasattr(ctrl, 'data') and ctrl.data == 'images_modal':
                    self.page.overlay.remove(ctrl)
                    break
            self.page.update()
        
        # Create grid
        grid = ft.GridView(runs_count=4, spacing=10, run_spacing=10, auto_scroll=True, height=450)
        
        for img in images:
            del_btn = ft.IconButton(ft.Icons.DELETE, icon_color=self.ERROR_COLOR)
            del_btn.on_click = lambda e, f=img['filename'], c=class_name: (
                self.dataset_manager.remove_image_from_class(c, f),
                self.project_config.save_config(),
                close_modal(),
                self._view_class_images(c),
                self._show_snackbar("✅ Eliminada", self.SUCCESS_COLOR)
            )[-1]
            
            card = ft.Container(
                content=ft.Stack([
                    ft.Image(src=img['path'], fit=ft.ImageFit.COVER, border_radius=8),
                    ft.Container(content=del_btn, alignment=ft.alignment.top_right,
                                bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.BLACK), padding=3, border_radius=8)
                ]),
                width=140, height=140, border_radius=8, border=ft.border.all(1, self.SURFACE_COLOR)
            )
            grid.controls.append(card)
        
        close_btn = ft.IconButton(ft.Icons.CLOSE, on_click=close_modal)
        
        modal_content = ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Text(f"Imágenes: {class_name}", size=18, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True),
                    close_btn
                ]),
                grid
            ], spacing=10),
            bgcolor=self.SURFACE_COLOR,
            padding=20,
            border_radius=12,
            width=750,
            shadow=ft.BoxShadow(blur_radius=20, color="000000")
        )
        
        # Create overlay container
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
            on_click=close_modal
        )
        overlay.data = 'images_modal'
        
        self.page.overlay.append(overlay)
        self.page.update()

    def _delete_image(self, image_path: str, filename: str = None):
        """Delete an image from the current class."""
        if filename is None:
            filename = os.path.basename(image_path)
        
        if self.dataset_manager.remove_image_from_class(self.current_view_class, filename):
            # Remove from config if exists
            dataset = self.project_config.config_data.get("dataset", {})
            if "classes" in dataset and self.current_view_class in dataset["classes"]:
                dataset["classes"][self.current_view_class] = [
                    p for p in dataset["classes"][self.current_view_class] if os.path.basename(p) != filename
                ]
                dataset["total_images"] = sum(len(paths) for paths in dataset["classes"].values())
            self.project_config.save_config()
            self._update_classes_display()
            self._view_class_images(self.current_view_class)  # Refresh
            self._show_snackbar(f"✅ Imagen '{filename}' eliminada", self.SUCCESS_COLOR)
        else:
            self._show_snackbar(f"❌ Error al eliminar la imagen", self.ERROR_COLOR)

    def _delete_class(self, class_name: str):
        """Delete a class - confirmation overlay."""
        def on_delete(e):
            try:
                self.dataset_manager.remove_class(class_name)
                self.project_config.remove_class(class_name)
                self.project_config.save_config()
                self._update_classes_display()
                close_modal()
                self._show_snackbar(f"✅ Eliminada", self.SUCCESS_COLOR)
            except Exception as ex:
                self._show_snackbar(f"❌ Error", self.ERROR_COLOR)
        
        def close_modal(e=None):
            for ctrl in self.page.overlay:
                if isinstance(ctrl, ft.Container) and hasattr(ctrl, 'data') and ctrl.data == 'confirm_modal':
                    self.page.overlay.remove(ctrl)
                    break
            self.page.update()
        
        # Modal content
        modal = ft.Container(
            content=ft.Column([
                ft.Text(f"¿Eliminar '{class_name}'?", size=18, weight=ft.FontWeight.BOLD),
                ft.Text("Se eliminarán todas sus imágenes", size=12, color=self.TEXT_SECONDARY),
                ft.Container(height=20),
                ft.Row([
                    ft.ElevatedButton("Cancelar", on_click=close_modal, 
                                    style=ft.ButtonStyle(bgcolor=self.SECONDARY_COLOR)),
                    ft.ElevatedButton("Eliminar", on_click=on_delete,
                                    style=ft.ButtonStyle(bgcolor=self.ERROR_COLOR, color=self.TEXT_PRIMARY))
                ], spacing=10)
            ], spacing=15),
            bgcolor=self.SURFACE_COLOR,
            padding=20,
            border_radius=12,
            width=350,
            shadow=ft.BoxShadow(blur_radius=20, color="000000")
        )
        
        overlay = ft.Container(
            content=ft.Column([
                ft.Container(expand=True),
                ft.Row([
                    ft.Container(expand=True),
                    modal,
                    ft.Container(expand=True)
                ]),
                ft.Container(expand=True)
            ]),
            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
            alignment=ft.alignment.center,
            expand=True,
            on_click=close_modal
        )
        overlay.data = 'confirm_modal'
        
        self.page.overlay.append(overlay)
        self.page.update()

    def _upload_images_to_class(self, class_name: str):
        """Upload images - show modal with 2 options."""
        self.current_upload_class = class_name
        
        def close_modal(e=None):
            for ctrl in self.page.overlay:
                if isinstance(ctrl, ft.Container) and hasattr(ctrl, 'data') and ctrl.data == 'upload_modal':
                    self.page.overlay.remove(ctrl)
                    break
            self.page.update()
        
        def on_pick_files(e):
            close_modal()
            self.file_picker.pick_files(
                allow_multiple=True,
                allowed_extensions=["jpg", "jpeg", "png", "bmp", "tiff", "webp"]
            )
        
        def on_pick_folder(e):
            close_modal()
            self.folder_picker.get_directory()
        
        # Create buttons
        btn_files = ft.ElevatedButton(
            "Subir Imágenes",
            icon=ft.Icons.IMAGE,
            on_click=on_pick_files,
            style=ft.ButtonStyle(bgcolor=self.PRIMARY_COLOR, color=self.TEXT_PRIMARY),
            width=250
        )
        
        btn_folder = ft.ElevatedButton(
            "Subir Carpeta",
            icon=ft.Icons.FOLDER,
            on_click=on_pick_folder,
            style=ft.ButtonStyle(bgcolor=self.SECONDARY_COLOR, color=self.TEXT_PRIMARY),
            width=250
        )
        
        btn_cancel = ft.ElevatedButton(
            "Cancelar",
            on_click=close_modal,
            style=ft.ButtonStyle(bgcolor="#666666"),
            width=250
        )
        
        # Modal content
        modal = ft.Container(
            content=ft.Column([
                ft.Text(f"Subir imágenes a '{class_name}'", size=18, weight=ft.FontWeight.BOLD),
                ft.Text("Elige cómo deseas agregar imágenes", size=12, color=self.TEXT_SECONDARY),
                ft.Container(height=20),
                btn_files,
                ft.Container(height=10),
                btn_folder,
                ft.Container(height=20),
                btn_cancel,
            ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor=self.SURFACE_COLOR,
            padding=30,
            border_radius=12,
            width=350,
            shadow=ft.BoxShadow(blur_radius=20, color="000000")
        )
        
        overlay = ft.Container(
            content=ft.Column([
                ft.Container(expand=True),
                ft.Row([
                    ft.Container(expand=True),
                    modal,
                    ft.Container(expand=True)
                ]),
                ft.Container(expand=True)
            ]),
            bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
            alignment=ft.alignment.center,
            expand=True,
            on_click=close_modal
        )
        overlay.data = 'upload_modal'
        
        self.page.overlay.append(overlay)
        self.page.update()

    def _start_quick_training(self, e):
        """Start quick training."""
        dataset_stats = self.dataset_manager.get_dataset_stats()
        if dataset_stats["total_images"] == 0:
            self._show_snackbar("❌ No hay imágenes en el dataset", self.ERROR_COLOR)
            return

        self.training_active = True
        self._update_training_buttons()

        config = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam"
        }

        if self.trainer.quick_train(**config):
            self.progress_bar.visible = True
            self.progress_text.visible = True
            self.progress_text.value = "Entrenamiento rápido iniciado..."
            self.page.update()

    def _start_advanced_training(self, e):
        """Start advanced training with project config."""
        dataset_stats = self.dataset_manager.get_dataset_stats()
        if dataset_stats["total_images"] == 0:
            self._show_snackbar("❌ No hay imágenes en el dataset", self.ERROR_COLOR)
            return

        self.training_active = True
        self._update_training_buttons()

        config = {
            "epochs": self.project_config.get_setting("epochs", 50),
            "batch_size": self.project_config.get_setting("batch_size", 32),
            "learning_rate": self.project_config.get_setting("learning_rate", 0.001),
            "optimizer": self.project_config.get_setting("optimizer", "adam")
        }

        if self.trainer.advanced_train(config):
            self.progress_bar.visible = True
            self.progress_text.visible = True
            self.progress_text.value = "Entrenamiento avanzado iniciado..."
            self.page.update()

    def _stop_training(self, e):
        """Stop ongoing training."""
        self.trainer.stop_training()
        self.training_active = False
        self._update_training_buttons()
        self.progress_bar.visible = False
        self.progress_text.visible = False
        self.page.update()

    def _update_training_buttons(self):
        """Update training button states."""
        # This will be called when rebuilding the tab
        pass

    def _refresh_model_list(self):
        """Refresh the list of available models."""
        try:
            # Ensure project_path exists
            if not self.project_path or not os.path.isdir(self.project_path):
                self.test_model_dropdown.options = [ft.dropdown.Option("No hay modelos disponibles")]
                return
            
            model_path = os.path.join(self.project_path, "modelo.plt")
            
            # Add current project model if it exists
            options = []
            if os.path.exists(model_path):
                options.append(ft.dropdown.Option("📦 Modelo del Proyecto"))
            
            # Also check in models subfolder
            models_dir = os.path.join(self.project_path, "models")
            if os.path.exists(models_dir):
                for model_file in os.listdir(models_dir):
                    if model_file.endswith(('.plt', '.pkl', '.pth')):
                        options.append(ft.dropdown.Option(f"📦 {model_file}"))
            
            if options:
                self.test_model_dropdown.options = options
            else:
                self.test_model_dropdown.options = [ft.dropdown.Option("No hay modelos disponibles")]
        except Exception as e:
            self._show_snackbar(f"Error al buscar modelos: {str(e)}", self.ERROR_COLOR)

    def _on_model_selected(self, e):
        """Handle model selection from dropdown."""
        if self.test_model_dropdown.value and "Modelo del Proyecto" in self.test_model_dropdown.value:
            model_path = os.path.join(self.project_path, "modelo.plt")
            self._load_model_for_testing(model_path)

    def _log_message(self, message: str):
        """Add a message to the logs display."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = ft.Text(f"[{timestamp}] {message}", size=12, color=self.TEXT_SECONDARY)

        self.logs_display.controls.append(log_entry)

        # Keep only last 100 logs
        if len(self.logs_display.controls) > 100:
            self.logs_display.controls.pop(0)

        # Check if training completed
        if "✅ Training completed" in message or "✅ Training completed!" in message:
            self.training_active = False
            self._update_training_buttons()
            # Update metrics display
            self._update_metrics_display()

        # Auto-scroll to bottom and update page only if control is added to page
        if hasattr(self.logs_display, 'page') and self.logs_display.page is not None:
            self.logs_display.scroll_to(offset=-1)
            self.page.update()

    def _show_snackbar(self, message: str, color: str = None):
        """Show a snackbar notification."""
        snackbar = ft.SnackBar(
            content=ft.Text(message, color=self.TEXT_PRIMARY),
            bgcolor=color or self.SURFACE_COLOR,
            duration=3000
        )
        self.page.snack_bar = snackbar
        snackbar.open = True
        self.page.update()

    def _update_status_bar(self):
        """Update the status bar with current project info."""
        project_info = self.project_config.get_project_info()
        self.status_bar.content.controls[0].value = (
            f"Proyecto: {project_info['name']} | "
            f"Clases: {project_info['classes_count']} | "
            f"Épocas: {project_info['epochs']}"
        )
        self.status_bar.content.controls[2].value = (
            f"Última modificación: {project_info['last_modified'][:19]}"
        )
        self.page.update()

    def _load_model_for_testing(self, model_path: str):
        """Load a model for testing."""
        try:
            if self.trainer.load_model(model_path):
                self.test_model_loaded = True
                self.test_model_path = model_path
                model_name = os.path.basename(model_path)
                
                # Get model info
                file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
                num_classes = len(self.trainer.class_names)
                classes_str = ", ".join(self.trainer.class_names[:3])
                if num_classes > 3:
                    classes_str += f", +{num_classes-3} más"
                
                # Update label with more info
                self.test_model_label.value = f"✅ Cargado\n📦 {model_name}\n🎓 {num_classes} clases\n💾 {file_size:.2f} MB"
                self.test_model_label.color = self.SUCCESS_COLOR
                
                # Update status
                self.test_status_label.value = f"✅ Listo - {num_classes} clases"
                self.test_status_label.color = self.SUCCESS_COLOR
                
                # Show classes info
                self._show_snackbar(f"✅ Modelo cargado - Clases: {classes_str}", self.SUCCESS_COLOR)
            else:
                self.test_model_label.value = "❌ Error al cargar modelo"
                self.test_model_label.color = self.ERROR_COLOR
                self.test_status_label.value = "❌ Error"
                self.test_status_label.color = self.ERROR_COLOR
                self._show_snackbar("❌ Error al cargar modelo", self.ERROR_COLOR)
        except Exception as e:
            self.test_model_label.value = "❌ Error"
            self.test_model_label.color = self.ERROR_COLOR
            self.test_status_label.value = "❌ Error"
            self.test_status_label.color = self.ERROR_COLOR
            self._show_snackbar(f"❌ Error: {str(e)}", self.ERROR_COLOR)
        
        self.page.update()

    def _test_model_with_image(self, image_path: str):
        """Test the model with a selected image."""
        try:
            if not self.test_model_loaded:
                self._show_snackbar("⚠️ Carga un modelo primero", self.ERROR_COLOR)
                return
            
            if not os.path.exists(image_path):
                self._show_snackbar("❌ Archivo de imagen no encontrado", self.ERROR_COLOR)
                return
            
            # Update image display
            self.test_image_label.value = f"✅ {os.path.basename(image_path)}"
            self.test_image_label.color = self.SUCCESS_COLOR
            
            # Display the image with better styling
            try:
                self.test_image_display.content = ft.Image(
                    src=image_path,
                    width=300,
                    height=300,
                    fit=ft.ImageFit.CONTAIN
                )
            except:
                self.test_image_display.content = ft.Text("❌ No se pudo mostrar la imagen", color=self.TEXT_SECONDARY)
            
            # Run prediction
            predicted_class, confidence, class_probs = self.trainer.predict_image(image_path)
            
            # Clear previous results
            self.test_result_container.controls.clear()
            
            if predicted_class == "Error":
                self.test_result_container.controls.append(
                    ft.Text("❌ Error en la predicción", color=self.ERROR_COLOR, size=14, weight=ft.FontWeight.BOLD)
                )
            else:
                # Add main prediction result with better styling
                confidence_color = self.PRIMARY_COLOR if confidence > 0.7 else self.SECONDARY_COLOR if confidence > 0.5 else self.ERROR_COLOR
                
                result_header = ft.Container(
                    content=ft.Column([
                        ft.Text("🎯 Predicción Principal", size=12, weight=ft.FontWeight.BOLD, color=self.TEXT_SECONDARY),
                        ft.Container(height=8),
                        ft.Text(predicted_class, size=20, weight=ft.FontWeight.BOLD, color=self.PRIMARY_COLOR),
                        ft.Container(height=5),
                        ft.Row([
                            ft.Text("Confianza:", size=12, weight=ft.FontWeight.BOLD),
                            ft.Text(f"{confidence:.1%}", size=16, color=confidence_color, weight=ft.FontWeight.BOLD),
                        ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
                        ft.Container(height=8),
                        # Confidence bar
                        ft.Container(
                            content=ft.Row([
                                ft.Container(
                                    height=8,
                                    width=int(confidence * 200),
                                    bgcolor=confidence_color,
                                    border_radius=4
                                ),
                            ]),
                            width=200,
                            height=8,
                            bgcolor=self.BACKGROUND_LIGHT,
                            border_radius=4,
                            padding=0
                        ),
                    ]),
                    bgcolor=self.SURFACE_COLOR,
                    padding=12,
                    border_radius=8,
                    border=ft.border.all(2, confidence_color)
                )
                self.test_result_container.controls.append(result_header)
                
                # Add class probabilities
                if class_probs:
                    self.test_result_container.controls.append(ft.Container(height=10))
                    self.test_result_container.controls.append(
                        ft.Text("📊 Todas las Clases", size=12, weight=ft.FontWeight.BOLD, color=self.TEXT_SECONDARY)
                    )
                    self.test_result_container.controls.append(ft.Container(height=5))
                    
                    # Sort probabilities in descending order
                    sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
                    
                    for idx, (class_name, prob) in enumerate(sorted_probs, 1):
                        # Color based on probability
                        if class_name == predicted_class:
                            bg_color = self.SURFACE_COLOR
                            border_color = self.PRIMARY_COLOR
                            text_color = self.PRIMARY_COLOR
                        else:
                            bg_color = self.BACKGROUND_LIGHT
                            border_color = None
                            text_color = self.TEXT_PRIMARY
                        
                        # Probability bar with ranking
                        prob_item = ft.Container(
                            content=ft.Row([
                                ft.Text(f"{idx}.", size=11, color=self.TEXT_SECONDARY, width=25),
                                ft.Text(class_name, size=11, width=110, color=text_color, weight=ft.FontWeight.BOLD),
                                ft.Container(
                                    content=ft.Row([
                                        ft.Container(
                                            height=16,
                                            width=int(prob * 180),
                                            bgcolor=self.PRIMARY_COLOR if class_name == predicted_class else self.SECONDARY_COLOR,
                                            border_radius=2
                                        ),
                                    ]),
                                    width=180,
                                    height=16,
                                    bgcolor=self.BACKGROUND_LIGHT,
                                    border_radius=2,
                                    padding=1
                                ),
                                ft.Text(f"{prob:.1%}", size=11, width=45, text_align=ft.TextAlign.RIGHT, color=text_color, weight=ft.FontWeight.BOLD),
                            ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                            bgcolor=bg_color,
                            padding=10,
                            border_radius=6,
                            margin=ft.margin.only(bottom=6),
                            border=ft.border.all(1, border_color) if border_color else None
                        )
                        self.test_result_container.controls.append(prob_item)
            
            self._show_snackbar("✅ Predicción completada", self.SUCCESS_COLOR)
            
        except Exception as e:
            self._show_snackbar(f"❌ Error: {str(e)}", self.ERROR_COLOR)
        
        self.page.update()

    def _populate_metrics_display(self, metrics: Dict[str, Any], class_names: List[str] = None):
        """Populate metrics display with given metrics dictionary."""
        try:
            # Clear previous metrics
            self.metrics_display.controls.clear()
            
            # Overall metrics in a grid
            overall_metrics = ft.Container(
                content=ft.Row([
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Exactitud", size=11, color=self.TEXT_SECONDARY),
                            ft.Text(f"{metrics['accuracy']:.2%}", size=18, weight=ft.FontWeight.BOLD, color=self.PRIMARY_COLOR),
                        ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        bgcolor=self.BACKGROUND_LIGHT,
                        padding=15,
                        border_radius=8,
                        expand=True
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Precisión", size=11, color=self.TEXT_SECONDARY),
                            ft.Text(f"{metrics['precision']:.2%}", size=18, weight=ft.FontWeight.BOLD, color=self.SECONDARY_COLOR),
                        ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        bgcolor=self.BACKGROUND_LIGHT,
                        padding=15,
                        border_radius=8,
                        expand=True
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Recall", size=11, color=self.TEXT_SECONDARY),
                            ft.Text(f"{metrics['recall']:.2%}", size=18, weight=ft.FontWeight.BOLD, color="#FF9800"),
                        ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        bgcolor=self.BACKGROUND_LIGHT,
                        padding=15,
                        border_radius=8,
                        expand=True
                    ),
                    ft.Container(
                        content=ft.Column([
                            ft.Text("F1-Score", size=11, color=self.TEXT_SECONDARY),
                            ft.Text(f"{metrics['f1']:.2%}", size=18, weight=ft.FontWeight.BOLD, color=self.SUCCESS_COLOR),
                        ], spacing=5, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                        bgcolor=self.BACKGROUND_LIGHT,
                        padding=15,
                        border_radius=8,
                        expand=True
                    ),
                ], spacing=10),
                margin=ft.margin.only(bottom=15)
            )
            
            self.metrics_display.controls.append(overall_metrics)
            
            # Per-class metrics
            class_metrics = ft.Column([
                ft.Text("Métricas por Clase", size=12, weight=ft.FontWeight.BOLD, color=self.TEXT_SECONDARY)
            ], spacing=8)
            
            # Use provided class_names or get from trainer
            names_to_use = class_names or (self.trainer.class_names if hasattr(self.trainer, 'class_names') else [])
            
            for class_name in names_to_use:
                class_data = metrics.get('class_report', {}).get(class_name, {})
                precision = class_data.get('precision', 0)
                recall = class_data.get('recall', 0)
                f1 = class_data.get('f1-score', 0)
                support = class_data.get('support', 0)
                
                class_item = ft.Container(
                    content=ft.Row([
                        ft.Text(class_name, size=11, width=100),
                        ft.Text(f"P: {precision:.2%}", size=11, width=80),
                        ft.Text(f"R: {recall:.2%}", size=11, width=80),
                        ft.Text(f"F1: {f1:.2%}", size=11, width=80),
                        ft.Text(f"({int(support)})", size=11, color=self.TEXT_SECONDARY),
                    ]),
                    bgcolor=self.BACKGROUND_LIGHT,
                    padding=8,
                    border_radius=4
                )
                class_metrics.controls.append(class_item)
            
            self.metrics_display.controls.append(class_metrics)

        except Exception as e:
            self._show_snackbar(f"❌ Error mostrando métricas: {str(e)}", self.ERROR_COLOR)

    def _update_metrics_display(self):
        """Update metrics display after training completes."""
        try:
            if not hasattr(self.trainer, 'final_metrics') or self.trainer.final_metrics is None:
                return
            
            metrics = self.trainer.final_metrics
            
            # Use the common method to populate metrics
            self._populate_metrics_display(metrics)
            
            # Show download button
            self.download_metrics_btn.visible = True
            
            # Show metrics section
            self.metrics_display.visible = True
            
            self.page.update()
            
        except Exception as e:
            self._show_snackbar(f"❌ Error mostrando métricas: {str(e)}", self.ERROR_COLOR)

    def _download_metrics(self, e):
        """Show available metrics files to download."""
        try:
            metrics_dir = os.path.join(self.project_path, "training_reports")
            
            if not os.path.exists(metrics_dir):
                self._show_snackbar("❌ No hay reportes disponibles", self.ERROR_COLOR)
                return
            
            # Create a modal to show available files
            def close_modal(e=None):
                for ctrl in self.page.overlay:
                    if isinstance(ctrl, ft.Container) and hasattr(ctrl, 'data') and ctrl.data == 'download_modal':
                        self.page.overlay.remove(ctrl)
                        break
                self.page.update()
            
            files_list = ft.Column(spacing=10)
            
            # Get all files in the metrics directory
            files_in_dir = []
            for filename in os.listdir(metrics_dir):
                file_path = os.path.join(metrics_dir, filename)
                if os.path.isfile(file_path):
                    files_in_dir.append((filename, file_path))
            
            # Sort files to show images first, then JSON
            files_in_dir.sort(key=lambda x: (not x[0].endswith('.png'), x[0]))
            
            for filename, file_path in files_in_dir:
                file_size = os.path.getsize(file_path) / 1024  # KB
                
                def create_download_handler(path, fname):
                    def on_download(e):
                        try:
                            dest_path = self._copy_to_downloads(path, fname)
                            self._show_download_success_dialog(dest_path, fname)
                        except Exception as ex:
                            self._show_snackbar(f"❌ Error: {str(ex)}", self.ERROR_COLOR)
                    return on_download
                
                # Determine icon based on file type
                if filename.endswith('.png'):
                    icon = ft.Icons.IMAGE
                elif filename.endswith('.json'):
                    icon = ft.Icons.DATA_OBJECT
                else:
                    icon = ft.Icons.FILE_PRESENT
                
                file_item = ft.Container(
                    content=ft.Row([
                        ft.Icon(icon, color=self.PRIMARY_COLOR),
                        ft.Column([
                            ft.Text(filename, size=12, weight=ft.FontWeight.BOLD),
                            ft.Text(f"{file_size:.1f} KB", size=10, color=self.TEXT_SECONDARY),
                        ], expand=True),
                        ft.IconButton(
                            ft.Icons.DOWNLOAD,
                            icon_color=self.PRIMARY_COLOR,
                            on_click=create_download_handler(file_path, filename),
                            tooltip="Descargar"
                        ),
                    ]),
                    bgcolor=self.BACKGROUND_LIGHT,
                    padding=12,
                    border_radius=6,
                    border=ft.border.all(1, ft.Colors.with_opacity(0.2, self.PRIMARY_COLOR))
                )
                files_list.controls.append(file_item)
            
            # Create modal content
            modal_content = ft.Container(
                content=ft.Column([
                    ft.Text("📊 Descargar Reportes", size=16, weight=ft.FontWeight.BOLD),
                    ft.Container(height=10),
                    ft.Text(f"Se encontraron {len(files_in_dir)} archivo(s)", size=12, color=self.TEXT_SECONDARY),
                    ft.Divider(height=1, color=self.SURFACE_COLOR),
                    ft.Container(
                        content=files_list,
                        height=300,
                        expand=True
                    ),
                    ft.Container(height=15),
                    ft.ElevatedButton(
                        "Cerrar",
                        on_click=close_modal,
                        style=ft.ButtonStyle(bgcolor=self.SURFACE_COLOR),
                        width=200
                    ),
                ], spacing=10, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=self.SURFACE_COLOR,
                padding=30,
                border_radius=12,
                width=500,
                shadow=ft.BoxShadow(blur_radius=20, color="000000")
            )
            
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
                on_click=lambda e: None if files_list.controls else close_modal()  # No cerrar si hace click en el modal
            )
            overlay.data = 'download_modal'
            
            self.page.overlay.append(overlay)
            self.page.update()
            
        except Exception as ex:
            self._show_snackbar(f"❌ Error: {str(ex)}", self.ERROR_COLOR)

    def _copy_to_downloads(self, source_path: str, filename: str) -> str:
        """Copy file to user's Downloads folder. Returns the destination path."""
        try:
            import shutil
            from pathlib import Path
            
            # Verify source file exists
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Archivo no encontrado: {source_path}")
            
            # Get Downloads folder
            downloads_dir = str(Path.home() / "Downloads")
            
            # Create Downloads folder if it doesn't exist
            if not os.path.exists(downloads_dir):
                os.makedirs(downloads_dir)
            
            # Create destination path
            dest_path = os.path.join(downloads_dir, f"BuildBrain_{filename}")
            
            # If file already exists, add timestamp
            if os.path.exists(dest_path):
                import time
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(downloads_dir, f"BuildBrain_{name}_{timestamp}{ext}")
            
            # Copy file
            shutil.copy2(source_path, dest_path)
            
            print(f"✅ Archivo copiado a: {dest_path}")
            return dest_path
            
        except FileNotFoundError as fe:
            print(f"❌ Error: {str(fe)}")
            raise
        except PermissionError as pe:
            print(f"❌ Error de permisos: {str(pe)}")
            raise
        except Exception as e:
            print(f"❌ Error descargando: {str(e)}")
            raise

    def _show_download_success_dialog(self, file_path: str, filename: str):
        """Show a success dialog with download information."""
        try:
            def close_dialog(e=None):
                for ctrl in self.page.overlay:
                    if isinstance(ctrl, ft.Container) and hasattr(ctrl, 'data') and ctrl.data == 'download_success_dialog':
                        self.page.overlay.remove(ctrl)
                        break
                self.page.update()
            
            def open_folder(e):
                try:
                    import subprocess
                    # Open the Downloads folder
                    subprocess.Popen(['explorer', '/select,', file_path])
                    close_dialog()
                except Exception as ex:
                    self._show_snackbar(f"❌ Error abriendo carpeta: {str(ex)}", self.ERROR_COLOR)
            
            def copy_path(e):
                try:
                    import subprocess
                    process = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
                    process.communicate(file_path.encode('utf-8'))
                    self._show_snackbar("✅ Ruta copiada al portapapeles", self.SUCCESS_COLOR)
                except:
                    try:
                        import tkinter as tk
                        root = tk.Tk()
                        root.withdraw()
                        root.clipboard_clear()
                        root.clipboard_append(file_path)
                        root.update()
                        root.destroy()
                        self._show_snackbar("✅ Ruta copiada al portapapeles", self.SUCCESS_COLOR)
                    except:
                        self._show_snackbar("❌ No se pudo copiar", self.ERROR_COLOR)
            
            # Get file size
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.2f} MB"
            elif file_size > 1024:
                size_str = f"{file_size / 1024:.2f} KB"
            else:
                size_str = f"{file_size} B"
            
            # Create dialog content
            dialog_content = ft.Container(
                content=ft.Column([
                    # Header with success icon
                    ft.Row([
                        ft.Icon(ft.Icons.CHECK_CIRCLE, color=self.SUCCESS_COLOR, size=32),
                        ft.Text("¡Descargado Exitosamente!", size=18, weight=ft.FontWeight.BOLD),
                    ], spacing=15, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    
                    ft.Container(height=20),
                    
                    # File info
                    ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Text("📄 Archivo:", size=11, weight=ft.FontWeight.BOLD, width=80),
                                ft.Text(filename, size=11, color=self.TEXT_SECONDARY),
                            ]),
                            ft.Container(height=8),
                            ft.Row([
                                ft.Text("💾 Tamaño:", size=11, weight=ft.FontWeight.BOLD, width=80),
                                ft.Text(size_str, size=11, color=self.TEXT_SECONDARY),
                            ]),
                            ft.Container(height=8),
                            ft.Row([
                                ft.Text("📍 Ubicación:", size=11, weight=ft.FontWeight.BOLD, width=80),
                                ft.Text(file_path, size=9, color=self.SECONDARY_COLOR, italic=True),
                            ]),
                        ], spacing=5),
                        bgcolor=self.BACKGROUND_LIGHT,
                        padding=15,
                        border_radius=8,
                        border=ft.border.all(1, ft.Colors.with_opacity(0.2, self.SUCCESS_COLOR))
                    ),
                    
                    ft.Container(height=20),
                    
                    # Action buttons
                    ft.Row([
                        ft.ElevatedButton(
                            "Abrir Carpeta",
                            icon=ft.Icons.FOLDER_OPEN,
                            on_click=open_folder,
                            style=ft.ButtonStyle(
                                bgcolor=self.PRIMARY_COLOR,
                                color=self.BACKGROUND_DARK
                            ),
                            expand=True
                        ),
                        ft.ElevatedButton(
                            "Copiar Ruta",
                            icon=ft.Icons.COPY,
                            on_click=copy_path,
                            style=ft.ButtonStyle(
                                bgcolor=self.SECONDARY_COLOR,
                                color=self.BACKGROUND_DARK
                            ),
                            expand=True
                        ),
                        ft.ElevatedButton(
                            "Cerrar",
                            icon=ft.Icons.CLOSE,
                            on_click=close_dialog,
                            style=ft.ButtonStyle(
                                bgcolor=self.SURFACE_COLOR,
                                color=self.TEXT_PRIMARY
                            ),
                            expand=True
                        ),
                    ], spacing=10),
                ], spacing=10),
                bgcolor=self.SURFACE_COLOR,
                padding=25,
                border_radius=12,
                width=550,
                shadow=ft.BoxShadow(blur_radius=20, color="000000")
            )
            
            # Create overlay
            overlay = ft.Container(
                content=ft.Column([
                    ft.Container(expand=True),
                    ft.Row([
                        ft.Container(expand=True),
                        dialog_content,
                        ft.Container(expand=True)
                    ]),
                    ft.Container(expand=True)
                ]),
                bgcolor=ft.Colors.with_opacity(0.5, ft.Colors.BLACK),
                alignment=ft.alignment.center,
                expand=True,
                on_click=close_dialog
            )
            overlay.data = 'download_success_dialog'
            
            self.page.overlay.append(overlay)
            self.page.update()
            
        except Exception as ex:
            self._show_snackbar(f"❌ Error: {str(ex)}", self.ERROR_COLOR)

    def _copy_logs_to_clipboard(self, e):
        """Copy all logs to clipboard."""
        try:
            # Collect all log messages
            logs_text = ""
            for log_entry in self.logs_display.controls:
                if isinstance(log_entry, ft.Text):
                    logs_text += log_entry.value + "\n"
            
            if not logs_text.strip():
                self._show_snackbar("⚠️ No hay logs para copiar", self.ERROR_COLOR)
                return
            
            # Copy to clipboard using pyperclip or built-in method
            try:
                import subprocess
                process = subprocess.Popen(['clip'], stdin=subprocess.PIPE)
                process.communicate(logs_text.encode('utf-8'))
                self._show_snackbar(f"✅ {len(self.logs_display.controls)} líneas de logs copiadas", self.SUCCESS_COLOR)
            except:
                # Fallback: try using tkinter
                try:
                    import tkinter as tk
                    root = tk.Tk()
                    root.withdraw()
                    root.clipboard_clear()
                    root.clipboard_append(logs_text)
                    root.update()
                    root.destroy()
                    self._show_snackbar(f"✅ {len(self.logs_display.controls)} líneas de logs copiadas", self.SUCCESS_COLOR)
                except:
                    self._show_snackbar("❌ No se pudo copiar al portapapeles", self.ERROR_COLOR)
            
        except Exception as ex:
            self._show_snackbar(f"❌ Error: {str(ex)}", self.ERROR_COLOR)

