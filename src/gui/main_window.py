"""
Main Window - Android Studio Style Interface

Complete UI implementation with modern dark theme and tabbed interface.
"""

import flet as ft
import os
import time
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
        # App Bar
        self.app_bar = self._create_app_bar()

        # Tab Bar
        self.tabs = self._create_tabs()

        # Status Bar
        self.status_bar = self._create_status_bar()

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
                        ]),
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

        # Add class dialog
        self.add_class_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Agregar Nueva Clase"),
            content=ft.TextField(label="Nombre de la clase", autofocus=True),
            actions=[
                ft.TextButton("Cancelar", on_click=self._close_add_class_dialog),
                ft.ElevatedButton("Agregar", on_click=self._add_class),
            ],
        )

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
                            on_click=self._show_add_class_dialog,
                            style=ft.ButtonStyle(bgcolor=self.SECONDARY_COLOR)
                        ),
                    ]),
                    margin=ft.margin.only(bottom=20)
                ),

                # Classes list
                self.classes_container,

            ], scroll=ft.ScrollMode.AUTO, spacing=20),
            padding=20
        )

    def _create_training_tab(self) -> ft.Container:
        """Create training tab content."""
        # Logs display
        self.logs_display = ft.Column(
            height=300,
            scroll=ft.ScrollMode.AUTO,
            spacing=2
        )

        # Progress indicators
        self.progress_bar = ft.ProgressBar(width=400, color=self.PRIMARY_COLOR, visible=False)
        self.progress_text = ft.Text("", size=14, color=self.TEXT_SECONDARY, visible=False)

        # Training buttons
        quick_train_btn = ft.ElevatedButton(
            "Entrenamiento Rápido",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._start_quick_training,
            style=ft.ButtonStyle(bgcolor=self.SUCCESS_COLOR),
            disabled=self.training_active
        )

        advanced_train_btn = ft.ElevatedButton(
            "Entrenamiento Avanzado",
            icon=ft.Icons.PLAY_CIRCLE_FILLED,
            on_click=self._start_advanced_training,
            style=ft.ButtonStyle(bgcolor=self.PRIMARY_COLOR),
            disabled=self.training_active
        )

        stop_train_btn = ft.ElevatedButton(
            "Detener Entrenamiento",
            icon=ft.Icons.STOP,
            on_click=self._stop_training,
            style=ft.ButtonStyle(bgcolor=self.ERROR_COLOR),
            visible=self.training_active
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

                # Logs console
                ft.Container(
                    content=ft.Column([
                        ft.Row([
                            ft.Icon(ft.Icons.TERMINAL, color=self.TEXT_SECONDARY),
                            ft.Text("Consola de Logs", size=16, weight=ft.FontWeight.BOLD),
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

        classes_info = self.dataset_manager.get_classes_info()

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
                            on_click=lambda e, name=class_info["name"]: self._upload_images_to_class(name),
                            style=ft.ButtonStyle(bgcolor=self.SECONDARY_COLOR)
                        ),
                        ft.ElevatedButton(
                            "Ver Imágenes",
                            icon=ft.Icons.VISIBILITY,
                            on_click=lambda e, name=class_info["name"]: self._view_class_images(name),
                        ),
                        ft.IconButton(
                            ft.Icons.DELETE,
                            icon_color=self.ERROR_COLOR,
                            tooltip="Eliminar clase",
                            on_click=lambda e, name=class_info["name"]: self._delete_class(name)
                        ),
                    ], spacing=10),
                ]),
                padding=20
            ),
            elevation=2
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
            # Update config values
            self.project_config.update_setting("project_name", self.config_fields["project_name"].value)
            self.project_config.update_setting("model_type", self.config_fields["model_type"].value)
            self.project_config.update_setting("epochs", int(self.config_fields["epochs"].value))
            self.project_config.update_setting("batch_size", int(self.config_fields["batch_size"].value))
            self.project_config.update_setting("learning_rate", float(self.config_fields["learning_rate"].value))
            self.project_config.update_setting("optimizer", self.config_fields["optimizer"].value)

            # Save to file
            if self.project_config.save_config():
                self._show_snackbar("✅ Configuración guardada exitosamente", self.SUCCESS_COLOR)
                # Update status bar
                self._update_status_bar()
            else:
                self._show_snackbar("❌ Error al guardar la configuración", self.ERROR_COLOR)

        except ValueError as ve:
            self._show_snackbar(f"❌ Error en los valores: {str(ve)}", self.ERROR_COLOR)
        except Exception as ex:
            self._show_snackbar(f"❌ Error inesperado: {str(ex)}", self.ERROR_COLOR)

    def _show_add_class_dialog(self, e):
        """Show add class dialog."""
        self.add_class_dialog.content.value = ""  # Clear previous value
        self.page.dialog = self.add_class_dialog
        self.add_class_dialog.open = True
        self.page.update()

    def _close_add_class_dialog(self, e):
        """Close add class dialog."""
        self.add_class_dialog.open = False
        self.page.update()

    def _add_class(self, e):
        """Add a new class."""
        class_name = self.add_class_dialog.content.value.strip()
        if class_name:
            if self.dataset_manager.add_class(class_name):
                self.project_config.add_class(class_name)
                self.project_config.save_config()
                self._update_classes_display()
                self._show_snackbar(f"✅ Clase '{class_name}' agregada", self.SUCCESS_COLOR)
            else:
                self._show_snackbar(f"❌ La clase '{class_name}' ya existe", self.ERROR_COLOR)

        self._close_add_class_dialog(None)

    def _delete_class(self, class_name: str):
        """Delete a class."""
        def confirm_delete(e):
            if self.dataset_manager.remove_class(class_name):
                self.project_config.remove_class(class_name)
                self.project_config.save_config()
                self._update_classes_display()
                self._show_snackbar(f"✅ Clase '{class_name}' eliminada", self.SUCCESS_COLOR)
            else:
                self._show_snackbar(f"❌ Error al eliminar clase '{class_name}'", self.ERROR_COLOR)
            confirm_dialog.open = False
            self.page.update()

        confirm_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Confirmar eliminación"),
            content=ft.Text(f"¿Estás seguro de que quieres eliminar la clase '{class_name}' y todas sus imágenes?"),
            actions=[
                ft.TextButton("Cancelar", on_click=lambda e: setattr(confirm_dialog, 'open', False) or self.page.update()),
                ft.ElevatedButton("Eliminar", on_click=confirm_delete, style=ft.ButtonStyle(bgcolor=self.ERROR_COLOR)),
            ],
        )

        self.page.dialog = confirm_dialog
        confirm_dialog.open = True
        self.page.update()

    def _upload_images_to_class(self, class_name: str):
        """Upload images to a class."""
        def pick_files_result(e: ft.FilePickerResultEvent):
            if e.files:
                file_paths = [f.path for f in e.files]
                success_count, error_count = self.dataset_manager.add_images_to_class(class_name, file_paths)

                if success_count > 0:
                    self._update_classes_display()
                    self._show_snackbar(f"✅ {success_count} imágenes agregadas a '{class_name}'", self.SUCCESS_COLOR)
                if error_count > 0:
                    self._show_snackbar(f"⚠️ {error_count} archivos no válidos ignorados", self.ERROR_COLOR)

        file_picker = ft.FilePicker(on_result=pick_files_result)
        self.page.overlay.append(file_picker)
        self.page.update()
        file_picker.pick_files(allow_multiple=True, allowed_extensions=["jpg", "jpeg", "png", "bmp"])

    def _view_class_images(self, class_name: str):
        """View images in a class."""
        images = self.dataset_manager.get_class_images(class_name, limit=20)

        if not images:
            self._show_snackbar(f"📭 No hay imágenes en la clase '{class_name}'", self.SECONDARY_COLOR)
            return

        # Create image gallery dialog
        image_rows = []
        for i in range(0, len(images), 4):  # 4 images per row
            row_images = images[i:i+4]
            row = ft.Row(
                [ft.Image(src=img["path"], width=100, height=100, fit=ft.ImageFit.COVER)
                 for img in row_images],
                spacing=10
            )
            image_rows.append(row)

        gallery_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(f"Imágenes en '{class_name}' ({len(images)})"),
            content=ft.Container(
                content=ft.Column(image_rows, scroll=ft.ScrollMode.AUTO, height=400),
                width=500
            ),
            actions=[
                ft.TextButton("Cerrar", on_click=lambda e: setattr(gallery_dialog, 'open', False) or self.page.update()),
            ],
        )

        self.page.dialog = gallery_dialog
        gallery_dialog.open = True
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

    def _log_message(self, message: str):
        """Add a message to the logs display."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = ft.Text(f"[{timestamp}] {message}", size=12, color=self.TEXT_SECONDARY)

        self.logs_display.controls.append(log_entry)

        # Keep only last 100 logs
        if len(self.logs_display.controls) > 100:
            self.logs_display.controls.pop(0)

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
        """
        Main Window for the AI/ML Trainer App

        Handles the overall layout and coordination of GUI components.
        """

import flet as ft
from .model_selector import ModelSelector
from .settings_panel import SettingsPanel
from .task_selector import TaskSelector
from .class_manager import ClassManager
from .dataset_uploader import DatasetUploader
from .training_controls import TrainingControls
from .logs_visualization import LogsVisualization
from .project_manager import ProjectManager
from .project_list import ProjectList


class MainWindow:
    """
    Main window class that orchestrates all GUI components.
    """

    def __init__(self, page: ft.Page, on_back_to_welcome=None):
        self.page = page
        self.on_back_to_welcome = on_back_to_welcome
        self.model_selector = ModelSelector(page)
        self.settings_panel = SettingsPanel(page)
        self.task_selector = TaskSelector(page)
        self.class_manager = ClassManager(page)
        self.dataset_uploader = DatasetUploader(page, on_update=self._refresh_ui)
        self.logs_visualization = LogsVisualization(page)

        # Component references for project manager
        self.component_refs = {
            "task_selector": self.task_selector,
            "model_selector": self.model_selector,
            "settings_panel": self.settings_panel,
            "class_manager": self.class_manager,
            "dataset_uploader": self.dataset_uploader,
        }

        self.training_controls = TrainingControls(page, self.component_refs, self.logs_visualization.add_log)
        self.project_manager = ProjectManager(page, self.component_refs)
        self.project_list = ProjectList(page, self.project_manager)
        self.project_manager.project_list = self.project_list

    def _refresh_ui(self):
        """Refresh the entire UI after loading project data."""
        # Rebuild the main window content
        self.page.controls.clear()
        self.page.add(self.build())
        self.page.update()

    def build(self):
        """
        Build the main window layout.
        """
        return ft.Container(
            content=ft.Column([
                # App Bar
                ft.Container(
                    content=ft.Row([
                        ft.IconButton(
                            ft.Icons.ARROW_BACK,
                            tooltip="Back to Welcome Screen",
                            on_click=self.on_back_to_welcome if self.on_back_to_welcome else lambda e: None
                        ),
                        ft.Text("AI/ML Trainer", size=20, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),  # Spacer
                        ft.Icon(ft.Icons.PSYCHOLOGY, size=24, color=ft.Colors.BLUE_600),
                    ], alignment=ft.MainAxisAlignment.START),
                    bgcolor=ft.Colors.BLUE_50,
                    padding=ft.padding.symmetric(horizontal=20, vertical=10),
                    border_radius=ft.border_radius.only(bottom_left=10, bottom_right=10),
                ),

                # Main content - Simple placeholder
                ft.Container(
                    content=ft.Column([
                        ft.Container(height=100),  # Spacer
                        ft.Icon(ft.Icons.CONSTRUCTION, size=64, color=ft.Colors.GREY_400),
                        ft.Container(height=20),
                        ft.Text(
                            "Esta sección está en desarrollo",
                            size=24,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.GREY_600,
                            text_align=ft.TextAlign.CENTER
                        ),
                        ft.Container(height=10),
                        ft.Text(
                            "Pronto agregaremos funciones reales para configurar, gestionar datos y entrenar modelos.",
                            size=16,
                            color=ft.Colors.GREY_500,
                            text_align=ft.TextAlign.CENTER
                        ),
                        ft.Container(height=50),
                        ft.Text(
                            "🚧 Work in Progress 🚧",
                            size=18,
                            color=ft.Colors.ORANGE_400,
                            text_align=ft.TextAlign.CENTER
                        ),
                    ], 
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    alignment=ft.MainAxisAlignment.CENTER),
                    expand=True,
                    padding=40
                ),
            ]),
            expand=True,
            padding=10
        )