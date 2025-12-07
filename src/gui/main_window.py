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
        self.dataset_uploader = DatasetUploader(page)
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

    def build(self):
        """
        Build the main window layout.
        """
        setup_tab = ft.Container(
            content=ft.Column([
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.SETTINGS, size=28, color=ft.Colors.BLUE_600),
                        ft.Text("Project Configuration", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                    ], alignment=ft.MainAxisAlignment.START),
                    margin=ft.margin.only(bottom=20)
                ),

                # Project management
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.Icons.FOLDER, size=24, color=ft.Colors.BLUE_600),
                                ft.Text("Project Management", size=18, weight=ft.FontWeight.BOLD),
                            ], alignment=ft.MainAxisAlignment.START),
                            ft.Container(height=10),
                            self.project_manager.build(),
                        ], spacing=10),
                        padding=20
                    ),
                    elevation=4,
                    margin=ft.margin.only(bottom=20)
                ),

                # Configuration cards in a grid
                ft.Row([
                    ft.Container(
                        content=ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Icon(ft.Icons.SELECT_ALL, size=32, color=ft.Colors.GREEN_600),
                                    ft.Text("Task & Model", size=16, weight=ft.FontWeight.BOLD),
                                    ft.Container(height=10),
                                    self.task_selector.build(),
                                    ft.Container(height=10),
                                    self.model_selector.build(),
                                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                                padding=20
                            ),
                            elevation=4
                        ),
                        width=400
                    ),
                    ft.Container(
                        content=ft.Card(
                            content=ft.Container(
                                content=ft.Column([
                                    ft.Icon(ft.Icons.TUNE, size=32, color=ft.Colors.ORANGE_600),
                                    ft.Text("Training Settings", size=16, weight=ft.FontWeight.BOLD),
                                    ft.Container(height=10),
                                    self.settings_panel.build(),
                                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                                padding=20
                            ),
                            elevation=4
                        ),
                        width=400
                    ),
                ], alignment=ft.MainAxisAlignment.START, spacing=20),

                ft.Container(height=20),

                # Class management
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.Icons.CATEGORY, size=24, color=ft.Colors.PURPLE_600),
                                ft.Text("Class Management", size=18, weight=ft.FontWeight.BOLD),
                            ], alignment=ft.MainAxisAlignment.START),
                            ft.Container(height=10),
                            self.class_manager.build(),
                        ], spacing=10),
                        padding=20
                    ),
                    elevation=4
                ),

            ], scroll=ft.ScrollMode.AUTO, spacing=20),
            padding=20
        )

        data_tab = ft.Container(
            content=ft.Column([
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.DATA_USAGE, size=28, color=ft.Colors.GREEN_600),
                        ft.Text("Dataset Management", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                    ], alignment=ft.MainAxisAlignment.START),
                    margin=ft.margin.only(bottom=20)
                ),

                # Dataset uploader
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.Icons.CLOUD_UPLOAD, size=24, color=ft.Colors.BLUE_600),
                                ft.Text("Dataset Upload", size=18, weight=ft.FontWeight.BOLD),
                            ], alignment=ft.MainAxisAlignment.START),
                            ft.Container(height=10),
                            self.dataset_uploader.build(),
                        ], spacing=10),
                        padding=20
                    ),
                    elevation=4
                ),

            ], scroll=ft.ScrollMode.AUTO, spacing=20),
            padding=20
        )

        training_tab = ft.Container(
            content=ft.Column([
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.PLAY_ARROW, size=28, color=ft.Colors.RED_600),
                        ft.Text("Training & Monitoring", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                    ], alignment=ft.MainAxisAlignment.START),
                    margin=ft.margin.only(bottom=20)
                ),

                # Training controls
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.Icons.PLAY_CIRCLE, size=24, color=ft.Colors.GREEN_600),
                                ft.Text("Training Controls", size=18, weight=ft.FontWeight.BOLD),
                            ], alignment=ft.MainAxisAlignment.START),
                            ft.Container(height=10),
                            self.training_controls.build(),
                        ], spacing=10),
                        padding=20
                    ),
                    elevation=4,
                    margin=ft.margin.only(bottom=20)
                ),

                # Logs and visualization
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row([
                                ft.Icon(ft.Icons.ASSESSMENT, size=24, color=ft.Colors.ORANGE_600),
                                ft.Text("Logs & Visualization", size=18, weight=ft.FontWeight.BOLD),
                            ], alignment=ft.MainAxisAlignment.START),
                            ft.Container(height=10),
                            self.logs_visualization.build(),
                        ], spacing=10),
                        padding=20
                    ),
                    elevation=4
                ),

            ], scroll=ft.ScrollMode.AUTO, spacing=20),
            padding=20
        )

        tabs = ft.Tabs(
            tabs=[
                ft.Tab(
                    text="Setup",
                    icon=ft.Icons.SETTINGS,
                    content=setup_tab
                ),
                ft.Tab(
                    text="Data",
                    icon=ft.Icons.DATA_USAGE,
                    content=data_tab
                ),
                ft.Tab(
                    text="Training",
                    icon=ft.Icons.PLAY_ARROW,
                    content=training_tab
                ),
            ],
            expand=True
        )

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

                # Main content
                tabs,
            ]),
            expand=True,
            padding=10
        )