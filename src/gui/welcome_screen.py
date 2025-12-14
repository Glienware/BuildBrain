"""
Welcome Screen for AI/ML Trainer App

Similar to Android Studio welcome screen with options for new and existing projects.
"""

import flet as ft
import os
import json
import shutil
import sys
from datetime import datetime
from .new_project_wizard import NewProjectWizard


class WelcomeScreen:
    """
    Welcome screen with options to create new model, open existing projects, or launch the AI agent workspace.
    """

    def __init__(self, page: ft.Page, on_new_project, on_open_project, on_project_created=None, on_agent_builder=None):
        self.page = page
        self.on_new_project = on_new_project
        self.on_open_project = on_open_project
        self.on_project_created = on_project_created
        self.on_agent_builder = on_agent_builder
        # Use persistent directory for compiled executables
        if getattr(sys, 'frozen', False):
            app_data = os.getenv('LOCALAPPDATA')
            self.projects_dir = os.path.join(app_data, 'BuildBrain', 'projects')
        else:
            self.projects_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "projects")
        os.makedirs(self.projects_dir, exist_ok=True)
        self.recent_projects = self.load_recent_projects()

    def load_recent_projects(self):
        """Load recent projects from projects directory."""
        projects = []
        if os.path.exists(self.projects_dir):
            for item in os.listdir(self.projects_dir):
                item_path = os.path.join(self.projects_dir, item)
                if os.path.isdir(item_path):
                    # First try to find the file with the folder name
                    buildb_file = os.path.join(item_path, f"{item}.buildb")
                    
                    # If not found, search for any .buildb file in the directory
                    if not os.path.exists(buildb_file):
                        for file in os.listdir(item_path):
                            if file.endswith('.buildb'):
                                buildb_file = os.path.join(item_path, file)
                                break
                    
                    if os.path.exists(buildb_file):
                        try:
                            with open(buildb_file, 'r') as f:
                                project_data = json.load(f)
                                project_data['filename'] = os.path.basename(buildb_file)
                                project_data['path'] = item_path
                                projects.append(project_data)
                        except:
                            pass

        # Sort by last modified date
        projects.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
        return projects[:5]  # Show only 5 most recent

    def build(self):
        """Build the welcome screen layout with premium design."""
        # Try to load logo
        logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "logo.png")
        logo_widget = None
        
        if os.path.exists(logo_path):
            try:
                logo_widget = ft.Image(
                    src=logo_path,
                    width=60,
                    height=60,
                    fit=ft.ImageFit.CONTAIN,
                )
            except Exception as e:
                print(f"Error loading logo: {e}")
                logo_widget = ft.Icon(ft.Icons.PSYCHOLOGY, size=60, color="#3DDC84")
        else:
            logo_widget = ft.Icon(ft.Icons.PSYCHOLOGY, size=60, color="#3DDC84")
        
        return ft.Container(
            content=ft.Column([
                # Premium Header with Logo
                ft.Container(
                    content=ft.Row([
                        logo_widget,
                        ft.Container(width=20),
                        ft.Column([
                            ft.Text("BuildBrain", size=28, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                            ft.Container(height=4),
                            ft.Text("Crea y entrena modelos de aprendizaje automático profesionalmente", size=13, color="#AAAAAA"),
                        ], spacing=0, expand=True),
                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                    bgcolor="#0D0D0D",
                    padding=ft.padding.symmetric(horizontal=32, vertical=20),
                    border=ft.border.only(bottom=ft.border.BorderSide(1.5, "#2D2D2D")),
                ),

                # Main content area with scroll
                ft.Container(
                    content=ft.Column([
                        # Get Started Section
                        ft.Column([
                            ft.Text("Empezar", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                            ft.Container(height=8),
                            ft.Text("Elige una opción para comenzar", size=15, color="#AAAAAA"),
                        ], spacing=0),

                        ft.Container(height=32),

                        # Action Cards Row
                        ft.Row([
                            # New Project Card
                            ft.Container(
                                content=ft.Column([
                                    ft.Row([
                                        ft.Icon(ft.Icons.ADD_CIRCLE, size=28, color="#3DDC84"),
                                        ft.Container(width=12),
                                        ft.Text("Crear Nuevo Modelo", size=16, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                                    ft.Container(height=12),
                                    ft.Text(
                                        "Inicia un nuevo proyecto de aprendizaje automático con la guía paso a paso",
                                        size=12,
                                        color="#AAAAAA",
                                    ),
                                    ft.Container(height=16),
                                    ft.ElevatedButton(
                                        "Crear Modelo",
                                        icon=ft.Icons.ARROW_FORWARD,
                                        on_click=self.show_new_project_wizard,
                                        style=ft.ButtonStyle(
                                            bgcolor="#3DDC84",
                                            color=ft.Colors.BLACK,
                                            padding=ft.padding.symmetric(horizontal=24, vertical=12),
                                            shape=ft.RoundedRectangleBorder(radius=8),
                                        ),
                                    ),
                                ], spacing=0),
                                padding=ft.padding.all(24),
                                bgcolor="#1A1A1A",
                                border=ft.border.all(1.5, "#2D2D2D"),
                                border_radius=12,
                                expand=True,
                            ),

                            ft.Container(width=20),

                            # Open Project Card
                            ft.Container(
                                content=ft.Column([
                                    ft.Row([
                                        ft.Icon(ft.Icons.FOLDER_SPECIAL, size=28, color="#82B1FF"),
                                        ft.Container(width=12),
                                        ft.Text("Abrir Modelo Existente", size=16, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                                    ft.Container(height=12),
                                    ft.Text(
                                        "Continúa trabajando en un proyecto que ya habías iniciado",
                                        size=12,
                                        color="#AAAAAA",
                                    ),
                                    ft.Container(height=16),
                                    ft.ElevatedButton(
                                        "Abrir Modelo",
                                        icon=ft.Icons.FOLDER,
                                        on_click=self.show_project_selector,
                                        style=ft.ButtonStyle(
                                            bgcolor="#82B1FF",
                                            color=ft.Colors.BLACK,
                                            padding=ft.padding.symmetric(horizontal=24, vertical=12),
                                            shape=ft.RoundedRectangleBorder(radius=8),
                                        ),
                                    ),
                                ], spacing=0),
                                padding=ft.padding.all(24),
                                bgcolor="#1A1A1A",
                                border=ft.border.all(1.5, "#2D2D2D"),
                                border_radius=12,
                                expand=True,
                            ),

                            ft.Container(width=20),

                            # AI Agent Card
                            ft.Container(
                                content=ft.Column([
                                    ft.Row([
                                        ft.Icon(ft.Icons.ANDROID, size=28, color="#FF6D00"),
                                        ft.Container(width=12),
                                        ft.Text("Agente de IA", size=16, weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
                                    ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                                    ft.Container(height=12),
                                    ft.Text(
                                        "Crea agentes inteligentes con nodos personalizados en un área dedicada",
                                        size=12,
                                        color="#AAAAAA",
                                    ),
                                    ft.Container(height=16),
                                    ft.ElevatedButton(
                                        "Crear Agente",
                                        icon=ft.Icons.ANDROID,
                                        on_click=self.show_agent_builder,
                                        style=ft.ButtonStyle(
                                            bgcolor="#FF6D00",
                                            color=ft.Colors.BLACK,
                                            padding=ft.padding.symmetric(horizontal=24, vertical=12),
                                            shape=ft.RoundedRectangleBorder(radius=8),
                                        ),
                                    ),
                                ], spacing=0),
                                padding=ft.padding.all(24),
                                bgcolor="#1A1A1A",
                                border=ft.border.all(1.5, "#2D2D2D"),
                                border_radius=12,
                                expand=True,
                            ),
                        ], spacing=0, expand=True),

                        ft.Container(height=40),

                        # Recent Projects Section
                        ft.Column([
                            ft.Text("Modelos Recientes", size=32, weight=ft.FontWeight.W_700, color=ft.Colors.WHITE),
                            ft.Container(height=8),
                            ft.Text("Abre rápidamente uno de tus proyectos recientes", size=15, color="#AAAAAA"),
                        ], spacing=0),

                        ft.Container(height=24),

                        # Recent Projects Grid with Scroll
                        ft.Container(
                            content=ft.Column([
                                self.build_recent_projects_grid(),
                            ], spacing=0, expand=True, scroll=ft.ScrollMode.AUTO),
                            bgcolor="#0D0D0D",
                            border=ft.border.all(1.5, "#2D2D2D"),
                            border_radius=12,
                            padding=ft.padding.all(16),
                            height=400,
                        ),

                        ft.Container(height=32),

                    ], spacing=0, scroll=ft.ScrollMode.AUTO),
                    padding=ft.padding.symmetric(horizontal=32, vertical=24),
                    expand=True,
                ),

            ], spacing=0),
            bgcolor="#0D0D0D",
            expand=True,
        )

    def build_recent_projects_list(self):
        """Build the list of recent projects."""
        if not self.recent_projects:
            return ft.Column([
                ft.Icon(ft.Icons.INBOX, size=48, color=ft.Colors.GREY_400),
                ft.Text("No recent models", size=16, color=ft.Colors.GREY_500),
                ft.Text("Create your first model to get started", size=14, color=ft.Colors.GREY_400),
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10)

        project_cards = []
        for project in self.recent_projects:
            project_cards.append(
                ft.Card(
                    content=ft.Container(
                        content=ft.Row([
                            ft.Column([
                                ft.Text(project.get('project_name', 'Unnamed Project'), size=16, weight=ft.FontWeight.BOLD),
                                ft.Text(f"Model: {project.get('model_type', 'Unknown')}", size=12, color=ft.Colors.GREY_600),
                                ft.Text(f"Task: {project.get('task_type', 'Unknown')}", size=12, color=ft.Colors.GREY_600),
                                ft.Text(f"Modified: {project.get('last_modified', 'Unknown')}", size=10, color=ft.Colors.GREY_500),
                            ], spacing=2, expand=True),
                            ft.IconButton(
                                ft.Icons.PLAY_ARROW,
                                tooltip="Open Project",
                                on_click=lambda e, p=project: self.open_project(p)
                            ),
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        padding=15,
                    ),
                    elevation=2,
                )
            )

        return ft.Column(project_cards, spacing=10, scroll=ft.ScrollMode.AUTO)

    def build_recent_projects_grid(self):
        """Build the recent projects grid with professional styling."""
        if not self.recent_projects:
            return ft.Container(
                content=ft.Column([
                    ft.Icon(ft.Icons.FOLDER_OPEN, size=56, color="#666666"),
                    ft.Container(height=16),
                    ft.Text("Sin modelos recientes", size=16, color="#888888", weight=ft.FontWeight.W_600),
                    ft.Container(height=8),
                    ft.Text("Crea un nuevo modelo para empezar", size=13, color="#666666"),
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=0),
                alignment=ft.alignment.center,
                height=200,
            )

        # Create grid rows (2 columns)
        grid_rows = []
        for i in range(0, len(self.recent_projects), 2):
            row_projects = self.recent_projects[i:i+2]
            cards = []
            
            for project in row_projects:
                project_name = project.get('project_name', 'Sin nombre')
                model_type = project.get('model_type', 'Sin definir')
                task_type = project.get('task_type', 'Sin especificar')
                last_modified = project.get('last_modified', 'Desconocido')
                
                card = ft.Container(
                    content=ft.Column([
                        # Header with icon
                        ft.Row([
                            ft.Icon(ft.Icons.FOLDER_SPECIAL, size=24, color="#3DDC84"),
                            ft.Container(width=12),
                            ft.Column([
                                ft.Text(
                                    project_name,
                                    size=15,
                                    weight=ft.FontWeight.W_600,
                                    color=ft.Colors.WHITE,
                                    overflow=ft.TextOverflow.ELLIPSIS,
                                    max_lines=1,
                                ),
                                ft.Text(
                                    f"Actualizado: {last_modified}",
                                    size=11,
                                    color="#888888",
                                ),
                            ], spacing=0, expand=True),
                        ], spacing=0, vertical_alignment=ft.CrossAxisAlignment.START),
                        
                        ft.Container(height=12),
                        
                        # Project info
                        ft.Column([
                            ft.Row([
                                ft.Icon(ft.Icons.DATASET, size=14, color="#82B1FF"),
                                ft.Container(width=8),
                                ft.Text(f"Modelo: {model_type}", size=11, color="#AAAAAA"),
                            ], spacing=0),
                            ft.Container(height=6),
                            ft.Row([
                                ft.Icon(ft.Icons.MODEL_TRAINING, size=14, color="#FFB74D"),
                                ft.Container(width=8),
                                ft.Text(f"Tarea: {task_type}", size=11, color="#AAAAAA"),
                            ], spacing=0),
                        ], spacing=0),
                        
                        ft.Container(height=16),
                        
                        # Action buttons
                        ft.Row([
                            ft.ElevatedButton(
                                "Abrir",
                                icon=ft.Icons.FOLDER_OPEN,
                                style=ft.ButtonStyle(
                                    bgcolor="#3DDC84",
                                    color=ft.Colors.BLACK,
                                    padding=ft.padding.symmetric(horizontal=20, vertical=10),
                                    shape=ft.RoundedRectangleBorder(radius=6),
                                ),
                                on_click=lambda e, p=project: self.open_project(p),
                                expand=True,
                            ),
                            ft.Container(width=8),
                            ft.IconButton(
                                icon=ft.Icons.DELETE,
                                icon_color="#FF6B6B",
                                tooltip="Eliminar proyecto",
                                on_click=self._create_delete_handler(project),
                            ),
                        ], spacing=0),
                    ], spacing=0),
                    padding=ft.padding.all(20),
                    bgcolor="#1A1A1A",
                    border=ft.border.all(1.5, "#2D2D2D"),
                    border_radius=12,
                    expand=True,
                    on_hover=lambda e: self._on_project_card_hover(e),
                )
                cards.append(card)
            
            # Fill empty space if odd number
            if len(cards) == 1:
                cards.append(ft.Container(expand=True))
            
            grid_rows.append(ft.Row(cards, spacing=16, expand=True))

        return ft.Column(
            grid_rows,
            spacing=16,
            expand=True,
        )

    def _on_project_card_hover(self, e):
        """Handle project card hover effect."""
        if e.data == "true":
            e.control.border = ft.border.all(1.5, "#3DDC84")
        else:
            e.control.border = ft.border.all(1.5, "#2D2D2D")
        e.control.update()

    def _create_delete_handler(self, project):
        """Create a delete handler for a specific project."""
        def handler(e):
            print(f"DEBUG: Delete button clicked for project: {project.get('project_name')}")
            self.delete_project(project)
        return handler

    def delete_project(self, project):
        """Delete a project immediately with confirmation via snackbar."""
        print(f"DEBUG: delete_project called for {project.get('project_name')}")
        
        project_name = project.get('project_name', 'Sin nombre')
        project_path = project.get('path')
        
        print(f"DEBUG: Project path: {project_path}")
        print(f"DEBUG: Project path exists: {os.path.exists(project_path) if project_path else 'No path'}")
        
        try:
            if project_path and os.path.exists(project_path):
                print(f"DEBUG: Deleting folder: {project_path}")
                shutil.rmtree(project_path)
                print(f"DEBUG: Folder deleted successfully")
                
                # Show success message
                snack = ft.SnackBar(
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.CHECK_CIRCLE, color="#3DDC84"),
                            ft.Text(f"✓ Proyecto '{project_name}' eliminado exitosamente", color=ft.Colors.WHITE),
                        ], spacing=10),
                        padding=10,
                    ),
                    bgcolor="#1A1A1A",
                    duration=3000,
                )
                self.page.snack_bar = snack
                snack.open = True
                
                # Reload recent projects
                self.recent_projects = self.load_recent_projects()
                print(f"DEBUG: Recent projects reloaded: {len(self.recent_projects)} projects")
                
                # Refresh welcome screen
                print(f"DEBUG: Refreshing welcome screen")
                import time
                time.sleep(0.5)  # Brief delay to show the snackbar
                self.show_welcome_screen()
            else:
                print(f"DEBUG: Path not found or doesn't exist")
                snack = ft.SnackBar(
                    ft.Text(f"Error: No se encontró la carpeta del proyecto", color=ft.Colors.WHITE),
                    bgcolor="#FF6B6B",
                    duration=3000,
                )
                self.page.snack_bar = snack
                snack.open = True
                self.page.update()
                
        except Exception as ex:
            print(f"Error deleting project: {ex}")
            import traceback
            traceback.print_exc()
            snack = ft.SnackBar(
                ft.Text(f"Error al eliminar proyecto: {str(ex)}", color=ft.Colors.WHITE),
                bgcolor="#FF6B6B",
                duration=3000,
            )
            self.page.snack_bar = snack
            snack.open = True
            self.page.update()

    def open_project(self, project_data):
        """Open an existing project."""
        if self.on_open_project:
            self.on_open_project(project_data)

    def show_agent_builder(self, e=None):
        """Navigate to the AI agent builder section."""
        if self.on_agent_builder:
            self.on_agent_builder(e)

    def show_project_selector(self, e):
        """Show project selection dialog with all projects."""
        def close_dialog(e):
            dlg.open = False
            self.page.update()

        def select_project(project):
            close_dialog(None)
            self.open_project(project)

        # Load all projects (not just recent)
        all_projects = []
        if os.path.exists(self.projects_dir):
            for item in os.listdir(self.projects_dir):
                item_path = os.path.join(self.projects_dir, item)
                if os.path.isdir(item_path):
                    buildb_file = os.path.join(item_path, f"{item}.buildb")
                    if not os.path.exists(buildb_file):
                        for file in os.listdir(item_path):
                            if file.endswith('.buildb'):
                                buildb_file = os.path.join(item_path, file)
                                break
                    
                    if os.path.exists(buildb_file):
                        try:
                            with open(buildb_file, 'r') as f:
                                project_data = json.load(f)
                                project_data['filename'] = os.path.basename(buildb_file)
                                project_data['path'] = item_path
                                all_projects.append(project_data)
                        except:
                            pass

        # Create project selection options
        project_options = []
        for project in all_projects:
            project_options.append(
                ft.ListTile(
                    title=ft.Text(project.get('project_name', 'Proyecto sin nombre')),
                    subtitle=ft.Text(f"Modelo: {project.get('model_type', 'Desconocido')} | Tarea: {project.get('task_type', 'Desconocida')}"),
                    trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT),
                    on_click=lambda e, p=project: select_project(p)
                )
            )

        if not project_options:
            project_options = [ft.ListTile(title=ft.Text("No hay proyectos disponibles"))]

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Seleccionar Proyecto"),
            content=ft.Container(
                content=ft.Column(project_options, scroll=ft.ScrollMode.AUTO),
                height=400,
                width=600,
            ),
            actions=[
                ft.TextButton("Cancelar", on_click=close_dialog),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.dialog = dlg
        dlg.open = True
        self.page.update()

    def show_new_project_wizard(self, e):
        """Show the new project wizard."""
        wizard = NewProjectWizard(
            self.page,
            on_complete=self.on_project_created or self.on_new_project,
            on_cancel=self.show_welcome_screen
        )
        self.page.clean()
        self.page.add(wizard.build())
        self.page.update()

    def show_welcome_screen(self, e=None):
        """Show the welcome screen again."""
        # Refresh recent projects list
        self.recent_projects = self.load_recent_projects()
        self.page.clean()
        self.page.add(self.build())
        self.page.update()

    def on_project_created(self, project_data):
        """Handle project creation completion."""
        if self.on_project_created:
            self.on_project_created(project_data)
        else:
            self.on_new_project(project_data)