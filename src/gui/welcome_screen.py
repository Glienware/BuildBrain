"""
Welcome Screen for AI/ML Trainer App

Similar to Android Studio welcome screen with options for new and existing projects.
"""

import flet as ft
import os
import json
from datetime import datetime
from .new_project_wizard import NewProjectWizard


class WelcomeScreen:
    """
    Welcome screen with options to create new model or open existing project.
    """

    def __init__(self, page: ft.Page, on_new_project, on_open_project, on_project_created=None):
        self.page = page
        self.on_new_project = on_new_project
        self.on_open_project = on_open_project
        self.on_project_created = on_project_created
        self.projects_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "projects")
        self.recent_projects = self.load_recent_projects()

    def load_recent_projects(self):
        """Load recent projects from projects directory."""
        projects = []
        if os.path.exists(self.projects_dir):
            for filename in os.listdir(self.projects_dir):
                if filename.endswith('.buildb'):
                    try:
                        with open(os.path.join(self.projects_dir, filename), 'r') as f:
                            project_data = json.load(f)
                            project_data['filename'] = filename
                            projects.append(project_data)
                    except:
                        pass

        # Sort by last modified date
        projects.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
        return projects[:5]  # Show only 5 most recent

    def build(self):
        """Build the welcome screen layout."""
        return ft.Container(
            content=ft.Column([
                # Header
                ft.Container(
                    content=ft.Row([
                        ft.Icon(ft.Icons.PSYCHOLOGY, size=48, color=ft.Colors.GREEN_400),
                        ft.Column([
                            ft.Text("AI/ML Trainer", size=32, weight=ft.FontWeight.BOLD, color=ft.Colors.GREEN_400),
                            ft.Text("Build and train machine learning models", size=16, color=ft.Colors.GREY_300),
                        ], spacing=5),
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    padding=ft.padding.symmetric(vertical=40),
                ),

                # Main content area
                ft.Row([
                    # Left side - Actions
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Get started", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_200),
                            ft.Container(height=20),

                            # New Project Card
                            ft.Card(
                                content=ft.Container(
                                    content=ft.Column([
                                        ft.Row([
                                            ft.Icon(ft.Icons.ADD, size=32, color=ft.Colors.GREEN_400),
                                            ft.Column([
                                                ft.Text("New Model", size=18, weight=ft.FontWeight.BOLD),
                                                ft.Text("Create a new machine learning project", size=14, color=ft.Colors.GREY_400),
                                            ], spacing=5),
                                        ], alignment=ft.MainAxisAlignment.START),
                                        ft.Container(height=20),
                                        ft.ElevatedButton(
                                            "Create New Model",
                                            icon=ft.Icons.ARROW_FORWARD,
                                            style=ft.ButtonStyle(
                                                bgcolor=ft.Colors.GREEN_400,
                                                color=ft.Colors.BLACK,
                                            ),
                                            on_click=self.show_new_project_wizard
                                        ),
                                    ], spacing=15),
                                    padding=20,
                                ),
                                elevation=4,
                            ),

                            ft.Container(height=20),

                            # Open Project Card
                            ft.Card(
                                content=ft.Container(
                                    content=ft.Column([
                                        ft.Row([
                                            ft.Icon(ft.Icons.FOLDER_OPEN, size=32, color=ft.Colors.BLUE_400),
                                            ft.Column([
                                                ft.Text("Open Model", size=18, weight=ft.FontWeight.BOLD),
                                                ft.Text("Continue working on an existing project", size=14, color=ft.Colors.GREY_400),
                                            ], spacing=5),
                                        ], alignment=ft.MainAxisAlignment.START),
                                        ft.Container(height=20),
                                        ft.ElevatedButton(
                                            "Open Existing Model",
                                            icon=ft.Icons.FOLDER,
                                            style=ft.ButtonStyle(
                                                bgcolor=ft.Colors.BLUE_400,
                                                color=ft.Colors.BLACK,
                                            ),
                                            on_click=self.show_project_selector
                                        ),
                                    ], spacing=15),
                                    padding=20,
                                ),
                                elevation=4,
                            ),

                        ], spacing=10),
                        width=400,
                        padding=ft.padding.symmetric(horizontal=20),
                    ),

                    # Right side - Recent Projects
                    ft.Container(
                        content=ft.Column([
                            ft.Text("Recent Models", size=24, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_200),
                            ft.Container(height=20),

                            ft.Container(
                                content=self.build_recent_projects_list(),
                                height=400,
                                padding=10,
                            ),

                        ], spacing=10),
                        width=500,
                        padding=ft.padding.symmetric(horizontal=20),
                    ),

                ], alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.START),

                # Footer
                ft.Container(
                    content=ft.Row([
                        ft.Text("AI/ML Trainer v1.0", size=12, color=ft.Colors.GREY_500),
                        ft.Container(width=20),
                        ft.Text("Built with Flet & Python", size=12, color=ft.Colors.GREY_500),
                    ], alignment=ft.MainAxisAlignment.CENTER),
                    padding=ft.padding.symmetric(vertical=20),
                ),

            ], spacing=0),
            bgcolor=ft.Colors.GREY_900,
            height=self.page.height,
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

    def open_project(self, project_data):
        """Open an existing project."""
        if self.on_open_project:
            self.on_open_project(project_data)

    def show_project_selector(self, e):
        """Show project selection dialog."""
        def close_dialog(e):
            dlg.open = False
            self.page.update()

        def select_project(project):
            close_dialog(None)
            self.open_project(project)

        # Create project selection dialog
        project_options = []
        for project in self.recent_projects:
            project_options.append(
                ft.ListTile(
                    title=ft.Text(project.get('project_name', 'Unnamed Project')),
                    subtitle=ft.Text(f"Model: {project.get('model_type', 'Unknown')}"),
                    trailing=ft.Icon(ft.Icons.CHEVRON_RIGHT),
                    on_click=lambda e, p=project: select_project(p)
                )
            )

        if not project_options:
            project_options = [ft.ListTile(title=ft.Text("No projects found"))]

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Select Project"),
            content=ft.Container(
                content=ft.Column(project_options, scroll=ft.ScrollMode.AUTO),
                height=300,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=close_dialog),
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
        self.page.controls.clear()
        self.page.add(wizard.build())
        self.page.update()

    def show_welcome_screen(self, e=None):
        """Show the welcome screen again."""
        # Refresh recent projects list
        self.recent_projects = self.load_recent_projects()
        self.page.controls.clear()
        self.page.add(self.build())
        self.page.update()

    def on_project_created(self, project_data):
        """Handle project creation completion."""
        if self.on_project_created:
            self.on_project_created(project_data)
        else:
            self.on_new_project(project_data)