#!/usr/bin/env python3
"""
AI/ML Model Trainer Desktop App

Main entry point for the application.
"""

import flet as ft
import os
from src.gui.welcome_screen import WelcomeScreen
from src.gui.main_window import AndroidStyleMainWindow


class AppController:
    """
    Controls navigation between welcome screen and main application.
    """

    def __init__(self, page: ft.Page):
        self.page = page
        self.current_view = None
        self.main_window = None
        self.welcome_screen = None
        self.projects_dir = os.path.join(os.path.dirname(__file__), "projects")
        self.projects_dir = os.path.join(os.path.dirname(__file__), "projects")

    def show_welcome_screen(self, e=None):
        """Show the welcome screen."""
        if self.welcome_screen is None:
            self.welcome_screen = WelcomeScreen(
                self.page,
                on_new_project=self.on_new_project,
                on_open_project=self.on_open_project,
                on_project_created=self.on_project_created
            )
        self.current_view = self.welcome_screen.build()
        self.page.controls.clear()
        self.page.add(self.current_view)
        self.page.update()

    def show_main_window(self, project_name: str):
        """Show the main application window."""
        project_path = os.path.join(self.projects_dir, project_name)

        if self.main_window is None:
            self.main_window = AndroidStyleMainWindow(
                self.page,
                project_path,
                on_back_to_welcome=self.show_welcome_screen
            )

        self.current_view = self.main_window.build()
        self.page.controls.clear()
        self.page.add(self.current_view)
        self.page.update()

    def on_new_project(self, project_data):
        """Handle new project creation."""
        project_name = project_data.get("project_name", "new_project")
        self.show_main_window(project_name)

    def on_project_created(self, project_data):
        """Handle project creation with data."""
        project_name = project_data.get("project_name", "new_project")
        self.show_main_window(project_name)

    def on_open_project(self, project_data):
        """Handle opening existing project."""
        project_name = project_data.get("project_name", "unknown_project")
        self.show_main_window(project_name)


def main(page: ft.Page):
    """
    Main function to set up the Flet page and start the application.
    """
    # Configure page
    page.title = "BuildBrain - AI/ML Trainer"
    page.window_width = 1400
    page.window_height = 900
    page.window_resizable = True
    page.window_maximizable = True
    page.window_minimizable = True

    # Create app controller and show welcome screen
    app_controller = AppController(page)
    app_controller.show_welcome_screen()


if __name__ == "__main__":
    ft.app(target=main)