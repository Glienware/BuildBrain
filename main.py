#!/usr/bin/env python3
"""
AI/ML Model Trainer Desktop App

Main entry point for the application.
"""

import flet as ft
from src.gui.welcome_screen import WelcomeScreen
from src.gui.main_window import MainWindow


class AppController:
    """
    Controls navigation between welcome screen and main application.
    """

    def __init__(self, page: ft.Page):
        self.page = page
        self.current_view = None
        self.main_window = None
        self.welcome_screen = None

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

    def show_main_window(self, project_data=None):
        """Show the main application window."""
        if self.main_window is None:
            self.main_window = MainWindow(self.page, on_back_to_welcome=self.show_welcome_screen)

        # If opening existing project, load it
        if project_data:
            self.main_window.project_manager.load_project_data(project_data)
            # Rebuild the UI after loading project data
            self.current_view = self.main_window.build()
        else:
            self.current_view = self.main_window.build()
            
        self.page.controls.clear()
        self.page.add(self.current_view)
        self.page.update()

    def on_new_project(self, project_data):
        """Handle new project creation."""
        self.show_main_window(project_data)

    def on_project_created(self, project_data):
        """Handle project creation with data."""
        self.show_main_window(project_data)

    def on_open_project(self, project_data):
        """Handle opening existing project."""
        self.show_main_window(project_data)


def main(page: ft.Page):
    """
    Main function to set up the Flet page and start the application.
    """
    # Configure page for Android Studio-like dark theme
    page.title = "AI/ML Trainer"
    page.theme_mode = ft.ThemeMode.DARK
    page.theme = ft.Theme(
        color_scheme=ft.ColorScheme(
            primary=ft.Colors.GREEN_400,  # #3DDC84 equivalent
            secondary=ft.Colors.BLUE_400,
            surface=ft.Colors.GREY_800,
            background=ft.Colors.GREY_900,  # #1E1E1E equivalent
            on_primary=ft.Colors.BLACK,
            on_secondary=ft.Colors.BLACK,
            on_surface=ft.Colors.WHITE,
            on_background=ft.Colors.WHITE,
        )
    )
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