#!/usr/bin/env python3
"""
AI/ML Model Trainer Desktop App

Main entry point for the application.
"""

__version__ = "1.0.0"

import flet as ft
import os
import sys
from src.gui.welcome_screen import WelcomeScreen
from src.gui.main_window import AndroidStyleMainWindow


def get_projects_directory():
    """
    Get the persistent projects directory.
    When running as .exe, use AppData\\Local\\BuildBrain\\projects
    When running as script, use ./projects
    """
    if getattr(sys, 'frozen', False):
        # Running as compiled exe
        app_data = os.getenv('LOCALAPPDATA')
        if app_data:
            projects_dir = os.path.join(app_data, 'BuildBrain', 'projects')
            os.makedirs(projects_dir, exist_ok=True)
            return projects_dir
    # Running as script
    return os.path.join(os.path.dirname(__file__), "projects")


def set_window_icon_windows(icon_path):
    """Set window icon using ctypes for better Windows compatibility."""
    try:
        if sys.platform == 'win32':
            import ctypes
            # Load the icon
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                # Try to set icon using Windows API
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("BuildBrain")
                print(f"Window icon set via Windows API")
            return True
    except Exception as e:
        print(f"Could not set window icon via Windows API: {e}")
    return False


class AppController:
    """
    Controls navigation between welcome screen and main application.
    """

    def __init__(self, page: ft.Page):
        self.page = page
        self.current_view = None
        self.main_window = None
        self.welcome_screen = None
        self.projects_dir = get_projects_directory()

    def show_welcome_screen(self, e=None):
        """Show the welcome screen."""
        # Reload recent projects to catch any changes (like renamed projects)
        if self.welcome_screen is None:
            self.welcome_screen = WelcomeScreen(
                self.page,
                on_new_project=self.on_new_project,
                on_open_project=self.on_open_project,
                on_project_created=self.on_project_created
            )
        else:
            # Refresh the recent projects list
            self.welcome_screen.recent_projects = self.welcome_screen.load_recent_projects()
        
        # Reset main window so it doesn't retain old state
        self.main_window = None
        
        self.current_view = self.welcome_screen.build()
        self.page.controls.clear()
        self.page.add(self.current_view)
        self.page.update()

    def show_main_window(self, project_identifier: str):
        """Show the main application window. Can be a project name or full path."""
        # Check if it's already a full path
        if os.path.isabs(project_identifier) and os.path.isdir(project_identifier):
            project_path = project_identifier
        else:
            # It's a project name, construct the path
            project_path = os.path.join(self.projects_dir, project_identifier)

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
        # Use the path directly if available, otherwise construct it from project name
        if 'path' in project_data:
            project_path = project_data['path']
        else:
            project_name = project_data.get("project_name", "unknown_project")
            project_path = os.path.join(self.projects_dir, project_name)
        
        self.show_main_window(project_path)


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
    
    # Set window icon - try multiple methods
    logo_ico = os.path.join(os.path.dirname(__file__), "assets", "logo.ico")
    logo_png = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
    
    # Method 1: Try ICO file
    if os.path.exists(logo_ico):
        try:
            page.window_icon = logo_ico
            print(f"Window icon set from ICO: {logo_ico}")
        except Exception as e:
            print(f"Method 1 failed: {e}")
    
    # Method 2: Try PNG file
    if os.path.exists(logo_png):
        try:
            page.window_icon = logo_png
            print(f"Window icon set from PNG: {logo_png}")
        except Exception as e:
            print(f"Method 2 failed: {e}")
    
    # Method 3: Try Windows API
    set_window_icon_windows(logo_ico or logo_png)

    # Create app controller and show welcome screen
    app_controller = AppController(page)
    app_controller.show_welcome_screen()


if __name__ == "__main__":
    ft.app(target=main)