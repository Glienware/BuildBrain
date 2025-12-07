"""
Project Manager Component

Handles saving and loading project configurations.
"""

import json
import os
import flet as ft


class ProjectManager:
    """
    Manages project configurations and drafts.
    """

    CONFIG_DIR = "projects"

    def __init__(self, page: ft.Page, components):
        self.page = page
        self.components = components  # Dict of component instances
        self.project_list = None  # Will be set later
        self.project_name = ft.TextField(label="Project Name", width=300)
        self.save_draft_btn = ft.ElevatedButton(
            "Save Draft",
            icon=ft.Icons.SAVE,
            on_click=self.save_draft,
            style=ft.ButtonStyle(bgcolor=ft.Colors.ORANGE_400)
        )
        self.load_project_dropdown = ft.Dropdown(
            label="Load Project",
            options=[],
            width=300,
            on_change=self.load_project
        )
        self.load_btn = ft.ElevatedButton(
            "Load",
            icon=ft.Icons.FOLDER_OPEN,
            on_click=self.refresh_projects
        )

        # Ensure config directory exists
        os.makedirs(self.CONFIG_DIR, exist_ok=True)
        self.refresh_projects()

    def get_config(self):
        """
        Get current configuration from all components.
        """
        config = {
            "project_name": self.project_name.value or "Untitled",
            "task_type": self.components["task_selector"].task_type.value,
            "model": self.components["model_selector"].selected_model.value,
            "preset": self.components["settings_panel"].preset.value,
            "manual_mode": self.components["settings_panel"].manual_mode.value,
            "hyperparams": {
                "learning_rate": self.components["settings_panel"].learning_rate.value,
                "batch_size": self.components["settings_panel"].batch_size.value,
                "epochs": self.components["settings_panel"].epochs.value,
                "optimizer": self.components["settings_panel"].optimizer.value,
                "weight_decay": self.components["settings_panel"].weight_decay.value,
                "max_depth": self.components["settings_panel"].max_depth.value,
                "n_estimators": self.components["settings_panel"].n_estimators.value,
            },
            "classes": self.components["class_manager"].classes,
            "dataset_path": self.components["dataset_uploader"].dataset_path,
        }
        return config

    def set_config(self, config):
        """
        Set configuration to all components.
        """
        self.project_name.value = config.get("project_name", "")
        self.components["task_selector"].task_type.value = config.get("task_type", "")
        self.components["model_selector"].selected_model.value = config.get("model", "")
        self.components["settings_panel"].preset.value = config.get("preset", "")
        self.components["settings_panel"].manual_mode.value = config.get("manual_mode", False)
        self.components["settings_panel"].learning_rate.value = config.get("hyperparams", {}).get("learning_rate", "")
        self.components["settings_panel"].batch_size.value = config.get("hyperparams", {}).get("batch_size", "")
        self.components["settings_panel"].epochs.value = config.get("hyperparams", {}).get("epochs", "")
        self.components["settings_panel"].optimizer.value = config.get("hyperparams", {}).get("optimizer", "")
        self.components["settings_panel"].weight_decay.value = config.get("hyperparams", {}).get("weight_decay", "")
        self.components["settings_panel"].max_depth.value = config.get("hyperparams", {}).get("max_depth", "")
        self.components["settings_panel"].n_estimators.value = config.get("hyperparams", {}).get("n_estimators", "")
        self.components["class_manager"].classes = config.get("classes", [])
        self.components["class_manager"].update_class_list()
        self.components["dataset_uploader"].dataset_path = config.get("dataset_path", "")
        # Update the page
        self.page.update()

    def save_draft(self, e):
        """
        Save current configuration as draft.
        """
        if not self.project_name.value:
            self.page.snack_bar = ft.SnackBar(ft.Text("Please enter a project name"))
            self.page.snack_bar.open = True
            self.page.update()
            return

        config = self.get_config()
        filename = f"{self.CONFIG_DIR}/{self.project_name.value}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Project '{self.project_name.value}' saved!"))
            self.page.snack_bar.open = True
            self.refresh_projects()
            if self.project_list:
                self.project_list.refresh_projects()
        except Exception as ex:
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Error saving project: {str(ex)}"))
            self.page.snack_bar.open = True
        self.page.update()

    def load_project(self, e):
        """
        Load selected project configuration.
        """
        if not self.load_project_dropdown.value:
            return

        # Try .buildb first, then .json
        filename_buildb = f"{self.CONFIG_DIR}/{self.load_project_dropdown.value}.buildb"
        filename_json = f"{self.CONFIG_DIR}/{self.load_project_dropdown.value}.json"
        
        filename = None
        if os.path.exists(filename_buildb):
            filename = filename_buildb
        elif os.path.exists(filename_json):
            filename = filename_json
            
        if not filename:
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Project file not found"))
            self.page.snack_bar.open = True
            self.page.update()
            return

        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            self.set_config(config)
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Project '{self.load_project_dropdown.value}' loaded!"))
            self.page.snack_bar.open = True
        except Exception as ex:
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Error loading project: {str(ex)}"))
            self.page.snack_bar.open = True
        self.page.update()

    def load_project_data(self, project_data):
        """
        Load project configuration from data dictionary.
        """
        try:
            self.set_config(project_data)
            self.page.snack_bar = ft.SnackBar(ft.Text("Project loaded successfully!"))
            self.page.snack_bar.open = True
            self.page.update()
        except Exception as ex:
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Error loading project: {str(ex)}"))
            self.page.snack_bar.open = True
            self.page.update()

    def refresh_projects(self, e=None):
        """
        Refresh the list of available projects.
        """
        try:
            files = os.listdir(self.CONFIG_DIR)
            projects = []
            for f in files:
                if f.endswith('.json'):
                    projects.append(f.replace('.json', ''))
                elif f.endswith('.buildb'):
                    projects.append(f.replace('.buildb', ''))
            self.load_project_dropdown.options = [ft.dropdown.Option(p) for p in projects]
        except:
            self.load_project_dropdown.options = []
            self.page.update()

    def save_trained_model(self, model_path):
        """
        Save the path of the trained model to the current project.
        """
        if self.current_config:
            self.current_config["trained_model_path"] = model_path
            self._save_current_config()
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Model path saved: {model_path}"))
            self.page.snack_bar.open = True
            self.page.update()

    def build(self):
        """
        Build the project manager UI.
        """
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.FOLDER, size=24),
                    ft.Text("Project Management", size=16, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.START),
                self.project_name,
                ft.Row([self.save_draft_btn], spacing=10),
                ft.Divider(),
                ft.Row([
                    self.load_project_dropdown,
                    self.load_btn
                ], spacing=10)
            ], spacing=10),
            padding=10
        )