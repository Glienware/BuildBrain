"""
Project List Component

Displays saved projects with step progress indicators.
"""

import json
import os
import flet as ft


class ProjectList:
    """
    Displays list of saved projects with progress steppers.
    """

    STEPS = ["Created", "Configured", "Dataset Loaded", "Trained", "Exported"]

    def __init__(self, page: ft.Page, project_manager):
        self.page = page
        self.project_manager = project_manager
        self.project_list = ft.Column(scroll=ft.ScrollMode.AUTO)
        self.refresh_projects()

    def get_project_status(self, config):
        """
        Determine the current step based on configuration completeness.
        """
        step = 0
        if config.get("project_name"):
            step = 1
        if config.get("task_type") and config.get("model"):
            step = 2
        if config.get("dataset_path"):
            step = 3
        # For trained and exported, we'd need to track that separately
        # For now, assume up to dataset loaded
        return step

    def create_stepper(self, current_step):
        """
        Create a visual stepper/progress indicator.
        """
        stepper_items = []
        for i, step in enumerate(self.STEPS):
            # Circle
            if i < current_step:
                circle = ft.Container(
                    content=ft.Icon(ft.Icons.CHECK, size=16, color=ft.Colors.WHITE),
                    width=36,
                    height=36,
                    bgcolor=ft.Colors.GREEN_600,
                    border_radius=18,
                    alignment=ft.alignment.center,
                    shadow=ft.BoxShadow(blur_radius=4, color=ft.Colors.GREEN_200)
                )
            elif i == current_step:
                circle = ft.Container(
                    content=ft.Text(str(i+1), size=16, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    width=36,
                    height=36,
                    bgcolor=ft.Colors.BLUE_600,
                    border_radius=18,
                    alignment=ft.alignment.center,
                    shadow=ft.BoxShadow(blur_radius=4, color=ft.Colors.BLUE_200)
                )
            else:
                circle = ft.Container(
                    content=ft.Text(str(i+1), size=16, color=ft.Colors.GREY_600),
                    width=36,
                    height=36,
                    bgcolor=ft.Colors.GREY_300,
                    border_radius=18,
                    alignment=ft.alignment.center
                )

            # Step label
            label = ft.Text(
                step,
                size=12,
                color=ft.Colors.BLUE_500 if i <= current_step else ft.Colors.GREY_500,
                weight=ft.FontWeight.BOLD if i == current_step else ft.FontWeight.NORMAL
            )

            stepper_items.append(
                ft.Column([
                    circle,
                    ft.Container(height=4),
                    label
                ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=4)
            )

            # Add line between steps (except last)
            if i < len(self.STEPS) - 1:
                line_color = ft.Colors.GREEN_600 if i < current_step else ft.Colors.GREY_400
                stepper_items.append(
                    ft.Container(
                        width=60,
                        height=3,
                        bgcolor=line_color,
                        border_radius=2,
                        margin=ft.margin.symmetric(vertical=18)
                    )
                )

        return ft.Row(
            stepper_items,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.START,
            spacing=8
        )

    def create_project_card(self, project_name, config):
        """
        Create a card for a project with stepper.
        """
        current_step = self.get_project_status(config)

        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    # Header with gradient background
                    ft.Container(
                        content=ft.Row([
                            ft.Icon(ft.Icons.FOLDER, size=24, color=ft.Colors.WHITE),
                            ft.Text(project_name, size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                            ft.Container(width=10),
                            ft.Chip(
                                label=ft.Text(f"Step {current_step + 1}", color=ft.Colors.WHITE),
                                bgcolor=ft.Colors.BLUE_700,
                            )
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                        padding=ft.padding.all(15),
                        bgcolor=ft.Colors.BLUE_600,
                        border_radius=ft.border_radius.only(top_left=10, top_right=10)
                    ),

                    # Content
                    ft.Container(
                        content=ft.Column([
                            ft.Container(height=10),
                            # Stepper
                            self.create_stepper(current_step),

                            ft.Divider(height=20, color=ft.Colors.TRANSPARENT),

                            # Quick info
                            ft.Row([
                                ft.Container(
                                    content=ft.Text(f"🤖 {config.get('model', 'Not set')}", size=13),
                                    padding=ft.padding.symmetric(horizontal=10, vertical=5),
                                    bgcolor=ft.Colors.BLUE_50,
                                    border_radius=15
                                ),
                                ft.Container(
                                    content=ft.Text(f"🎯 {config.get('task_type', 'Not set')}", size=13),
                                    padding=ft.padding.symmetric(horizontal=10, vertical=5),
                                    bgcolor=ft.Colors.ORANGE_50,
                                    border_radius=15
                                ),
                            ], alignment=ft.MainAxisAlignment.START, spacing=10),

                            ft.Divider(height=15, color=ft.Colors.TRANSPARENT),

                            # Actions
                            ft.Row([
                                ft.ElevatedButton(
                                    "Load Project",
                                    icon=ft.Icons.PLAY_ARROW,
                                    on_click=lambda e, name=project_name: self.load_project(name),
                                    style=ft.ButtonStyle(
                                        bgcolor=ft.Colors.BLUE_600,
                                        color=ft.Colors.WHITE,
                                        elevation=2
                                    )
                                ),
                                ft.OutlinedButton(
                                    "Delete",
                                    icon=ft.Icons.DELETE,
                                    on_click=lambda e, name=project_name: self.delete_project(name),
                                    style=ft.ButtonStyle(color=ft.Colors.RED_600)
                                )
                            ], alignment=ft.MainAxisAlignment.END, spacing=10)

                        ], spacing=10),
                        padding=ft.padding.all(20)
                    )

                ], spacing=0),
                padding=0
            ),
            elevation=6,
            margin=ft.margin.all(8),
            shadow_color=ft.Colors.BLUE_200
        )

    def load_project(self, project_name):
        """
        Load a project by setting it in the dropdown and triggering load.
        """
        self.project_manager.load_project_dropdown.value = project_name
        self.project_manager.load_project(None)

    def delete_project(self, project_name):
        """
        Delete a project file.
        """
        filename = f"{self.project_manager.CONFIG_DIR}/{project_name}.json"
        try:
            os.remove(filename)
            self.refresh_projects()
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Project '{project_name}' deleted!"))
            self.page.snack_bar.open = True
        except Exception as ex:
            self.page.snack_bar = ft.SnackBar(ft.Text(f"Error deleting project: {str(ex)}"))
            self.page.snack_bar.open = True
        self.page.update()

    def refresh_projects(self):
        """
        Refresh the project list display.
        """
        self.project_list.controls.clear()
        try:
            files = os.listdir(self.project_manager.CONFIG_DIR)
            projects = [f for f in files if f.endswith('.json')]
            for project_file in projects:
                project_name = project_file.replace('.json', '')
                try:
                    with open(f"{self.project_manager.CONFIG_DIR}/{project_file}", 'r') as f:
                        config = json.load(f)
                    self.project_list.controls.append(self.create_project_card(project_name, config))
                except:
                    continue
        except:
            pass

        if not self.project_list.controls:
            self.project_list.controls.append(
                ft.Container(
                    content=ft.Column([
                        ft.Icon(ft.Icons.INBOX, size=64, color=ft.Colors.GREY_400),
                        ft.Text("No projects saved yet", size=20, color=ft.Colors.GREY_600, weight=ft.FontWeight.BOLD),
                        ft.Text("Create and save your first ML project!", size=14, color=ft.Colors.GREY_500),
                        ft.Container(height=20),
                        ft.ElevatedButton(
                            "Go to Setup",
                            icon=ft.Icons.ARROW_FORWARD,
                            on_click=lambda e: self.page.go("/setup"),  # This might not work, but placeholder
                            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_600, color=ft.Colors.WHITE)
                        )
                    ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=15),
                    padding=60,
                    alignment=ft.alignment.center,
                    bgcolor=ft.Colors.GREY_50,
                    border_radius=15
                )
            )

        self.page.update()

    def build(self):
        """
        Build the project list UI.
        """
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.LIST, size=24),
                    ft.Text("Saved Projects", size=20, weight=ft.FontWeight.BOLD),
                    ft.Container(width=10),
                    ft.ElevatedButton(
                        "Refresh",
                        icon=ft.Icons.REFRESH,
                        on_click=lambda e: self.refresh_projects(),
                        style=ft.ButtonStyle(bgcolor=ft.Colors.GREY_100)
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(height=10),
                self.project_list
            ], spacing=10),
            padding=10,
            expand=True
        )