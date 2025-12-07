"""
New Project Wizard for AI/ML Trainer App

Step-by-step wizard similar to Android Studio for creating new ML projects.
"""

import flet as ft
import os
import json
from datetime import datetime
from typing import Dict, Any
from .dataset_uploader import DatasetUploader


class NewProjectWizard:
    """
    Step-by-step wizard for creating new ML projects with Android Studio-like interface.
    """

    def __init__(self, page: ft.Page, on_complete, on_cancel):
        self.page = page
        self.on_complete = on_complete
        self.on_cancel = on_cancel

        # Project data
        self.project_data = {
            "project_name": "",
            "task_type": "classification",
            "model_type": "LogisticRegression",
            "training_settings": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "advanced_enabled": False
            },
            "classes": [],
            "dataset_path": None,
            "dataset_type": None,
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }

        # UI state
        self.current_step = 0
        self.steps = [
            "Project Name",
            "Task Configuration",
            "Class Management",
            "Balanced Class",
            "Dataset Upload",
            "Training & Logs"
        ]

        # UI components
        self.step_indicator = None
        self.content_area = None
        self.next_button = None
        self.back_button = None

    def build(self):
        """Build the wizard interface."""
        # Initialize navigation buttons first
        self._build_navigation_buttons()

        self.step_indicator = self._build_step_indicator()
        self.content_area = self._build_content_area()

        return ft.Container(
            content=ft.Column([
                # Header with navigation
                ft.Container(
                    content=ft.Row([
                        ft.IconButton(
                            ft.Icons.CLOSE,
                            on_click=self._show_cancel_dialog,
                            tooltip="Cancel"
                        ),
                        ft.Text("Create New Model", size=24, weight=ft.FontWeight.BOLD),
                        ft.Container(expand=True),
                        ft.Row([
                            self.back_button,
                            ft.Container(width=10),
                            self.next_button,
                        ], spacing=0),
                        ft.Container(width=20),
                        ft.Text("AI/ML Trainer", size=14, color=ft.Colors.GREY_600),
                    ], alignment=ft.MainAxisAlignment.START),
                    bgcolor=ft.Colors.GREY_800,
                    padding=ft.padding.all(20),
                ),

                # Main content - Scrollable
                ft.Container(
                    content=ft.Row([
                        # Left side - Step indicator
                        ft.Container(
                            content=self.step_indicator,
                            width=300,
                            bgcolor=ft.Colors.GREY_900,
                            padding=ft.padding.all(20),
                        ),

                        # Right side - Content
                        ft.Container(
                            content=ft.Column([
                                self.content_area
                            ], scroll=ft.ScrollMode.AUTO, spacing=0),
                            expand=True,
                            bgcolor=ft.Colors.GREY_800,
                            padding=ft.padding.all(30),
                        ),
                    ], alignment=ft.MainAxisAlignment.START),
                    expand=True,
                ),

            ], spacing=0),
            bgcolor=ft.Colors.GREY_900,
            height=self.page.height - 100,  # Leave some space
        )

    def _build_step_indicator(self):
        """Build the vertical step indicator."""
        step_items = []
        for i, step_name in enumerate(self.steps):
            is_active = i == self.current_step
            is_completed = i < self.current_step

            step_items.append(
                ft.Container(
                    content=ft.Row([
                        # Step circle
                        ft.Container(
                            content=ft.Text(
                                str(i + 1),
                                size=16,
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.WHITE if is_active or is_completed else ft.Colors.GREY_600,
                            ),
                            width=40,
                            height=40,
                            bgcolor=ft.Colors.GREEN_600 if is_completed else (ft.Colors.BLUE_600 if is_active else ft.Colors.GREY_700),
                            border_radius=20,
                            alignment=ft.alignment.center,
                        ),

                        # Step text
                        ft.Container(
                            content=ft.Column([
                                ft.Text(
                                    step_name,
                                    size=14,
                                    weight=ft.FontWeight.BOLD,
                                    color=ft.Colors.WHITE if is_active or is_completed else ft.Colors.GREY_500,
                                ),
                                ft.Text(
                                    self._get_step_description(i),
                                    size=12,
                                    color=ft.Colors.GREY_400,
                                ),
                            ], spacing=2, tight=True),
                            padding=ft.padding.only(left=15),
                        ),
                    ], alignment=ft.MainAxisAlignment.START),
                    padding=ft.padding.symmetric(vertical=15),
                )
            )

            # Add connector line (except for last item)
            if i < len(self.steps) - 1:
                step_items.append(
                    ft.Container(
                        content=ft.Container(
                            width=2,
                            height=30,
                            bgcolor=ft.Colors.GREEN_600 if is_completed else ft.Colors.GREY_700,
                        ),
                        padding=ft.padding.only(left=19),
                    )
                )

        return ft.Column(step_items, spacing=0, scroll=ft.ScrollMode.AUTO)

    def _get_step_description(self, step_index):
        """Get description for each step."""
        descriptions = [
            "Name your project",
            "Configure model and training",
            "Define classes",
            "Check class balance",
            "Upload dataset",
            "Train and monitor"
        ]
        return descriptions[step_index]

    def _build_content_area(self):
        """Build the main content area for current step."""
        if self.current_step == 0:
            return self._build_project_name_step()
        elif self.current_step == 1:
            return self._build_task_config_step()
        elif self.current_step == 2:
            return self._build_class_management_step()
        elif self.current_step == 3:
            return self._build_balanced_class_step()
        elif self.current_step == 4:
            return self._build_dataset_upload_step()
        elif self.current_step == 5:
            return self._build_training_logs_step()
        return ft.Text("Step not implemented")

    def _refresh_current_step(self):
        """Refresh the current step content."""
        # Rebuild the content area
        self.content_area.content = self._build_content_area()
        self.page.update()

    def _build_content_area(self):
        """Build the main content area for current step."""
        if self.current_step == 0:
            return self._build_project_name_step()
        elif self.current_step == 1:
            return self._build_task_config_step()
        elif self.current_step == 2:
            return self._build_class_management_step()
        elif self.current_step == 3:
            return self._build_balanced_class_step()
        elif self.current_step == 4:
            return self._build_dataset_upload_step()
        elif self.current_step == 5:
            return self._build_training_logs_step()
        return ft.Text("Step not implemented")

    def _build_project_name_step(self):
        """Build project name input step."""
        project_name_field = ft.TextField(
            label="Project Name",
            hint_text="Enter a name for your ML project",
            value=self.project_data["project_name"],
            on_change=lambda e: self._update_project_name(e.data),
            width=400,
            border_radius=8,
        )

        return ft.Container(
            content=ft.Column([
                ft.Text("Project Name", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                ft.Text("Give your machine learning project a meaningful name", size=16, color=ft.Colors.GREY_600),
                ft.Container(height=40),

                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Icon(ft.Icons.EDIT, size=48, color=ft.Colors.BLUE_600),
                            ft.Container(height=20),
                            project_name_field,
                            ft.Container(height=20),
                            ft.Text("This name will be used to save your project file", size=12, color=ft.Colors.GREY_500),
                        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=10),
                        padding=40,
                    ),
                    elevation=4,
                ),
            ], horizontal_alignment=ft.CrossAxisAlignment.START, spacing=20),
        )

    def _build_task_config_step(self):
        """Build task configuration step."""
        # Model type selection
        model_options = [
            ft.dropdown.Option("LogisticRegression", "Logistic Regression"),
            ft.dropdown.Option("RandomForest", "Random Forest"),
            ft.dropdown.Option("XGBoost", "XGBoost"),
            ft.dropdown.Option("PyTorch_CNN", "PyTorch CNN"),
            ft.dropdown.Option("PyTorch_MLP", "PyTorch MLP"),
        ]

        model_dropdown = ft.Dropdown(
            label="Model Type",
            options=model_options,
            value=self.project_data["model_type"],
            on_change=lambda e: self._update_model_type(e.data),
            width=300,
        )

        # Task type selection
        task_radio = ft.RadioGroup(
            content=ft.Row([
                ft.Radio(value="classification", label="Classification"),
                ft.Container(width=20),
                ft.Radio(value="regression", label="Regression"),
            ]),
            value=self.project_data["task_type"],
            on_change=lambda e: self._update_task_type(e.value),
        )

        # Basic settings
        epochs_field = ft.TextField(
            label="Epochs",
            value=str(self.project_data["training_settings"]["epochs"]),
            on_change=lambda e: self._update_setting("epochs", int(e.data) if e.data.isdigit() else 100),
            width=150,
        )

        batch_size_field = ft.TextField(
            label="Batch Size",
            value=str(self.project_data["training_settings"]["batch_size"]),
            on_change=lambda e: self._update_setting("batch_size", int(e.data) if e.data.isdigit() else 32),
            width=150,
        )

        learning_rate_field = ft.TextField(
            label="Learning Rate",
            value=str(self.project_data["training_settings"]["learning_rate"]),
            on_change=lambda e: self._update_setting("learning_rate", float(e.data) if e.data.replace('.', '').replace('-', '').isdigit() else 0.001),
            width=150,
        )

        optimizer_dropdown = ft.Dropdown(
            label="Optimizer",
            options=[
                ft.dropdown.Option("adam", "Adam"),
                ft.dropdown.Option("sgd", "SGD"),
                ft.dropdown.Option("rmsprop", "RMSprop"),
            ],
            value=self.project_data["training_settings"]["optimizer"],
            on_change=lambda e: self._update_setting("optimizer", e.data),
            width=150,
        )

        return ft.Container(
            content=ft.Column([
                ft.Text("Task Configuration", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                ft.Text("Configure your model and training parameters", size=16, color=ft.Colors.GREY_600),
                ft.Container(height=30),

                # Model and Task Selection
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Model & Task", size=18, weight=ft.FontWeight.BOLD),
                            ft.Container(height=15),
                            ft.Row([model_dropdown, ft.Container(width=20), task_radio]),
                        ], spacing=15),
                        padding=20,
                    ),
                    elevation=2,
                ),

                ft.Container(height=20),

                # Training Settings
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Training Settings", size=18, weight=ft.FontWeight.BOLD),
                            ft.Container(height=15),
                            ft.Row([epochs_field, batch_size_field]),
                            ft.Container(height=10),
                            ft.Row([learning_rate_field, optimizer_dropdown]),
                        ], spacing=15),
                        padding=20,
                    ),
                    elevation=2,
                ),

            ], horizontal_alignment=ft.CrossAxisAlignment.START, spacing=20),
        )

    def _build_class_management_step(self):
        """Build class management step."""
        # Class input
        class_name_field = ft.TextField(
            label="Class Name",
            hint_text="Enter class name",
            width=300,
        )

        add_class_button = ft.ElevatedButton(
            "Add Class",
            icon=ft.Icons.ADD,
            on_click=lambda e: self._add_class(class_name_field.value),
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_600, color=ft.Colors.WHITE),
        )

        # Classes list
        self.classes_list = ft.Column(spacing=10)
        self._update_classes_display()

        return ft.Container(
            content=ft.Column([
                ft.Text("Class Management", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                ft.Text("Define the classes for your classification task", size=16, color=ft.Colors.GREY_600),
                ft.Container(height=30),

                # Add class section
                ft.Card(
                    content=ft.Container(
                        content=ft.Row([
                            class_name_field,
                            ft.Container(width=10),
                            add_class_button,
                        ], alignment=ft.MainAxisAlignment.START),
                        padding=20,
                    ),
                    elevation=2,
                ),

                ft.Container(height=20),

                # Classes list
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Defined Classes", size=18, weight=ft.FontWeight.BOLD),
                            ft.Container(height=15),
                            ft.Container(
                                content=self.classes_list,
                                height=200,
                                border=ft.border.all(1, ft.Colors.GREY_300),
                                border_radius=8,
                                padding=10,
                            ),
                        ], spacing=15),
                        padding=20,
                    ),
                    elevation=2,
                ),

            ], horizontal_alignment=ft.CrossAxisAlignment.START, spacing=20),
        )

    def _build_balanced_class_step(self):
        """Build balanced class analysis step."""
        # Class distribution visualization
        self.balance_chart = ft.Column()

        balance_button = ft.ElevatedButton(
            "Balance Classes",
            icon=ft.Icons.BALANCE,
            on_click=self._balance_classes,
            style=ft.ButtonStyle(bgcolor=ft.Colors.ORANGE_600, color=ft.Colors.WHITE),
        )

        self._update_balance_display()

        return ft.Container(
            content=ft.Column([
                ft.Text("Class Balance Analysis", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                ft.Text("Check and balance your class distribution", size=16, color=ft.Colors.GREY_600),
                ft.Container(height=30),

                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Class Distribution", size=18, weight=ft.FontWeight.BOLD),
                            ft.Container(height=15),
                            self.balance_chart,
                            ft.Container(height=20),
                            balance_button,
                        ], spacing=15),
                        padding=20,
                    ),
                    elevation=2,
                ),

            ], horizontal_alignment=ft.CrossAxisAlignment.START, spacing=20),
        )

    def _build_dataset_upload_step(self):
        """Build dataset upload step with class-based organization."""
        # Initialize dataset uploader with existing classes
        if not hasattr(self, 'dataset_uploader'):
            self.dataset_uploader = DatasetUploader(self.page, existing_classes=self.project_data.get("classes", []), on_update=self._refresh_current_step)

        return ft.Container(
            content=ft.Column([
                ft.Text("Dataset Classes", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                ft.Text("Upload images for each class defined in the previous step", size=16, color=ft.Colors.GREY_600),
                ft.Container(height=30),

                # Dataset uploader component
                self.dataset_uploader.build(),

                # Dataset summary
                ft.Container(
                    content=self._build_dataset_summary(),
                    margin=ft.margin.only(top=20),
                ),

            ], horizontal_alignment=ft.CrossAxisAlignment.START, spacing=20),
        )

    def _build_dataset_summary(self):
        """Build dataset summary card."""
        if not hasattr(self, 'dataset_uploader'):
            return ft.Container()

        dataset_info = self.dataset_uploader.get_dataset_info()

        return ft.Card(
            content=ft.Container(
                content=ft.Column([
                    ft.Text("Dataset Summary", size=18, weight=ft.FontWeight.BOLD),
                    ft.Container(height=10),
                    ft.Row([
                        ft.Column([
                            ft.Text(f"Classes: {dataset_info['num_classes']}", size=14),
                            ft.Text(f"Total Images: {dataset_info['total_images']}", size=14),
                        ], spacing=5),
                        ft.Container(width=50),
                        ft.Column([
                            ft.Text("Class Distribution:", size=14, weight=ft.FontWeight.BOLD),
                            ft.Container(
                                content=ft.Column([
                                    ft.Text(f"• {class_name}: {len(images)} images", size=12)
                                    for class_name, images in dataset_info['classes'].items()
                                ], spacing=2, scroll=ft.ScrollMode.AUTO),
                                height=100,
                            ),
                        ], spacing=5, expand=True),
                    ], alignment=ft.MainAxisAlignment.START),
                ], spacing=10),
                padding=20,
            ),
            elevation=2,
        )

    def _build_training_logs_step(self):
        """Build training and logs step."""
        # Training controls
        quick_train_button = ft.ElevatedButton(
            "Quick Training",
            icon=ft.Icons.PLAY_ARROW,
            on_click=self._start_quick_training,
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_600, color=ft.Colors.WHITE),
            width=200,
        )

        advanced_train_button = ft.ElevatedButton(
            "Advanced Training",
            icon=ft.Icons.SETTINGS,
            on_click=self._start_advanced_training,
            style=ft.ButtonStyle(bgcolor=ft.Colors.PURPLE_600, color=ft.Colors.WHITE),
            width=200,
        )

        # Progress and logs
        self.training_progress = ft.ProgressBar(value=0, width=400)
        self.training_logs = ft.Column(scroll=ft.ScrollMode.AUTO, height=200)

        return ft.Container(
            content=ft.Column([
                ft.Text("Training & Logs", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
                ft.Text("Start training your model and monitor progress", size=16, color=ft.Colors.GREY_600),
                ft.Container(height=30),

                # Training buttons
                ft.Card(
                    content=ft.Container(
                        content=ft.Row([
                            quick_train_button,
                            ft.Container(width=20),
                            advanced_train_button,
                        ], alignment=ft.MainAxisAlignment.CENTER),
                        padding=30,
                    ),
                    elevation=2,
                ),

                ft.Container(height=20),

                # Progress and logs
                ft.Card(
                    content=ft.Container(
                        content=ft.Column([
                            ft.Text("Training Progress", size=18, weight=ft.FontWeight.BOLD),
                            ft.Container(height=15),
                            self.training_progress,
                            ft.Container(height=20),
                            ft.Text("Training Logs", size=16, weight=ft.FontWeight.BOLD),
                            ft.Container(
                                content=self.training_logs,
                                height=200,
                                border=ft.border.all(1, ft.Colors.GREY_300),
                                border_radius=8,
                                padding=10,
                                bgcolor=ft.Colors.GREY_50,
                            ),
                        ], spacing=15),
                        padding=20,
                    ),
                    elevation=2,
                ),

            ], horizontal_alignment=ft.CrossAxisAlignment.START, spacing=20),
        )

    def _build_navigation_buttons(self):
        """Build navigation buttons."""
        self.back_button = ft.ElevatedButton(
            "Back",
            icon=ft.Icons.ARROW_BACK,
            on_click=self._go_back,
            disabled=self.current_step == 0,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREY_700, color=ft.Colors.WHITE),
        )

        next_text = "Complete" if self.current_step == len(self.steps) - 1 else "Next"
        self.next_button = ft.ElevatedButton(
            next_text,
            icon=ft.Icons.ARROW_FORWARD if next_text != "Complete" else ft.Icons.CHECK,
            on_click=self._go_next,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_400, color=ft.Colors.BLACK),
        )

    def _show_cancel_dialog(self, e):
        """Show cancel confirmation dialog."""
        def close_dialog(e):
            dlg.open = False
            self.page.update()

        def confirm_cancel(e):
            dlg.open = False
            self.page.update()
            self.on_cancel(e)

        dlg = ft.AlertDialog(
            modal=True,
            title=ft.Text("Cancel Project Creation"),
            content=ft.Text("Are you sure you want to cancel? All progress will be lost and not saved."),
            actions=[
                ft.TextButton("Continue Creating", on_click=close_dialog),
                ft.ElevatedButton(
                    "Cancel & Exit",
                    on_click=confirm_cancel,
                    style=ft.ButtonStyle(bgcolor=ft.Colors.RED_600, color=ft.Colors.WHITE)
                ),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
        )

        self.page.dialog = dlg
        dlg.open = True
        self.page.update()
        """Build navigation buttons."""
        self.back_button = ft.ElevatedButton(
            "Back",
            icon=ft.Icons.ARROW_BACK,
            on_click=self._go_back,
            disabled=self.current_step == 0,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREY_600, color=ft.Colors.WHITE),
        )

        next_text = "Complete" if self.current_step == len(self.steps) - 1 else "Next"
        self.next_button = ft.ElevatedButton(
            next_text,
            icon=ft.Icons.ARROW_FORWARD if next_text != "Complete" else ft.Icons.CHECK,
            on_click=self._go_next,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_600, color=ft.Colors.WHITE),
        )

    def _go_next(self, e):
        """Go to next step."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self._refresh_ui()
        else:
            # Complete the wizard
            self._complete_wizard()

    def _go_back(self, e):
        """Go to previous step."""
        if self.current_step > 0:
            self.current_step -= 1
            self._refresh_ui()

    def _refresh_ui(self):
        """Refresh the UI for current step."""
        self.step_indicator = self._build_step_indicator()
        self.content_area = self._build_content_area()
        self._build_navigation_buttons()

        # Rebuild the entire wizard
        self.page.controls.clear()
        self.page.add(self.build())
        self.page.update()

    # Helper methods for updating project data
    def _update_project_name(self, name):
        self.project_data["project_name"] = name

    def _update_model_type(self, model_type):
        self.project_data["model_type"] = model_type

    def _update_task_type(self, task_type):
        self.project_data["task_type"] = task_type

    def _update_setting(self, key, value):
        self.project_data["training_settings"][key] = value

    def _add_class(self, class_name):
        if class_name and class_name not in self.project_data["classes"]:
            self.project_data["classes"].append(class_name)
            self._update_classes_display()

    def _update_classes_display(self):
        """Update the classes list display."""
        self.classes_list.controls.clear()
        for i, class_name in enumerate(self.project_data["classes"]):
            self.classes_list.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Text(f"{i+1}. {class_name}", size=14),
                        ft.IconButton(
                            ft.Icons.DELETE,
                            on_click=lambda e, idx=i: self._remove_class(idx),
                            icon_size=16,
                        ),
                    ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                    padding=ft.padding.all(5),
                )
            )

    def _remove_class(self, index):
        if 0 <= index < len(self.project_data["classes"]):
            self.project_data["classes"].pop(index)
            self._update_classes_display()

    def _update_balance_display(self):
        """Update class balance visualization."""
        self.balance_chart.controls.clear()

        if not self.project_data["classes"]:
            self.balance_chart.controls.append(
                ft.Text("No classes defined yet", color=ft.Colors.GREY_500)
            )
            return

        # Mock balance data (in real app, this would analyze actual dataset)
        for class_name in self.project_data["classes"]:
            # Mock count - in real app, analyze dataset
            count = 100  # Mock value
            percentage = 100 / len(self.project_data["classes"])

            self.balance_chart.controls.append(
                ft.Container(
                    content=ft.Row([
                        ft.Text(class_name, size=14, width=100),
                        ft.Container(
                            content=ft.Text(f"{count} samples", size=12),
                            width=80,
                        ),
                        ft.ProgressBar(value=percentage/100, width=150),
                        ft.Text(f"{percentage:.1f}%", size=12, width=50),
                    ], alignment=ft.MainAxisAlignment.START),
                    padding=ft.padding.symmetric(vertical=5),
                )
            )

    def _balance_classes(self, e):
        """Balance classes automatically."""
        # In real implementation, this would apply SMOTE or other balancing techniques
        self._add_log("Classes balanced using SMOTE algorithm")

    def _select_dataset(self, e):
        """Select dataset files/folders."""
        # In real implementation, this would open file picker
        self._add_log("Dataset selection functionality would be implemented here")

    def _start_quick_training(self, e):
        """Start quick training."""
        self.training_progress.value = 0
        self._add_log("Starting quick training...")
        # In real implementation, this would start training in background
        self._simulate_training()

    def _start_advanced_training(self, e):
        """Start advanced training."""
        self._add_log("Starting advanced training...")
        # In real implementation, this would open advanced training dialog

    def _simulate_training(self):
        """Simulate training progress."""
        import time
        for i in range(101):
            time.sleep(0.05)  # Simulate training time
            self.training_progress.value = i / 100
            if i % 10 == 0:
                self._add_log(f"Epoch {i//10}/10 completed - Loss: {1.0 - i/200:.3f}")
            self.page.update()

    def _add_log(self, message):
        """Add message to training logs."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.training_logs.controls.append(
            ft.Text(f"[{timestamp}] {message}", size=12, color=ft.Colors.GREY_700)
        )
        self.page.update()

    def _complete_wizard(self):
        """Complete the wizard and save project."""
        # Save project to .buildb file
        self._save_project()
        self.on_complete(self.project_data)

    def _save_project(self):
        """Save project to .buildb file."""
        # Create projects directory if it doesn't exist
        projects_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "projects")
        os.makedirs(projects_dir, exist_ok=True)

        project_name = self.project_data['project_name']
        project_dir = os.path.join(projects_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)

        # Generate filename
        filename = f"{project_name}.buildb"
        filepath = os.path.join(project_dir, filename)

        # Update timestamps
        self.project_data["last_modified"] = datetime.now().isoformat()

        # Save dataset information if available
        if hasattr(self, 'dataset_uploader'):
            dataset_info = self.dataset_uploader.get_dataset_info()
            # Copy images to project dataset folder
            copied_dataset = self._copy_dataset_to_project(dataset_info, project_dir)
            self.project_data["dataset"] = copied_dataset

        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(self.project_data, f, indent=2)

        self._add_log(f"Project saved to {filename}")

    def _copy_dataset_to_project(self, dataset_info, project_dir):
        """Copy dataset images to project folder and return updated dataset info."""
        import shutil

        dataset_dir = os.path.join(project_dir, "dataset")
        copied_dataset = {"classes": {}, "total_images": 0, "num_classes": 0}

        for class_name, image_paths in dataset_info.get("classes", {}).items():
            class_dir = os.path.join(dataset_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            copied_paths = []

            for src_path in image_paths:
                if os.path.exists(src_path):
                    filename = os.path.basename(src_path)
                    dst_path = os.path.join(class_dir, filename)
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_paths.append(dst_path)
                    except Exception as e:
                        self._add_log(f"Error copying {src_path}: {e}")
                else:
                    self._add_log(f"Source image not found: {src_path}")

            copied_dataset["classes"][class_name] = copied_paths

        copied_dataset["total_images"] = sum(len(paths) for paths in copied_dataset["classes"].values())
        copied_dataset["num_classes"] = len(copied_dataset["classes"])

        return copied_dataset