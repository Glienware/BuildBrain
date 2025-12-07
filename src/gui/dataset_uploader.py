"""
Dataset Uploader Component

Handles uploading image datasets organized by classes.
"""

import flet as ft
import os
import json


class DatasetUploader:
    """
    File picker for image datasets organized by classes.
    """

    def __init__(self, page: ft.Page, existing_classes=None, on_update=None):
        self.page = page
        self.file_picker = ft.FilePicker(on_result=self.on_files_selected)
        self.folder_picker = ft.FilePicker(on_result=self.on_folder_selected)
        self.classes = {}  # {class_name: [image_paths]}
        self.class_cards = []
        self.on_update = on_update  # Callback to notify parent of updates
        self.dataset_path = None  # For compatibility with training system
        self.page.overlay.append(self.file_picker)
        self.page.overlay.append(self.folder_picker)

        # Initialize with existing classes if provided
        if existing_classes:
            for class_name in existing_classes:
                self.classes[class_name] = []

    def build(self):
        """Build the dataset uploader UI."""
        self.update_class_cards()

        content = [
            ft.Row([
                ft.Icon(ft.Icons.IMAGE, size=24),
                ft.Text("Dataset Classes", size=16, weight=ft.FontWeight.BOLD),
            ], alignment=ft.MainAxisAlignment.START),
        ]

        # Only show "Add new class" section if no existing classes
        if not self.classes:
            content.append(
                ft.Container(
                    content=ft.Row([
                        ft.TextField(
                            label="New Class Name",
                            width=200,
                            on_submit=self.add_class
                        ),
                        ft.ElevatedButton(
                            "Add Class",
                            icon=ft.Icons.ADD,
                            on_click=self.add_class,
                            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_400)
                        ),
                    ], alignment=ft.MainAxisAlignment.START),
                    margin=ft.margin.only(bottom=20)
                )
            )

        content.append(
            ft.Container(
                content=ft.Column(self.class_cards, spacing=15, scroll=ft.ScrollMode.AUTO),
                height=400,
            )
        )

        return ft.Container(
            content=ft.Column(content, spacing=15),
            padding=10
        )

    def add_class(self, e):
        """Add a new class."""
        class_name = ""
        if hasattr(e, 'control') and hasattr(e.control, 'value'):
            class_name = e.control.value.strip()
        elif isinstance(e, ft.ControlEvent) and hasattr(e, 'data'):
            class_name = e.data.strip()

        if class_name and class_name not in self.classes:
            self.classes[class_name] = []
            self.update_class_cards()
            if self.on_update:
                self.on_update()
            else:
                self.page.update()

    def update_class_cards(self):
        """Update the class cards display."""
        self.class_cards = []
        for class_name, images in self.classes.items():
            card = ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        # Class header
                        ft.Row([
                            ft.Text(class_name, size=18, weight=ft.FontWeight.BOLD),
                            ft.IconButton(
                                ft.Icons.DELETE,
                                on_click=lambda e, c=class_name: self.remove_class(c),
                                tooltip="Remove class"
                            )
                        ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),

                        # Image count
                        ft.Text(f"{len(images)} images", size=12, color=ft.Colors.GREY_600),

                        # Upload buttons
                        ft.Row([
                            ft.ElevatedButton(
                                "Select Images",
                                icon=ft.Icons.FILE_UPLOAD,
                                on_click=lambda e, c=class_name: self.select_images_for_class(c),
                                style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400, color=ft.Colors.WHITE)
                            ),
                            ft.ElevatedButton(
                                "Select Folder",
                                icon=ft.Icons.FOLDER_OPEN,
                                on_click=lambda e, c=class_name: self.select_folder_for_class(c),
                                style=ft.ButtonStyle(bgcolor=ft.Colors.ORANGE_400, color=ft.Colors.WHITE)
                            ),
                        ], spacing=10),

                        # Preview images (show first few)
                        ft.Container(
                            content=self.build_image_preview(class_name),
                            height=100,
                        )

                    ], spacing=10),
                    padding=20,
                    width=350
                ),
                elevation=3
            )
            self.class_cards.append(card)

    def build_image_preview(self, class_name):
        """Build preview of images for a class."""
        images = self.classes[class_name]
        if not images:
            return ft.Text("No images uploaded", size=12, color=ft.Colors.GREY_500)

        # Show first 4 images as thumbnails
        preview_images = []
        for i, img_path in enumerate(images[:4]):
            try:
                preview_images.append(
                    ft.Container(
                        content=ft.Image(
                            src=img_path,
                            width=60,
                            height=60,
                            fit=ft.ImageFit.COVER,
                            border_radius=5
                        ),
                        border=ft.border.all(1, ft.Colors.GREY_300),
                        border_radius=5
                    )
                )
            except:
                preview_images.append(
                    ft.Container(
                        content=ft.Icon(ft.Icons.IMAGE, size=30, color=ft.Colors.GREY_400),
                        width=60,
                        height=60,
                        border=ft.border.all(1, ft.Colors.GREY_300),
                        border_radius=5,
                        alignment=ft.alignment.center
                    )
                )

        return ft.Row(preview_images, spacing=5, scroll=ft.ScrollMode.AUTO)

    def remove_class(self, class_name):
        """Remove a class and its images."""
        if class_name in self.classes:
            del self.classes[class_name]
            self.update_class_cards()
            if self.on_update:
                self.on_update()
            else:
                self.page.update()

    def select_images_for_class(self, class_name):
        """Select multiple images for a class."""
        self.current_class = class_name
        self.file_picker.pick_files(
            allow_multiple=True,
            allowed_extensions=['png', 'jpg', 'jpeg', 'gif', 'bmp']
        )

    def select_folder_for_class(self, class_name):
        """Select a folder of images for a class."""
        self.current_class = class_name
        self.folder_picker.get_directory_path()

    def on_files_selected(self, e: ft.FilePickerResultEvent):
        """Handle multiple file selection."""
        if e.files and hasattr(self, 'current_class'):
            class_name = self.current_class
            if class_name not in self.classes:
                self.classes[class_name] = []

            for file in e.files:
                if file.path not in self.classes[class_name]:
                    self.classes[class_name].append(file.path)

            self.update_class_cards()
            if self.on_update:
                self.on_update()
            else:
                self.page.update()

    def on_folder_selected(self, e: ft.FilePickerResultEvent):
        """Handle folder selection."""
        if e.path and hasattr(self, 'current_class'):
            class_name = self.current_class
            if class_name not in self.classes:
                self.classes[class_name] = []

            # Add all image files from the folder
            import os
            for root, dirs, files in os.walk(e.path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        full_path = os.path.join(root, file)
                        if full_path not in self.classes[class_name]:
                            self.classes[class_name].append(full_path)

            self.update_class_cards()
            if self.on_update:
                self.on_update()
            else:
                self.page.update()

    def load_dataset_info(self, dataset_info):
        """Load dataset information from saved project data."""
        if dataset_info and 'classes' in dataset_info:
            self.classes = dataset_info['classes']
            self.update_class_cards()
            # Always update the page when loading dataset info
            self.page.update()