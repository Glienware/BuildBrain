"""
Dataset Uploader Component

Handles uploading image datasets organized by classes.
"""

import flet as ft
import os
import json
import threading
import sys


class DatasetUploader:
    """
    File picker for image datasets organized by classes.
    """

    def __init__(self, page: ft.Page, existing_classes=None, on_update=None):
        self.page = page
        self.file_picker = ft.FilePicker(on_result=self.on_files_selected)
        self.page.overlay.append(self.file_picker)
        self.classes = {}  # {class_name: [image_paths]}
        self.class_cards = []
        self.on_update = on_update  # Callback to notify parent of updates
        self.dataset_path = None  # For compatibility with training system
        self.current_class = None  # Track which class is being edited
        self.selecting_folder = False  # Flag to indicate folder selection mode

        # Initialize with existing classes if provided
        if existing_classes:
            for class_name in existing_classes:
                self.classes[class_name] = []

    def build(self):
        """Build the dataset uploader UI with 2-column grid."""
        self.update_class_cards()

        content = [
            ft.Row([
                ft.Icon(ft.Icons.IMAGE, size=24),
                ft.Text("Dataset Classes", size=16, weight=ft.FontWeight.BOLD),
            ], alignment=ft.MainAxisAlignment.START),
        ]

        # Add quick load option for CSV files
        content.append(
            ft.Container(
                content=ft.Row([
                    ft.ElevatedButton(
                        "Load CSV Dataset",
                        icon=ft.Icons.UPLOAD_FILE,
                        on_click=self.select_csv_file,
                        style=ft.ButtonStyle(bgcolor="#FFB74D", color=ft.Colors.BLACK)
                    ),
                    ft.Text("Load data from CSV/Excel", size=12, color="#AAAAAA"),
                ], spacing=10, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                margin=ft.margin.only(bottom=20)
            )
        )

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

        # Build 3-column grid of class cards
        class_grid_rows = []
        for i in range(0, len(self.class_cards), 3):
            row_items = []
            
            # Add up to 3 cards per row
            for j in range(3):
                if i + j < len(self.class_cards):
                    row_items.append(self.class_cards[i + j])
                else:
                    row_items.append(ft.Container(expand=True))
            
            class_grid_rows.append(
                ft.Row(
                    controls=row_items,
                    spacing=12,
                    expand=True,
                )
            )

        content.append(
            ft.Container(
                content=ft.Column(class_grid_rows, spacing=12),
            )
        )

        return ft.Container(
            content=ft.Column(content, spacing=15),
            padding=10
        )

    def add_class(self, e):
        """Add a new class."""
        class_name = ""
        
        # Try to get class name from different event types
        if hasattr(e, 'control') and hasattr(e.control, 'value'):
            class_name = e.control.value
            if class_name:
                class_name = class_name.strip()
        elif hasattr(e, 'data') and e.data is not None:
            class_name = str(e.data).strip()

        if class_name and class_name not in self.classes:
            self.classes[class_name] = []
            self.update_class_cards()
            if self.on_update:
                self.on_update()
            else:
                try:
                    self.page.update()
                except:
                    pass

    def update_class_cards(self):
        """Update the class cards display."""
        self.class_cards = []
        print(f"DEBUG: Updating class cards. Classes: {list(self.classes.keys())}")
        for class_name, images in self.classes.items():
            print(f"DEBUG: Class '{class_name}' has {len(images)} images")
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
                                style=ft.ButtonStyle(bgcolor="#82B1FF", color=ft.Colors.BLACK)
                            ),
                            ft.ElevatedButton(
                                "Select Folder",
                                icon=ft.Icons.FOLDER_OPEN,
                                on_click=lambda e, c=class_name: self.select_folder_for_class(c),
                                style=ft.ButtonStyle(bgcolor="#FFB74D", color=ft.Colors.BLACK)
                            ),
                        ], spacing=10),

                        # Preview images (show first few)
                        ft.Container(
                            content=self.build_image_preview(class_name),
                            height=100,
                        )

                    ], spacing=10),
                    padding=20,
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

    def select_csv_file(self, e):
        """Select a CSV or Excel file with dataset."""
        try:
            self.file_picker.pick_files(
                allowed_extensions=['csv', 'xlsx', 'xls'],
                dialog_title="Select CSV or Excel file"
            )
        except Exception as ex:
            print(f"Error opening file picker: {ex}")

    def select_images_for_class(self, class_name):
        """Select multiple images for a class."""
        self.current_class = class_name
        try:
            self.file_picker.pick_files(
                allow_multiple=True,
                allowed_extensions=['png', 'jpg', 'jpeg', 'gif', 'bmp']
            )
        except Exception as ex:
            print(f"Error opening file picker: {ex}")

    def select_folder_for_class(self, class_name):
        """Select a folder of images for a class."""
        self.current_class = class_name
        self.selecting_folder = True
        
        # Run folder selection in a separate thread (NOT daemon)
        thread = threading.Thread(
            target=self._select_folder_thread,
            args=(class_name,)
        )
        thread.start()

    def _select_folder_thread(self, class_name):
        """Thread function to handle folder selection using Windows API."""
        try:
            if sys.platform == 'win32':
                # Use Windows native folder picker via PowerShell
                import subprocess
                
                ps_command = f"""
[System.Reflection.Assembly]::LoadWithPartialName("System.windows.forms") | Out-Null;
$folder = New-Object System.Windows.Forms.FolderBrowserDialog;
$folder.Description = "Select folder with images for '{class_name}'";
$folder.ShowNewFolderButton = $true;
$result = $folder.ShowDialog();
if ($result -eq "OK") {{ 
    Write-Host $folder.SelectedPath 
}}
"""
                
                result = subprocess.run(
                    ['powershell', '-NoProfile', '-Command', ps_command],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                folder_path = result.stdout.strip()
                print(f"DEBUG: PowerShell returned: '{folder_path}'")
                
                if folder_path and os.path.exists(folder_path):
                    print(f"DEBUG: Folder exists, loading images")
                    self.load_images_from_folder(class_name, folder_path)
                else:
                    print(f"DEBUG: No folder selected or path invalid: {folder_path}")
            else:
                # For Linux/Mac, use zenity or similar
                import subprocess
                result = subprocess.run([
                    'zenity', '--file-selection', '--directory',
                    '--title=Select folder with images for ' + class_name
                ], capture_output=True, text=True)
                
                folder_path = result.stdout.strip()
                if folder_path and os.path.exists(folder_path):
                    self.load_images_from_folder(class_name, folder_path)
            
        except Exception as ex:
            print(f"Error in folder selection: {ex}")
            import traceback
            traceback.print_exc()

    def on_files_selected(self, e: ft.FilePickerResultEvent):
        """Handle multiple file selection - can be images or CSV."""
        if not e.files:
            return
        
        try:
            # Check if CSV/Excel files are selected
            csv_extensions = ('.csv', '.xlsx', '.xls')
            is_csv = any(file.path.lower().endswith(csv_extensions) for file in e.files)
            
            if is_csv:
                # Handle CSV/Excel file
                if e.files:
                    csv_file = e.files[0].path
                    self.dataset_path = csv_file
                    print(f"CSV Dataset loaded: {csv_file}")
                    
                    # Try to auto-detect classes from CSV
                    try:
                        import pandas as pd
                        if csv_file.endswith('.csv'):
                            df = pd.read_csv(csv_file)
                        else:
                            df = pd.read_excel(csv_file)
                        
                        self.classes['data'] = [csv_file]
                        self.dataset_path = csv_file
                        
                        print(f"CSV Info: {len(df)} rows, {len(df.columns)} columns")
                        print(f"Columns: {list(df.columns)}")
                        
                    except Exception as ex:
                        print(f"Error reading CSV: {ex}")
                    
                    self.update_class_cards()
                    if self.on_update:
                        self.on_update()
                    else:
                        try:
                            self.page.update()
                        except:
                            pass
            else:
                # Handle image files
                if self.current_class:
                    class_name = self.current_class
                    if class_name not in self.classes:
                        self.classes[class_name] = []

                    # If selecting folder, get the parent directory path
                    if self.selecting_folder and e.files:
                        # Get common directory from selected files
                        first_file = e.files[0].path
                        folder_path = os.path.dirname(first_file)
                        
                        # Load ALL images from that folder
                        image_count = 0
                        for file in e.files:
                            if file.path and file.path not in self.classes[class_name]:
                                self.classes[class_name].append(file.path)
                                image_count += 1
                        
                        print(f"Loaded {image_count} images from folder to class '{class_name}'")
                        self.dataset_path = folder_path
                        self.selecting_folder = False
                    else:
                        # Regular image selection (one by one)
                        image_count = 0
                        for file in e.files:
                            if file.path and file.path not in self.classes[class_name]:
                                self.classes[class_name].append(file.path)
                                image_count += 1

                        print(f"Added {image_count} images to class '{class_name}'")
                        self.dataset_path = class_name
                    
                    self.update_class_cards()
                    if self.on_update:
                        self.on_update()
                    else:
                        try:
                            self.page.update()
                        except:
                            pass
                else:
                    print("No class selected for images")
                    
        except Exception as ex:
            print(f"Error in on_files_selected: {ex}")
            import traceback
            traceback.print_exc()

    def on_folder_selected(self, e: ft.FilePickerResultEvent):
        """Handle folder selection (legacy)."""
        if e.path and hasattr(self, 'current_class'):
            self.load_images_from_folder(self.current_class, e.path)

    def load_images_from_folder(self, class_name, folder_path):
        """Load all images from a folder."""
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return
        
        if class_name not in self.classes:
            self.classes[class_name] = []

        # Add all image files from the folder (recursive search in subfolders too)
        image_count = 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    full_path = os.path.join(root, file)
                    if full_path not in self.classes[class_name]:
                        self.classes[class_name].append(full_path)
                        image_count += 1

        print(f"Loaded {image_count} images for class '{class_name}' from {folder_path}")
        print(f"DEBUG: Current classes state: {list(self.classes.keys())}")
        for cls, imgs in self.classes.items():
            print(f"DEBUG: {cls} -> {len(imgs)} images")
        
        # Store dataset path for training
        self.dataset_path = folder_path
        
        # Force update immediately
        print("DEBUG: Calling update_class_cards()")
        self.update_class_cards()
        
        # Also call the callback if available
        if self.on_update:
            try:
                print("DEBUG: Calling on_update callback")
                self.on_update()
            except Exception as e:
                print(f"Error calling on_update callback: {e}")
        
        # Try page update as well
        try:
            print("DEBUG: Calling page.update()")
            self.page.update()
        except Exception as e:
            print(f"Error updating page: {e}")

    def load_dataset_info(self, dataset_info):
        """Load dataset information from saved project data."""
        if dataset_info and 'classes' in dataset_info:
            self.classes = dataset_info['classes']
            self.update_class_cards()
            if self.on_update:
                self.on_update()
            else:
                self.page.update()

    def get_dataset_info(self):
        """Get information about the current dataset."""
        total_images = sum(len(images) for images in self.classes.values())
        return {
            "classes": self.classes,
            "total_images": total_images,
            "num_classes": len(self.classes)
        }