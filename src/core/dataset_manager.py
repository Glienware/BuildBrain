"""
Dataset Manager

Handles dataset organization, image management, and class operations.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import flet as ft


class DatasetManager:
    """
    Manages dataset structure, classes, and image operations.
    """

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, project_path: str, project_config=None):
        self.project_path = project_path
        self.dataset_path = os.path.join(project_path, "dataset")
        self.project_config = project_config
        self._ensure_dataset_structure()

    def _ensure_dataset_structure(self):
        """Ensure the dataset directory structure exists."""
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

    def get_classes_info(self) -> List[Dict[str, any]]:
        """Get information about all classes and their image counts."""
        classes_info = []

        if not os.path.exists(self.dataset_path):
            return classes_info

        for item in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, item)
            if os.path.isdir(class_path):
                image_count = self._count_images_in_class(item)
                classes_info.append({
                    "name": item,
                    "image_count": image_count,
                    "path": class_path
                })

        # If no classes in folder, try to load from config
        if not classes_info and self.project_config:
            dataset = self.project_config.config_data.get("dataset", {})
            if "classes" in dataset:
                for class_name, images in dataset["classes"].items():
                    classes_info.append({
                        "name": class_name,
                        "image_count": len(images),
                        "path": f"external:{class_name}"  # indicate external
                    })

        return sorted(classes_info, key=lambda x: x["name"])

    def add_class(self, class_name: str) -> bool:
        """Add a new class directory."""
        try:
            class_path = os.path.join(self.dataset_path, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
                return True
            return False  # Class already exists
        except Exception as e:
            print(f"Error adding class {class_name}: {e}")
            return False

    def remove_class(self, class_name: str) -> bool:
        """Remove a class and all its images."""
        try:
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.exists(class_path):
                shutil.rmtree(class_path)
                return True
            return False  # Class doesn't exist
        except Exception as e:
            print(f"Error removing class {class_name}: {e}")
            return False

    def rename_class(self, old_name: str, new_name: str) -> bool:
        """Rename a class directory."""
        try:
            old_path = os.path.join(self.dataset_path, old_name)
            new_path = os.path.join(self.dataset_path, new_name)

            if os.path.exists(old_path) and not os.path.exists(new_path):
                os.rename(old_path, new_path)
                return True
            return False
        except Exception as e:
            print(f"Error renaming class {old_name} to {new_name}: {e}")
            return False

    def add_images_to_class(self, class_name: str, image_paths: List[str]) -> Tuple[int, int]:
        """
        Add images to a class directory.
        Returns (success_count, error_count)
        """
        class_path = os.path.join(self.dataset_path, class_name)
        if not os.path.exists(class_path):
            self.add_class(class_name)

        success_count = 0
        error_count = 0

        for image_path in image_paths:
            try:
                if self._is_valid_image(image_path):
                    filename = self._generate_unique_filename(class_path, os.path.basename(image_path))
                    dest_path = os.path.join(class_path, filename)
                    shutil.copy2(image_path, dest_path)
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"Error copying image {image_path}: {e}")
                error_count += 1

        return success_count, error_count

    def add_images_from_folder(self, class_name: str, folder_path: str) -> Tuple[int, int]:
        """
        Add all images from a folder to a class.
        Returns (success_count, error_count)
        """
        if not os.path.exists(folder_path):
            return 0, 0

        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if self._is_valid_image(file):
                    image_paths.append(os.path.join(root, file))

        return self.add_images_to_class(class_name, image_paths)

    def remove_image_from_class(self, class_name: str, image_filename: str) -> bool:
        """Remove a specific image from a class."""
        try:
            image_path = os.path.join(self.dataset_path, class_name, image_filename)
            if os.path.exists(image_path):
                os.remove(image_path)
                return True
            return False
        except Exception as e:
            print(f"Error removing image {image_filename}: {e}")
            return False

    def get_class_images(self, class_name: str, limit: int = 50) -> List[Dict[str, any]]:
        """Get list of images in a class with metadata."""
        class_path = os.path.join(self.dataset_path, class_name)
        if os.path.exists(class_path):
            images = []
            for filename in os.listdir(class_path):
                if self._is_valid_image(filename):
                    filepath = os.path.join(class_path, filename)
                    try:
                        # Get image info
                        img = Image.open(filepath)
                        images.append({
                            "filename": filename,
                            "path": filepath,
                            "size": img.size,
                            "format": img.format,
                            "mode": img.mode
                        })
                        img.close()

                        if len(images) >= limit:
                            break
                    except Exception as e:
                        print(f"Error reading image {filename}: {e}")

            return images
        elif self.project_config:
            # Try from config
            dataset = self.project_config.config_data.get("dataset", {})
            if "classes" in dataset and class_name in dataset["classes"]:
                images = []
                for filepath in dataset["classes"][class_name][:limit]:
                    try:
                        img = Image.open(filepath)
                        images.append({
                            "filename": os.path.basename(filepath),
                            "path": filepath,
                            "size": img.size,
                            "format": img.format,
                            "mode": img.mode
                        })
                        img.close()
                    except Exception as e:
                        print(f"Error reading image {filepath}: {e}")
                return images
        return []

    def get_dataset_stats(self) -> Dict[str, any]:
        """Get overall dataset statistics."""
        classes_info = self.get_classes_info()
        total_images = sum(cls["image_count"] for cls in classes_info)
        total_classes = len(classes_info)

        return {
            "total_classes": total_classes,
            "total_images": total_images,
            "classes": classes_info,
            "avg_images_per_class": total_images / max(total_classes, 1)
        }

    def validate_dataset(self) -> List[str]:
        """Validate dataset structure and return issues."""
        issues = []

        if not os.path.exists(self.dataset_path):
            issues.append("Dataset directory does not exist")
            return issues

        classes_info = self.get_classes_info()

        if len(classes_info) == 0:
            issues.append("No classes found in dataset")

        for cls in classes_info:
            if cls["image_count"] == 0:
                issues.append(f"Class '{cls['name']}' has no images")

        return issues

    def _count_images_in_class(self, class_name: str) -> int:
        """Count valid images in a class directory."""
        class_path = os.path.join(self.dataset_path, class_name)
        if not os.path.exists(class_path):
            return 0

        count = 0
        for filename in os.listdir(class_path):
            if self._is_valid_image(filename):
                count += 1
        return count

    def _is_valid_image(self, filename: str) -> bool:
        """Check if a file is a valid image format."""
        return Path(filename).suffix.lower() in self.SUPPORTED_FORMATS

    def _generate_unique_filename(self, directory: str, original_filename: str) -> str:
        """Generate a unique filename to avoid conflicts."""
        base_name = Path(original_filename).stem
        extension = Path(original_filename).suffix
        counter = 1
        new_filename = original_filename

        while os.path.exists(os.path.join(directory, new_filename)):
            new_filename = f"{base_name}_{counter}{extension}"
            counter += 1

        return new_filename