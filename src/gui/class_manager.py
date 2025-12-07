"""
Class Manager Component

Manages class labels for classification tasks.
"""

import flet as ft


class ClassManager:
    """
    Interface for adding/removing class labels and mapping.
    """

    def __init__(self, page: ft.Page):
        self.page = page
        self.classes = []
        self.class_list = ft.Column()
        self.new_class_input = ft.TextField(label="New Class Label", width=200)
        self.add_button = ft.ElevatedButton("Add Class", on_click=self.add_class)
        
        # Balance options
        self.balance_method = ft.Dropdown(
            label="Balance Method",
            options=[
                ft.dropdown.Option("None"),
                ft.dropdown.Option("Oversample (SMOTE)"),
                ft.dropdown.Option("Undersample")
            ],
            value="None",
            width=200
        )
        self.balance_btn = ft.ElevatedButton("Apply Balance", on_click=self.apply_balance)

    def add_class(self, e):
        """
        Add a new class label.
        """
        if self.new_class_input.value:
            self.classes.append(self.new_class_input.value)
            self.update_class_list()
            self.new_class_input.value = ""
            self.page.update()

    def apply_balance(self, e):
        """
        Apply class balancing to the dataset.
        """
        if self.balance_method.value == "None":
            self.page.snack_bar = ft.SnackBar(ft.Text("No balancing method selected"))
            self.page.snack_bar.open = True
            self.page.update()
            return
            
        # This would need access to the dataset, for now just show message
        method = "oversampling" if "Oversample" in self.balance_method.value else "undersampling"
        self.page.snack_bar = ft.SnackBar(ft.Text(f"Class {method} applied (will be implemented during training)"))
        self.page.snack_bar.open = True
        self.page.update()

    def update_class_list(self):
        """
        Update the displayed list of classes.
        """
        self.class_list.controls = [
            ft.Row([
                ft.Text(cls),
                ft.ElevatedButton("Remove", on_click=lambda e, c=cls: self.remove_class(c))
            ]) for cls in self.classes
        ]

    def build(self):
        """
        Build the class manager UI.
        """
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.LABEL, size=24),
                    ft.Text("Class Labels (Classification)", size=16, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.START),
                ft.Row([self.new_class_input, self.add_button], spacing=10),
                ft.Container(
                    content=self.class_list,
                    height=150,
                    border=ft.border.all(1, ft.Colors.OUTLINE),
                    border_radius=5,
                    padding=10
                ),
                ft.Divider(),
                ft.Text("Class Balancing", size=14, weight=ft.FontWeight.BOLD),
                ft.Row([self.balance_method, self.balance_btn], spacing=10)
            ], spacing=10),
            padding=10
        )