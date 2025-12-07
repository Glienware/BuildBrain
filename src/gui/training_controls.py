"""
Training Controls Component

Buttons for starting training and progress display.
"""

import flet as ft


class TrainingControls:
    """
    Controls for starting training with progress indicators.
    """

    def __init__(self, page: ft.Page, components, logs_callback):
        self.page = page
        self.components = components
        self.logs_callback = logs_callback
        self.trainer = None
        
        self.quick_train_btn = ft.ElevatedButton(
            "Quick Train", 
            icon=ft.Icons.FLASH_ON,
            on_click=self.start_quick_train,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_400)
        )
        self.advanced_train_btn = ft.ElevatedButton(
            "Advanced Train", 
            icon=ft.Icons.SCHOOL,
            on_click=self.start_advanced_train,
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400)
        )
        self.progress_bar = ft.ProgressBar(width=400, visible=False)
        self.progress_text = ft.Text("", visible=False)
        self.export_btn = ft.ElevatedButton(
            "Export Model",
            icon=ft.Icons.DOWNLOAD,
            on_click=self.export_model,
            disabled=True,
            style=ft.ButtonStyle(bgcolor=ft.Colors.PURPLE_400)
        )

    def start_quick_train(self, e):
        """
        Start quick training with fast preset.
        """
        if not self._validate_training_setup():
            return
            
        # Quick preset
        quick_params = {
            'max_iter': 1000,
            'random_state': 42
        }
        
        self._start_training(quick_params)

    def start_advanced_train(self, e):
        """
        Start advanced training with current settings.
        """
        if not self._validate_training_setup():
            return
            
        # Get current hyperparameters
        params = {}
        settings = self.components["settings_panel"]
        
        if settings.learning_rate.value:
            params['learning_rate'] = float(settings.learning_rate.value)
        if settings.batch_size.value:
            params['batch_size'] = int(settings.batch_size.value)
        if settings.epochs.value:
            params['epochs'] = int(settings.epochs.value)
        if settings.optimizer.value:
            params['optimizer'] = settings.optimizer.value
        if settings.weight_decay.value:
            params['weight_decay'] = float(settings.weight_decay.value)
        if settings.max_depth.value:
            params['max_depth'] = int(settings.max_depth.value)
        if settings.n_estimators.value:
            params['n_estimators'] = int(settings.n_estimators.value)
            
        self._start_training(params)

    def _validate_training_setup(self):
        """
        Validate that training can start.
        """
        model = self.components["model_selector"].selected_model.value
        task = self.components["task_selector"].task_type.value
        dataset = self.components["dataset_uploader"].dataset_path
        
        if not model:
            self._show_error("Please select a model")
            return False
        if not task:
            self._show_error("Please select a task type")
            return False
        if not dataset:
            self._show_error("Please upload a dataset")
            return False
            
        return True

    def _start_training(self, params):
        """
        Start the actual training process.
        """
        from ..training.trainer import Trainer
        
        self.progress_bar.visible = True
        self.progress_text.visible = True
        self.progress_text.value = "Initializing training..."
        self.quick_train_btn.disabled = True
        self.advanced_train_btn.disabled = True
        self.export_btn.disabled = True
        
        model = self.components["model_selector"].selected_model.value
        task = self.components["task_selector"].task_type.value
        dataset = self.components["dataset_uploader"].dataset_path
        
        # Get project directory from project manager
        project_dir = "projects"  # default
        if "project_manager" in self.components:
            # Use projects directory for now
            project_dir = "projects"
        
        self.trainer = Trainer(
            model_type=model,
            task_type=task,
            hyperparameters=params,
            dataset_path=dataset,
            log_callback=self.logs_callback,
            project_dir=project_dir
        )
        
        # Start training in thread
        import threading
        thread = threading.Thread(target=self._run_training)
        thread.start()
        
        self.page.update()

    def _run_training(self):
        """
        Run training in background thread.
        """
        try:
            self.trainer.start_training()
            # Wait for completion
            while self.trainer.is_training:
                import time
                time.sleep(0.1)
            
            self.page.snack_bar = ft.SnackBar(ft.Text("Training completed!"))
            self.page.snack_bar.open = True
            self.export_btn.disabled = False
            
            # Save model path to project if available
            if self.trainer.saved_model_path and "project_manager" in self.components:
                self.components["project_manager"].save_trained_model(self.trainer.saved_model_path)
            
        except Exception as e:
            self._show_error(f"Training failed: {str(e)}")
        finally:
            self.progress_bar.visible = False
            self.progress_text.visible = False
            self.quick_train_btn.disabled = False
            self.advanced_train_btn.disabled = False
            self.page.update()

    def export_model(self, e):
        """
        Export the trained model.
        """
        if not self.trainer or not self.trainer.model:
            self._show_error("No trained model to export")
            return
            
        try:
            import joblib
            import os
            
            # Create exports directory
            os.makedirs("exports", exist_ok=True)
            
            model_name = f"model_{self.components['model_selector'].selected_model.value.lower()}"
            
            # Export with joblib
            joblib_path = f"exports/{model_name}.joblib"
            joblib.dump(self.trainer.model, joblib_path)
            
            # Export with ONNX if possible
            try:
                import onnxruntime
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                
                # Get sample input for ONNX
                import pandas as pd
                df = pd.read_csv(self.components["dataset_uploader"].dataset_path)
                if 'target' in df.columns:
                    sample_input = df.drop(columns=['target']).iloc[:1].values.astype('float32')
                else:
                    sample_input = df.iloc[:1, :-1].values.astype('float32')
                
                initial_type = [('float_input', FloatTensorType([None, sample_input.shape[1]]))]
                onnx_model = convert_sklearn(self.trainer.model, initial_types=initial_type)
                
                onnx_path = f"exports/{model_name}.onnx"
                with open(onnx_path, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                    
                self.logs_callback(f"Model exported to: {joblib_path} and {onnx_path}")
                
            except ImportError:
                self.logs_callback(f"Model exported to: {joblib_path} (ONNX export requires skl2onnx)")
            except Exception as ex:
                self.logs_callback(f"Model exported to: {joblib_path} (ONNX export failed: {str(ex)})")
                
        except Exception as ex:
            self._show_error(f"Export failed: {str(ex)}")

    def _show_error(self, message):
        """
        Show error message.
        """
        self.page.snack_bar = ft.SnackBar(ft.Text(message, color=ft.Colors.RED))
        self.page.snack_bar.open = True
        self.page.update()

    def build(self):
        """
        Build the training controls UI.
        """
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.PLAY_CIRCLE, size=24),
                    ft.Text("Training Controls", size=16, weight=ft.FontWeight.BOLD),
                ], alignment=ft.MainAxisAlignment.START),
                ft.Row([
                    ft.ElevatedButton(
                        "Quick Train",
                        icon=ft.Icons.FLASH_ON,
                        on_click=self.start_quick_train,
                        style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_400)
                    ),
                    ft.ElevatedButton(
                        "Advanced Train",
                        icon=ft.Icons.SCHOOL,
                        on_click=self.start_advanced_train,
                        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_400)
                    ),
                    self.export_btn
                ], spacing=10),
                ft.Column([
                    ft.Text("Progress:", size=14),
                    self.progress_bar,
                    self.progress_text
                ], spacing=5)
            ], spacing=10),
            padding=10
        )