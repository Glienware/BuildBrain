"""
Logs and Visualization Component

Displays training logs, metrics, and plots.
"""

import flet as ft
import matplotlib.pyplot as plt
import io
import base64


class LogsVisualization:
    """
    Area for logs and training visualizations.
    """

    def __init__(self, page: ft.Page):
        self.page = page
        self.logs = ft.Text("", font_family="monospace", selectable=True)
        self.plot_area = ft.Container(height=300, bgcolor=ft.Colors.GREY_100)
        self.metrics_history = []  # Store training metrics for plotting

    def update_plot(self):
        """
        Update the plot area with current metrics.
        """
        if not self.metrics_history:
            return
            
        try:
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Group metrics by type
            accuracy_data = [m for m in self.metrics_history if m["type"] == "accuracy"]
            precision_data = [m for m in self.metrics_history if m["type"] == "precision"]
            recall_data = [m for m in self.metrics_history if m["type"] == "recall"]
            f1_data = [m for m in self.metrics_history if m["type"] == "f1"]
            
            if accuracy_data:
                ax.plot([m["step"] for m in accuracy_data], [m["value"] for m in accuracy_data], 
                       label='Accuracy', marker='o', color='#4CAF50')
            if precision_data:
                ax.plot([m["step"] for m in precision_data], [m["value"] for m in precision_data], 
                       label='Precision', marker='s', color='#2196F3')
            if recall_data:
                ax.plot([m["step"] for m in recall_data], [m["value"] for m in recall_data], 
                       label='Recall', marker='^', color='#FF9800')
            if f1_data:
                ax.plot([m["step"] for m in f1_data], [m["value"] for m in f1_data], 
                       label='F1-Score', marker='d', color='#9C27B0')
            
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Metric Value')
            ax.set_title('Training Metrics')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Convert to image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            # Update plot area
            self.plot_area.content = ft.Image(
                src_base64=image_base64,
                width=600,
                height=250,
                fit=ft.ImageFit.CONTAIN
            )
            
        except ImportError:
            self.plot_area.content = ft.Text("Install matplotlib for plots", color=ft.Colors.GREY_500)
        except Exception as e:
            self.plot_area.content = ft.Text(f"Plot error: {str(e)}", color=ft.Colors.RED)

    def add_log(self, message):
        """
        Add a message to the logs and update visualizations if it's a metric.
        """
        self.logs.value += message + "\n"
        
        # Try to extract metrics from the message
        if "Accuracy:" in message:
            try:
                acc = float(message.split("Accuracy:")[1].strip())
                self.metrics_history.append({"type": "accuracy", "value": acc, "step": len(self.metrics_history)})
                self.update_plot()
            except:
                pass
        elif "Precision:" in message:
            try:
                prec = float(message.split("Precision:")[1].strip())
                self.metrics_history.append({"type": "precision", "value": prec, "step": len(self.metrics_history)})
                self.update_plot()
            except:
                pass
        elif "Recall:" in message:
            try:
                rec = float(message.split("Recall:")[1].strip())
                self.metrics_history.append({"type": "recall", "value": rec, "step": len(self.metrics_history)})
                self.update_plot()
            except:
                pass
        elif "F1-Score:" in message:
            try:
                f1 = float(message.split("F1-Score:")[1].strip())
                self.metrics_history.append({"type": "f1", "value": f1, "step": len(self.metrics_history)})
                self.update_plot()
            except:
                pass
                
        self.page.update()

    def copy_logs(self, e):
        """
        Copy all logs to clipboard.
        """
        try:
            import pyperclip
            pyperclip.copy(self.logs.value)
            self.page.snack_bar = ft.SnackBar(ft.Text("Logs copied to clipboard!"))
            self.page.snack_bar.open = True
            self.page.update()
        except ImportError:
            # Fallback: use Flet's clipboard functionality if available
            try:
                self.page.set_clipboard(self.logs.value)
                self.page.snack_bar = ft.SnackBar(ft.Text("Logs copied to clipboard!"))
                self.page.snack_bar.open = True
                self.page.update()
            except Exception as ex:
                self.page.snack_bar = ft.SnackBar(ft.Text(f"Copy failed: {str(ex)}", color=ft.Colors.RED))
                self.page.snack_bar.open = True
                self.page.update()

    def build(self):
        """
        Build the logs and visualization UI.
        """
        return ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Icon(ft.Icons.ASSESSMENT, size=24),
                    ft.Text("Training Logs & Visualization", size=16, weight=ft.FontWeight.BOLD),
                    ft.IconButton(
                        icon=ft.Icons.CONTENT_COPY,
                        on_click=self.copy_logs,
                        tooltip="Copy logs to clipboard"
                    )
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Container(
                    content=self.logs,
                    height=200,
                    bgcolor=ft.Colors.BLACK,
                    border_radius=5,
                    padding=10
                ),
                ft.Container(
                    content=self.plot_area,
                    height=300,
                    border=ft.border.all(1, ft.Colors.OUTLINE),
                    border_radius=5,
                    padding=10
                )
            ], spacing=10),
            padding=10
        )