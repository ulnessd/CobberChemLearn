# cobber_residue.py
# An application to visualize model performance by plotting
# predicted vs. actual values and their residuals.
# Refactored for the CobberLearnChem launcher.

import sys
import os
import pandas as pd
import numpy as np

# --- Add project root to path for robust imports ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('.')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QTextEdit, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor  # Added for branding
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PlotCanvas(FigureCanvas):
    """A helper class for a Matplotlib canvas."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        FigureCanvas.updateGeometry(self)


# --- RENAMED class for consistency ---
class CobberResidueApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- BRANDING ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberResidue")
        self.setGeometry(100, 100, 1400, 800)
        self.setFont(self.lato_font)  # Apply base font

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # --- Top Controls ---
        top_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Dataset")
        # --- BRANDING ---
        self.load_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.cobber_maroon.name()};
                color: white;
                font-size: 18px;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }}
            #QPushButton:hover {{
             #   background-color: #FFD700;
            }}
        """)
        self.load_button.clicked.connect(self.load_csv)

        self.info_label = QLabel(" \nPlease load a CSV dataset containing 'Actual' and 'Predicted' columns.")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.info_label.setStyleSheet("font-size: 12pt; font-weight: bold;")

        top_layout.addWidget(self.load_button)
        top_layout.addWidget(self.info_label, 1)
        self.layout.addLayout(top_layout)

        # --- Main Content Area (Plots and Metrics) ---
        main_content_layout = QHBoxLayout()
        self.layout.addLayout(main_content_layout)

        # Plots Layout
        plots_layout = QHBoxLayout()
        self.pv_a_plot = PlotCanvas(self, width=6, height=5)
        self.residuals_plot = PlotCanvas(self, width=6, height=5)
        plots_layout.addWidget(self.pv_a_plot)
        plots_layout.addWidget(self.residuals_plot)

        # Metrics Layout
        metrics_layout = QVBoxLayout()
        metrics_label = QLabel("Model Evaluation Metrics:")
        metrics_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        metrics_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumWidth(250)
        self.metrics_text.setMaximumWidth(350)
        self.metrics_text.setFont(QFont("Lato", 12))  # BRANDING

        metrics_layout.addWidget(metrics_label)
        metrics_layout.addWidget(self.metrics_text)

        main_content_layout.addLayout(plots_layout, 3)
        main_content_layout.addLayout(metrics_layout, 1)

        # Initialize Data
        self.data = None
        self.clear_plots()

    def load_csv(self):
        """Opens a file dialog to load and process a CSV file."""
        # --- PATH IMPROVEMENT: Default to assets directory ---
        assets_dir = os.path.join(project_root, "assets")
        file_name, _ = QFileDialog.getOpenFileName(self, "Open CSV Dataset", assets_dir,
                                                   "CSV Files (*.csv);;All Files (*)")

        if file_name:
            try:
                self.data = pd.read_csv(file_name)
                # Make column names lowercase for consistency
                self.data.columns = self.data.columns.str.lower()

                # Check for required columns (case-insensitive)
                if 'actual' not in self.data.columns or 'predicted' not in self.data.columns:
                    self.info_label.setText(
                        "<font color='red'>Error: CSV must contain 'Actual' and 'Predicted' columns.</font>")
                    self.data = None
                    return

                self.info_label.setText(f"Loaded: <b>{os.path.basename(file_name)}</b>")
                self.plot_data()
            except Exception as e:
                self.info_label.setText(f"<font color='red'>Error loading CSV: {e}</font>")
                self.data = None

    def plot_data(self):
        """Calculates metrics and updates plots based on loaded data."""
        if self.data is None:
            self.clear_plots()
            return

        # Use lowercase column names now
        actual = self.data['actual'].values
        predicted = self.data['predicted'].values
        residuals = predicted - actual

        # Compute Metrics
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        # Update Metrics Display
        metrics_str = f"""
        <style>
            /* Basic styling for the table */
            table {{
                width: 100%;
                font-size: 11pt; /* Base font size */
            }}
            td.label {{
                padding-right: 10px; /* Space between label and value */
                padding-bottom: 15px; /* Space between metric rows */
            }}
            td.value {{
                font-family: Monospace; /* Gives numbers a clean, aligned look */
                font-size: 14pt;
                font-weight: bold;
            }}
        </style>
        <table>
            <tr>
                <td class="label">Mean Absolute Error (MAE):</td>
                <td class="value">{mae:.4f}</td>
            </tr>
            <tr>
                <td class="label">Mean Squared Error (MSE):</td>
                <td class="value">{mse:.4f}</td>
            </tr>
            <tr>
                <td class="label">R-squared (R²):</td>
                <td class="value">{r2:.4f}</td>
            </tr>
        </table>
        """
        self.metrics_text.setHtml(metrics_str)

        # --- Plot Predicted vs Actual ---
        self.pv_a_plot.figure.clear()
        ax1 = self.pv_a_plot.figure.add_subplot(111)
        ax1.scatter(actual, predicted, alpha=0.7, label="Data Points",s=7)
        lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]), max(ax1.get_xlim()[1], ax1.get_ylim()[1])]
        ax1.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Ideal Fit")
        ax1.set_xlabel("Actual Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title("Predicted vs. Actual Values")
        ax1.legend()
        ax1.grid(True)
        self.pv_a_plot.draw()

        # --- Plot Residuals ---
        self.residuals_plot.figure.clear()
        ax2 = self.residuals_plot.figure.add_subplot(111)
        ax2.scatter(actual, residuals, alpha=0.7, label="Residuals",s=7)
        ax2.axhline(0, color='r', linestyle='--', label="Zero Error")
        ax2.set_xlabel("Actual Values")
        ax2.set_ylabel("Residual (Predicted - Actual)")
        ax2.set_title("Residual Plot")
        ax2.legend()
        ax2.grid(True)
        self.residuals_plot.draw()

    def clear_plots(self):
        """Clears plots and shows placeholder text."""
        for canvas, title in [(self.pv_a_plot, "Predicted vs. Actual"), (self.residuals_plot, "Residuals")]:
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Load a CSV to generate plot",
                    ha='center', va='center', fontsize=12, alpha=0.5)
            ax.set_title(title)
            canvas.draw()
        self.metrics_text.setHtml("Load a dataset to see model<br>performance metrics.")


# --- Standalone Execution Guard ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberResidueApp()  # Use the new class name
    window.show()
    sys.exit(app.exec())