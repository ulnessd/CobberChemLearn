# CobberLand.py
# An application for exploring various regression models on the "Cobberland" dataset.
# Refactored for the CobberLearnChem launcher.

import sys
import os
import time
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QSlider, QLineEdit,
    QMessageBox, QStatusBar
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor  # Added for branding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Add project root to path for robust imports ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('.')


class PlotCanvas(FigureCanvas):
    """A helper class for an embedded Matplotlib canvas."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


# --- RENAMED Main Application Class ---
class CobberLandApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- BRANDING ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberLand")
        self.setGeometry(100, 100, 1400, 700)
        self.setFont(self.lato_font)

        # Data and model placeholders
        self.data = None
        self.model = None

        # --- Main Layout ---
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        # --- Left Panel (Controls) ---
        self.create_controls_panel()

        # --- Right Panel (Plots and Results) ---
        self.create_results_panel()

        # Add panels to main layout
        self.main_layout.addWidget(self.controls_widget, 1)
        self.main_layout.addWidget(self.results_widget, 3)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready. Please load the KernelData.csv file.")

    def create_controls_panel(self):
        """Creates the left-hand side panel with all user controls."""
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_widget.setMaximumWidth(350)

        # File loading
        self.open_button = QPushButton("Load KernelData.csv")
        self.open_button.clicked.connect(self.open_file)

        # Algorithm selection
        alg_label = QLabel("<b>1. Select ML Algorithm:</b>")
        self.alg_combo = QComboBox()
        self.algorithms = ["Linear Regression", "Decision Tree", "Random Forest", "Support Vector Machine",
                           "k-Nearest Neighbors"]
        self.alg_combo.addItems(self.algorithms)

        # Train/Test Split
        split_label = QLabel("<b>2. Set Training Data Percentage:</b>")
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(50, 90);
        self.split_slider.setValue(80)
        self.split_slider.setTickInterval(10);
        self.split_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.split_value_label = QLabel(f"{self.split_slider.value()}%")
        self.split_slider.valueChanged.connect(lambda v: self.split_value_label.setText(f"{v}%"))

        # Run Training Button
        self.train_button = QPushButton("Run ML Training")
        self.train_button.clicked.connect(self.run_training)

        # Manual Prediction
        pred_label = QLabel("<b>3. Predict STALKiness Manually:</b>")
        self.ear_entry = QLineEdit();
        self.ear_entry.setPlaceholderText("Enter EARitability")
        self.amaize_entry = QLineEdit();
        self.amaize_entry.setPlaceholderText("Enter aMAIZEingness")
        self.pred_button = QPushButton("Predict STALKiness")
        self.pred_button.clicked.connect(self.predict_y)
        self.prediction_result_label = QLabel("Prediction: -");
        self.prediction_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- BRANDING ---
       # button_style = f"background-color: {self.cobber_gold.name()}; color: {self.cobber_maroon.name()}; font-weight: bold; padding: 5px; border-radius: 3px;"
       # self.open_button.setStyleSheet(button_style)
       # self.train_button.setStyleSheet(button_style)

        # Add widgets to layout
        controls_layout.addWidget(self.open_button);
        controls_layout.addSpacing(20)
        controls_layout.addWidget(alg_label);
        controls_layout.addWidget(self.alg_combo);
        controls_layout.addSpacing(20)
        controls_layout.addWidget(split_label);
        controls_layout.addWidget(self.split_slider)
        controls_layout.addWidget(self.split_value_label, alignment=Qt.AlignmentFlag.AlignCenter);
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.train_button);
        controls_layout.addSpacing(30)
        controls_layout.addWidget(pred_label);
        controls_layout.addWidget(self.ear_entry)
        controls_layout.addWidget(self.amaize_entry);
        controls_layout.addWidget(self.pred_button);
        controls_layout.addWidget(self.prediction_result_label)
        controls_layout.addStretch()

    def create_results_panel(self):
        """Creates the right-hand side panel for plots and metrics."""
        self.results_widget = QWidget()
        results_layout = QVBoxLayout(self.results_widget)
        plots_layout = QHBoxLayout()
        self.pv_a_plot = PlotCanvas(self);
        self.residuals_plot = PlotCanvas(self)
        plots_layout.addWidget(self.pv_a_plot);
        plots_layout.addWidget(self.residuals_plot)
        metrics_label = QLabel("<b>Model Evaluation Metrics:</b>")
        self.metrics_display = QLabel("Train a model to see results.")
        self.metrics_display.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ccc; padding: 10px; border-radius: 5px;")
        self.metrics_display.setAlignment(Qt.AlignmentFlag.AlignTop)
        results_layout.addLayout(plots_layout);
        results_layout.addWidget(metrics_label);
        results_layout.addWidget(self.metrics_display)
        self.clear_plots()

    # --- All other methods have UNCHANGED logic ---

    def open_file(self):
        # --- PATH IMPROVEMENT ---
        assets_dir = os.path.join(project_root, "assets")
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", assets_dir, "CSV Files (*.csv)")
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                required_cols = {'EARitability', 'aMAIZEingness', 'STALKiness'}
                if not required_cols.issubset(self.data.columns):
                    raise ValueError(f"CSV must contain the columns: {', '.join(required_cols)}")
                self.statusBar().showMessage(f"Successfully loaded {os.path.basename(file_path)}", 5000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file: {e}")
                self.data = None

    def run_training(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load the KernelData.csv file first.")
            return
        try:
            X = self.data[['EARitability', 'aMAIZEingness']];
            y = self.data['STALKiness']
            train_size = self.split_slider.value() / 100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
            alg_name = self.alg_combo.currentText()
            model_map = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Support Vector Machine": SVR(),
                "k-Nearest Neighbors": KNeighborsRegressor()
            }
            self.model = model_map[alg_name]
            start_time = time.time()
            self.model.fit(X_train, y_train)
            training_time = time.time() - start_time
            y_pred = self.model.predict(X_test)
            self.update_plots(y_test, y_pred, alg_name)
            self.update_metrics(y_test, y_pred, training_time)
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"An error occurred during training: {e}")

    def update_plots(self, y_test, y_pred, alg_name):
        residuals = y_pred - y_test
        self.pv_a_plot.figure.clear();
        ax1 = self.pv_a_plot.figure.add_subplot(111)
        ax1.scatter(y_test, y_pred, alpha=0.7, s=1)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax1.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        ax1.set_xlabel("Actual STALKiness");
        ax1.set_ylabel("Predicted STALKiness");
        ax1.set_title(f"{alg_name}\nActual vs. Predicted");
        ax1.grid(True)
        self.pv_a_plot.draw()
        self.residuals_plot.figure.clear();
        ax2 = self.residuals_plot.figure.add_subplot(111)
        ax2.scatter(y_test, residuals, alpha=0.7, s=1)
        ax2.axhline(0, color='r', linestyle='--');
        ax2.set_xlabel("Actual STALKiness");
        ax2.set_ylabel("Residuals");
        ax2.set_title("Residuals Plot");
        ax2.grid(True)
        self.residuals_plot.draw()

    def update_metrics(self, y_test, y_pred, training_time):
        mae = mean_absolute_error(y_test, y_pred);
        mse = mean_squared_error(y_test, y_pred);
        r2 = r2_score(y_test, y_pred)
        metrics_html = f"""
        <b>Training Time:</b> {training_time:.4f} seconds<br><br>
        <b>Mean Absolute Error (MAE):</b> {mae:.4f}<br>
        <b>Mean Squared Error (MSE):</b> {mse:.4f}<br>
        <b>R-squared (R²):</b> {r2:.4f}
        """
        self.metrics_display.setText(metrics_html)

    def predict_y(self):
        if self.model is None: QMessageBox.warning(self, "No Model", "Please train a model first."); return
        try:
            ear = float(self.ear_entry.text());
            amaize = float(self.amaize_entry.text())
            prediction = self.model.predict([[ear, amaize]])[0]
            self.prediction_result_label.setText(f"<b>Prediction: {prediction:.3f}</b>")
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please enter valid numbers for EARitability and aMAIZEingness.")

    def clear_plots(self):
        for canvas, title in [(self.pv_a_plot, "Actual vs. Predicted"), (self.residuals_plot, "Residuals")]:
            canvas.figure.clear();
            ax = canvas.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Train a model to see plot", ha='center', va='center')
            ax.set_title(title);
            canvas.draw()


# --- Standalone Execution Guard ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CobberLandApp()  # Use the new class name
    window.show()
    sys.exit(app.exec())