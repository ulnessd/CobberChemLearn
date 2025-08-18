# CobberNeuron.py
# A PyQt6 application for exploring single neurons and full neural networks.
# Refactored for the CobberLearnChem launcher.

import sys
import numpy as np
from typing import List, Tuple

# --- Required Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QTabWidget, QPushButton, QSlider, QFormLayout,
    QSpinBox, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor  # Added for branding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- Core Engine (LOGIC UNCHANGED) ---

VAPOR_PRESSURE_DATA = {
    0: 0.61, 10: 1.23, 20: 2.34, 30: 4.25, 40: 7.38,
    50: 12.35, 60: 19.94, 70: 31.19, 80: 47.36, 90: 70.14, 100: 101.33
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def calculate_neuron_output(x_input: np.ndarray, weight: float, bias: float) -> np.ndarray:
    return sigmoid(weight * x_input + bias)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def draw_mid_arrow(ax, x0, y0, x1, y1, frac=0.18, color='grey', lw=1, alpha=0.8, head=12):
    """
    Draws a line with a small arrow centered on the segment.
    - frac: fraction of the full segment length to use for the arrow body
    - head: arrow head size (mutation_scale)
    """
    # base line (optional; comment out if you only want the arrow)
    ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=lw)

    # direction vector
    dx, dy = (x1 - x0), (y1 - y0)
    L = (dx**2 + dy**2) ** 0.5
    if L == 0:
        return
    ux, uy = dx / L, dy / L

    # short arrow centered at midpoint
    midx, midy = (x0 + x1) / 2, (y0 + y1) / 2
    half = (frac * L) / 2
    xa, ya = midx - ux * half, midy - uy * half
    xb, yb = midx + ux * half, midy + uy * half

    arrow = FancyArrowPatch(
        (xa, ya), (xb, yb),
        arrowstyle='-|>',
        mutation_scale=head,
        linewidth=lw-1, color=color, alpha=alpha
    )
    ax.add_patch(arrow)

def draw_mid_arrow_on_line(ax, x0, x1, y, frac=0.18, color='grey', lw=1, alpha=0.8, head=12):
    """
    Draw a short centered arrow on a perfectly horizontal line at y.
    - frac: fraction of the (x1-x0) used for the arrow body.
    """
    # The underlying line (optional—comment out if you already draw it elsewhere)
    ax.plot([x0, x1], [y, y], color=color, alpha=alpha, linewidth=lw)

    mid = 0.5 * (x0 + x1)
    half = 0.5 * frac * (x1 - x0)
    xa, xb = mid - half, mid + half

    arrow = FancyArrowPatch(
        (xa, y), (xb, y),
        arrowstyle='-|>',
        mutation_scale=head,
        linewidth=lw,
        color=color,
        alpha=alpha
    )
    ax.add_patch(arrow)

def _ynudge(y, ax, frac=0.012):
    """Move y down by a fraction of the data height."""
    y0, y1 = ax.get_ylim()
    return y - frac * (y1 - y0)

def y_nudge_pixels(ax, y, pixels):
    """Return a y value nudged by a given number of screen pixels."""
    # current (x,y) -> display coords
    _, y_disp = ax.transData.transform((0, y))
    # add pixel offset, then go back to data coords
    y2 = ax.transData.inverted().transform((0, y_disp + pixels))[1]
    return y2


# --- GUI Development (LOGIC UNCHANGED) ---

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Turn on constrained_layout so titles/labels don’t get clipped
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        # Do NOT call self.fig.tight_layout() when using constrained_layout



class SingleNeuronTab(QWidget):
    def __init__(self, temps_c, pressures_kpa):
        super().__init__()
        self.temps_c = temps_c;
        self.pressures_kpa = pressures_kpa
        self.scaled_temps, self.temp_min, self.temp_max = self.normalize(self.temps_c)
        self.scaled_pressures, self.pressure_min, self.pressure_max = self.normalize(self.pressures_kpa)
        layout = QHBoxLayout(self)
        controls_panel = QFrame();
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel);
        controls_layout = QVBoxLayout(controls_panel)
        schematic_label = QLabel("<h3>The Single Neuron Model</h3>")
        self.schematic_canvas = MplCanvas(self, width=3, height=2, dpi=70)
        schematic_text = QLabel("<p>Your goal is to adjust the Weight and Bias to find the lowest MSE.</p>");
        schematic_text.setWordWrap(True)
        form_layout = QFormLayout()
        self.weight_slider = QSlider(Qt.Orientation.Horizontal);
        self.bias_slider = QSlider(Qt.Orientation.Horizontal)
        self.mse_label = QLabel("<b>MSE Score:</b> N/A");
        self.mse_label.setStyleSheet("font-size: 16px;")
        self.weight_slider.setRange(-100, 100);
        self.weight_slider.setValue(50)
        self.bias_slider.setRange(-50, 50);
        self.bias_slider.setValue(-25)
        form_layout.addRow("Weight (w):", self.weight_slider);
        form_layout.addRow("Bias (b):", self.bias_slider)
        controls_layout.addWidget(schematic_label);
        controls_layout.addWidget(self.schematic_canvas);
        controls_layout.addWidget(schematic_text)
        controls_layout.addStretch(1);
        controls_layout.addLayout(form_layout);
        controls_layout.addWidget(self.mse_label);
        controls_layout.addStretch(2)
        plot_panel = QFrame();
        plot_panel.setFrameShape(QFrame.Shape.StyledPanel);
        plot_layout = QVBoxLayout(plot_panel)
        self.canvas = MplCanvas(self, width=7, height=6, dpi=100);
        plot_layout.addWidget(self.canvas)
        layout.addWidget(controls_panel, 1);
        layout.addWidget(plot_panel, 2)
        self.weight_slider.valueChanged.connect(self.update_prediction);
        self.bias_slider.valueChanged.connect(self.update_prediction)
        self.update_prediction()

    def normalize(self, data):
        min_val, max_val = np.min(data), np.max(data)
        if max_val == min_val: return data, min_val, max_val
        return (data - min_val) / (max_val - min_val), min_val, max_val

    def denormalize(self, scaled_data, min_val, max_val):
        return scaled_data * (max_val - min_val) + min_val

    def update_prediction(self):
        weight = self.weight_slider.value() / 10.0
        bias = self.bias_slider.value() / 10.0

        self.draw_schematic(weight, bias)

        # Predict (in scaled space), compute MSE (scaled), then denormalize
        predicted_scaled = calculate_neuron_output(self.scaled_temps, weight, bias)
        mse = calculate_mse(self.scaled_pressures, predicted_scaled)
        self.mse_label.setText(f"<b>MSE Score:</b> {mse:.4f}")

        predicted_denormalized = self.denormalize(predicted_scaled, self.pressure_min, self.pressure_max)

        # --- Main figure: data + sigmoid fit ---
        ax = self.canvas.axes
        ax.clear()
        ax.scatter(self.temps_c, self.pressures_kpa, label='Experimental Data')
        ax.plot(self.temps_c, predicted_denormalized, 'r--',
                label=f'Neuron Prediction (w={weight:.1f}, b={bias:.1f})')

        ax.set_title("Vapor Pressure of Water vs. Temperature")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Vapor Pressure (kPa)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')  # <- move legend

        # --- Inset: residuals (Observed − Predicted) ---
        residuals = self.pressures_kpa - predicted_denormalized
        inset_ax = ax.inset_axes([0.07, 0.55, 0.38, 0.38])  # [x0, y0, width, height] in axes fraction
        inset_ax.axhline(0.0, linestyle='--', alpha=0.7)
        inset_ax.vlines(self.temps_c, 0.0, residuals, alpha=0.6)
        inset_ax.scatter(self.temps_c, residuals, s=18)

        inset_ax.set_title("Residuals", fontsize=9)
        inset_ax.set_xlabel("Temp (°C)", fontsize=8)
        inset_ax.set_ylabel("kPa", fontsize=8)
        inset_ax.tick_params(labelsize=8)
        inset_ax.grid(True, linestyle=':', alpha=0.4)

        self.canvas.draw()

    def draw_schematic(self, weight, bias):
        ax = self.schematic_canvas.axes;
        ax.clear()
        input_y, hidden_y, output_y = 0.5, 0.5, 0.5
        #ax.plot([0.1, 0.5], [input_y, hidden_y], 'grey', alpha=0.8);
        #ax.plot([0.5, 0.9], [hidden_y, output_y], 'grey', alpha=0.8)
       # ax.plot([0, 0.1], [hidden_y, output_y], 'grey', alpha=0.8)
        #ax.plot([0.9, 1], [hidden_y, output_y], 'grey', alpha=0.8)
        draw_mid_arrow(ax, 0.94, input_y, 1.02, hidden_y, color='grey', lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, -0.03, input_y, 0.07, hidden_y, color='grey', lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, 0.1, input_y, 0.5, hidden_y, color='grey', lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, 0.5, hidden_y, 0.9, output_y, color='grey', lw=2, alpha=0.8, head=14)

        ax.text(0.37, 0.62, f'w={weight:.1f}', ha='center', va='center', fontsize=16,
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))
        ax.text(0.53, 0.25, f'b={bias:.1f}', ha='center', va='center', fontsize=16,
                bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))
        ax.scatter([0.1], [input_y], s=1000, c='#d3d3d3', marker='s', zorder=5);
        ax.text(-0.05, output_y, "T", ha='center', va='center', fontsize=16)
        ax.text(0.1, input_y - 0.4, "Input\nNeuron", ha='center', va='center', fontsize=16)
        ax.scatter([0.5], [hidden_y], s=1000, c='#6C1D45', marker='s', zorder=5)
        ax.scatter([0.9], [output_y], s=1000, c='#3D3D3D', marker='s', zorder=5);
        ax.text(0.9, output_y - 0.4, "Output\nNeuron", ha='center', va='center', fontsize=16)
        ax.text(1.07, output_y, r"P$_{\text{vap}}$", ha='center', va='center', fontsize=16)
        ax.set_xlim(-0.1, 1.1);
        ax.set_ylim(0, 1);
        ax.axis('off');
        self.schematic_canvas.draw()


class NeuralNetworkTab(QWidget):
    def __init__(self, temps_c, pressures_kpa, main_window):
        super().__init__()
        self.main_window = main_window
        self.temps_c = temps_c
        self.pressures_kpa = pressures_kpa
        self.model = None
        self.scaler_X = None
        self.scaler_y = None

        layout = QHBoxLayout(self)

        # --- Controls (left) ---
        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)

        form_layout = QFormLayout()
        self.neurons_spinner = QSpinBox()
        self.neurons_spinner.setRange(2, 16)
        self.neurons_spinner.setValue(8)
        self.learning_rate_input = QLineEdit("0.005")
        self.epochs_input = QLineEdit("10000")
        self.train_button = QPushButton("Train Network")

        form_layout.addRow("Neurons in Hidden Layer:", self.neurons_spinner)
        form_layout.addRow("Learning Rate:", self.learning_rate_input)
        form_layout.addRow("Training Cycles (Epochs):", self.epochs_input)
        controls_layout.addWidget(QLabel("<h3>Network Configuration</h3>"))
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.train_button)
        controls_layout.addStretch()

        # --- Schematic (middle) ---
        schematic_panel = QFrame()
        schematic_panel.setFrameShape(QFrame.Shape.StyledPanel)
        schematic_layout = QVBoxLayout(schematic_panel)
        schematic_layout.addWidget(QLabel("<b>Network Architecture</b>"))
        self.schematic_canvas = MplCanvas(self, width=7, height=7)
        schematic_layout.addWidget(self.schematic_canvas)

        # --- Results (right) ---
        results_panel = QTabWidget()

        # Tab 1: Training Progress
        self.loss_canvas = MplCanvas(self)
        results_panel.addTab(self.loss_canvas, "Training Progress")

        # Tab 2: Final Model Fit
        self.fit_canvas = MplCanvas(self)
        results_panel.addTab(self.fit_canvas, "Final Model Fit")

        # Tab 3: Residuals
        self.resid_tab = QWidget()
        resid_layout = QVBoxLayout(self.resid_tab)
        self.resid_canvas = MplCanvas(self)

        self.mae_label = QLabel("MAE: -")
        self.mse_label = QLabel("MSE: -")

        mae_font = QFont(self.font()); mae_font.setPointSize(16)
        self.mae_label.setFont(mae_font)
        mse_font = QFont(self.font()); mse_font.setPointSize(16)
        self.mse_label.setFont(mse_font)

        resid_layout.setSpacing(6)
        resid_layout.setContentsMargins(6, 6, 6, 6)
        labels_layout = QHBoxLayout()
        labels_layout.setSpacing(12)
        labels_layout.addStretch(1)
        labels_layout.addWidget(self.mae_label)
        labels_layout.addWidget(self.mse_label)
        labels_layout.addStretch(1)
        resid_layout.addWidget(self.resid_canvas)
        resid_layout.addLayout(labels_layout)

        results_panel.addTab(self.resid_tab, "Residuals")        # ✅ add the tab

        layout.addWidget(controls_panel, 3)                      # ✅ add to main layout
        layout.addWidget(schematic_panel, 7)                     # ✅ add to main layout
        layout.addWidget(results_panel, 7)                       # ✅ add to main layout

        self.train_button.clicked.connect(self.train_network)    # ✅ connect signals
        self.neurons_spinner.valueChanged.connect(self.update_schematic)

        self.setup_initial_plots()                                # ✅ draw initial content
        self.update_schematic()

    def train_network(self):
        try:
            hidden_neurons = self.neurons_spinner.value()
            learning_rate = float(self.learning_rate_input.text())
            epochs = int(self.epochs_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please ensure all parameters are valid numbers.")
            return

        X = self.temps_c.reshape(-1, 1)
        y = self.pressures_kpa

        self.train_button.setText("Training...")
        self.train_button.setEnabled(False)
        QApplication.processEvents()

        scaler_X = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))
        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons,),
            activation='relu',
            solver='adam',
            learning_rate_init=learning_rate,
            max_iter=epochs,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=500
        )
        self.model.fit(X_scaled, y_scaled)

        self.train_button.setText("Train Network")
        self.train_button.setEnabled(True)

        self.update_schematic()

        # --- Tab 1: Training Progress ---
        self.loss_canvas.axes.clear()
        self.loss_canvas.axes.plot(self.model.loss_curve_)
        self.loss_canvas.axes.set_title("Loss vs. Epoch")
        self.loss_canvas.axes.set_xlabel("Epoch")
        self.loss_canvas.axes.set_ylabel("Mean Squared Error (Loss)")
        self.loss_canvas.axes.grid(True)
        self.loss_canvas.draw()

        # --- Tab 2: Final Model Fit ---
        x_smooth = np.linspace(self.temps_c.min(), self.temps_c.max(), 200).reshape(-1, 1)
        x_smooth_scaled = scaler_X.transform(x_smooth)
        y_pred_scaled = self.model.predict(x_smooth_scaled)
        y_pred_denormalized = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

        self.fit_canvas.axes.clear()
        self.fit_canvas.axes.scatter(self.temps_c, self.pressures_kpa, label='Experimental Data')
        self.fit_canvas.axes.plot(x_smooth, y_pred_denormalized, 'r-', label='Neural Network Fit', linewidth=2)
        self.fit_canvas.axes.set_title("Vapor Pressure Data & Model Fit")
        self.fit_canvas.axes.set_xlabel("Temperature (°C)")
        self.fit_canvas.axes.set_ylabel("Vapor Pressure (kPa)")
        self.fit_canvas.axes.legend()
        self.fit_canvas.axes.grid(True)
        self.fit_canvas.draw()

        # --- Tab 3: Residuals (new) ---
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        # Predictions on original training X (so we can compute residuals)
        y_train_pred_scaled = self.model.predict(X_scaled)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

        residuals = self.pressures_kpa - y_train_pred  # Observed − Predicted
        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(residuals ** 2))

        axr = self.resid_canvas.axes
        axr.clear()
        axr.axhline(0.0, linestyle='--', alpha=0.7)

        # “Drop lines” from each point to zero line + markers at the tips
        axr.vlines(self.temps_c, 0.0, residuals, alpha=0.6)
        axr.scatter(self.temps_c, residuals, s=25)

        axr.set_title("Residuals (Observed − Predicted)")
        axr.set_xlabel("Temperature (°C)")
        axr.set_ylabel("Residual (kPa)")
        axr.grid(True, linestyle=':', alpha=0.5)
        self.resid_canvas.draw()

        self.mae_label.setText(f"MAE: {mae:.4f} kPa")
        self.mse_label.setText(f"MSE: {mse:.4f} (kPa²)")

    def update_schematic(self):
        ax = self.schematic_canvas.axes
        ax.clear()
        num_neurons = self.neurons_spinner.value()
        scale_font_size = 20 - num_neurons

        input_y = [0.5]
        hidden_y = np.linspace(0.1, 0.9, num_neurons)
        output_y = [0.5]

        if self.model and self.model.hidden_layer_sizes[0] == num_neurons:
            input_weights = self.model.coefs_[0]
            hidden_biases = self.model.intercepts_[0]
            output_weights = self.model.coefs_[1]

            for i, hy in enumerate(hidden_y):
                #ax.plot([0.1, 0.5], [input_y[0], hy], 'grey', alpha=0.5)
                draw_mid_arrow(ax, 0.1, input_y[0], 0.5, hy, color='grey', lw=2, alpha=0.8, head=14)
                ax.text(
                    0.3, (input_y[0] + hy) / 2,
                    f'w={input_weights[0][i]:.1f}',
                    ha='center', va='center', fontsize=scale_font_size,
                    bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                )
                ax.text(
                    0.5, (input_y[0]+ hy)-.545,
                    f'b = {hidden_biases[i]: .1f}',
                    ha='center', va='center', fontsize=scale_font_size,
                    bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                )

            for i, hy in enumerate(hidden_y):
                #ax.plot([0.5, 0.9], [hy, output_y[0]], 'grey', alpha=0.5)
                draw_mid_arrow(ax, 0.5, hy, 0.9, output_y[0], color='grey', lw=2, alpha=0.8, head=14)
                ax.text(
                    0.7, (hy + output_y[0]) / 2,
                    f'w={output_weights[i][0]:.1f}',
                    ha='center', va='center', fontsize=scale_font_size,
                    bbox=dict(facecolor='white', alpha=0.0, edgecolor='none')
                )
        else:
            for hy in hidden_y:
                #ax.plot([0.1, 0.5], [input_y[0], hy], 'grey')
                #ax.plot([0.5, 0.9], [hy, output_y[0]], 'grey')
                draw_mid_arrow(ax, 0.1, input_y[0], 0.5, hy, color='grey', lw=1.8, alpha=0.8, head=12)
                draw_mid_arrow(ax, 0.5, hy, 0.9, output_y[0], color='grey', lw=1.8, alpha=0.8, head=12)

        yin = y_nudge_pixels(ax, input_y[0], +1)  # tweak frac as needed (0.008–0.02)
        yout = y_nudge_pixels(ax, output_y[0], +1)

        # Input arrow (from left into the input node)
        draw_mid_arrow_on_line(ax, -0.02, 0.08, yin, color='grey', lw=2, alpha=0.8, head=14)

        # Output arrow (from output node to the right)
        draw_mid_arrow_on_line(ax, 0.95, 1.02, yout, color='grey', lw=2, alpha=0.8, head=14)
        ax.scatter([0.1], input_y, s=1000, c='#d3d3d3', marker='s', zorder=5)
        ax.text(0.075, input_y[0] - 0.09, "Input\nNeuron", ha='center', va='center', fontsize=13)
        ax.scatter([0.5] * num_neurons, hidden_y, s=400, c='#6C1D45', marker='s', zorder=5)
        ax.text(0.5, input_y[0] + 0.475, "Hidden Layer", ha='center', va='center', fontsize=16)
        ax.scatter([0.9], output_y, s=1000, c='#3D3D3D', marker='s', zorder=5)
        ax.text(0.925, output_y[0] - 0.09, "Output\nNeuron", ha='center', va='center', fontsize=13)
        ax.text(-.05, output_y[0], "T", ha='center', va='center', fontsize=16)
        ax.text(1.075, output_y[0], r"P$_{\text{vap}}$", ha='center', va='center', fontsize=16)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        self.schematic_canvas.draw()

    def setup_initial_plots(self):
        # Training progress
        self.loss_canvas.axes.set_title("Loss vs. Epoch")
        self.loss_canvas.axes.set_xlabel("Epoch")
        self.loss_canvas.axes.set_ylabel("Mean Squared Error (Loss)")
        self.loss_canvas.axes.grid(True)
        self.loss_canvas.draw()

        # Final model fit (initial: just the scatter)
        self.fit_canvas.axes.clear()
        self.fit_canvas.axes.scatter(self.temps_c, self.pressures_kpa, label='Experimental Data')
        self.fit_canvas.axes.set_title("Vapor Pressure Data & Model Fit")
        self.fit_canvas.axes.set_xlabel("Temperature (°C)")
        self.fit_canvas.axes.set_ylabel("Vapor Pressure (kPa)")
        self.fit_canvas.axes.legend()
        self.fit_canvas.axes.grid(True)
        self.fit_canvas.draw()

        # Residuals placeholder
        self.resid_canvas.axes.clear()
        self.resid_canvas.axes.axhline(0.0, linestyle='--', alpha=0.7)
        self.resid_canvas.axes.set_title("Residuals (Observed − Predicted)")
        self.resid_canvas.axes.set_xlabel("Temperature (°C)")
        self.resid_canvas.axes.set_ylabel("Residual (kPa)")
        self.resid_canvas.axes.text(
            0.5, 0.5, "Train the network to see residuals.",
            ha='center', va='center', transform=self.resid_canvas.axes.transAxes, alpha=0.7
        )
        self.resid_canvas.draw()



# --- RENAMED Main Application Class ---
class CobberNeuronApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # --- BRANDING ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberNeuron")
        self.setGeometry(100, 100, 1400, 700)
        self.setFont(self.lato_font)

        self.temps_c = np.array(list(VAPOR_PRESSURE_DATA.keys()))
        self.pressures_kpa = np.array(list(VAPOR_PRESSURE_DATA.values()))

        tabs = QTabWidget();
        self.setCentralWidget(tabs)
        tabs.addTab(SingleNeuronTab(self.temps_c, self.pressures_kpa), "The Single Neuron")
        # Pass self to the tab so it can access branding colors
        tabs.addTab(NeuralNetworkTab(self.temps_c, self.pressures_kpa, self), "The Neural Network")


# --- Standalone Execution Guard ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberNeuronApp()  # Use new class name
    window.show()
    sys.exit(app.exec())