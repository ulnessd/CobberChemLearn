# CobberK.py
# A PyQt6 application for exploring K-Means clustering and K-Nearest Neighbors
# classification using p-block elements.
# Refactored for the CobberLearnChem launcher.

import sys
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
from collections import Counter
import random

# --- Matplotlib and PyQt6 Integration ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QTabWidget, QPushButton, QSpinBox, QFormLayout,
    QMessageBox, QTextEdit, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor # Added for branding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle


# --- Core Engine (LOGIC IS 100% UNCHANGED) ---

@dataclass
class PBlockElement:
    name: str
    symbol: str
    atomic_radius_pm: int
    electronegativity: float
    family: str
    cluster: int = -1
    coords_original: Tuple[float, float] = field(init=False)
    coords_scaled: Tuple[float, float] = field(default=(0.0, 0.0))

    def __post_init__(self): self.coords_original = (self.electronegativity, self.atomic_radius_pm)

P_BLOCK_DATASET: List[PBlockElement] = [
    PBlockElement("Boron", "B", 87, 2.04, "Metalloid"), PBlockElement("Carbon", "C", 67, 2.55, "Carbon Group"),
    PBlockElement("Nitrogen", "N", 56, 3.04, "Pnictogen"), PBlockElement("Oxygen", "O", 48, 3.44, "Chalcogen"),
    PBlockElement("Fluorine", "F", 42, 3.98, "Halogen"),
    PBlockElement("Aluminum", "Al", 118, 1.61, "Boron Group"), PBlockElement("Silicon", "Si", 111, 1.90, "Metalloid"),
    PBlockElement("Phosphorus", "P", 98, 2.19, "Pnictogen"), PBlockElement("Sulfur", "S", 88, 2.58, "Chalcogen"),
    PBlockElement("Chlorine", "Cl", 79, 3.16, "Halogen"),
    PBlockElement("Gallium", "Ga", 136, 1.81, "Boron Group"), PBlockElement("Germanium", "Ge", 125, 2.01, "Metalloid"),
    PBlockElement("Arsenic", "As", 114, 2.18, "Metalloid"), PBlockElement("Selenium", "Se", 103, 2.55, "Chalcogen"),
    PBlockElement("Bromine", "Br", 94, 2.96, "Halogen"),
    PBlockElement("Indium", "In", 156, 1.78, "Boron Group"), PBlockElement("Tin", "Sn", 145, 1.96, "Carbon Group"),
    PBlockElement("Antimony", "Sb", 133, 2.05, "Metalloid"), PBlockElement("Tellurium", "Te", 123, 2.1, "Metalloid"),
    PBlockElement("Iodine", "I", 115, 2.66, "Halogen"),
    PBlockElement("Thallium", "Tl", 196, 1.62, "Boron Group"), PBlockElement("Lead", "Pb", 180, 2.33, "Carbon Group"),
    PBlockElement("Bismuth", "Bi", 160, 2.02, "Pnictogen"), PBlockElement("Polonium", "Po", 190, 2.0, "Metalloid"),
    PBlockElement("Astatine", "At", 127, 2.2, "Metalloid"),
]

def normalize_dataset(dataset: List[PBlockElement]):
    min_x = min(el.coords_original[0] for el in dataset); max_x = max(el.coords_original[0] for el in dataset)
    min_y = min(el.coords_original[1] for el in dataset); max_y = max(el.coords_original[1] for el in dataset)
    for el in dataset:
        scaled_x = (el.coords_original[0] - min_x) / (max_x - min_x) if (max_x - min_x) != 0 else 0
        scaled_y = (el.coords_original[1] - min_y) / (max_y - min_y) if (max_y - min_y) != 0 else 0
        el.coords_scaled = (scaled_x, scaled_y)

normalize_dataset(P_BLOCK_DATASET)

def calculate_distance(p1, p2): return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
def assign_to_clusters(elements, centroids):
    for el in elements:
        if not centroids: el.cluster = -1; continue
        el.cluster = min(centroids.keys(), key=lambda cid: calculate_distance(el.coords_scaled, centroids[cid]))
def update_centroids(elements, k):
    clusters = {i: [] for i in range(k)}
    for el in elements:
        if el.cluster != -1: clusters[el.cluster].append(el.coords_scaled)
    return {i: (np.mean([p[0] for p in points]), np.mean([p[1] for p in points])) for i, points in clusters.items() if points}
def calculate_inertia(elements, centroids):
    return sum(calculate_distance(el.coords_scaled, centroids[el.cluster]) ** 2 for el in elements if el.cluster != -1 and el.cluster in centroids)
def find_k_neighbors(training_data, unknown, k):
    distances = sorted([(calculate_distance(unknown.coords_scaled, el.coords_scaled), el) for el in training_data])
    return [el for dist, el in distances[:k]]
def predict_class(neighbors):
    return Counter(n.family for n in neighbors).most_common(1)[0][0]

# --- GUI Components (LOGIC IS 100% UNCHANGED) ---

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class KMeansTab(QWidget):
    def __init__(self, dataset, main_window):
        super().__init__()
        self.main_window = main_window # Store reference for branding colors
        self.dataset = [el for el in dataset]; self.centroids = {};
        self.cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        layout = QHBoxLayout(self); controls_layout = QVBoxLayout(); form_layout = QFormLayout()
        self.k_spinner = QSpinBox(); self.k_spinner.setRange(2, 8); self.k_spinner.setValue(5)
        form_layout.addRow("Number of Clusters (k):", self.k_spinner)
        self.init_button = QPushButton("Initialize Centroids")
        self.run_step_button = QPushButton("Update Centroids (1 Step)")
        self.run_full_button = QPushButton("Run Full Algorithm")
        self.inertia_label = QLabel("<b>Total Inertia:</b> N/A"); self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        # --- BRANDING ---
        #self.init_button.setStyleSheet(f" font-weight: bold; padding: 5px; border-radius: 3px;")
        #self.run_full_button.setStyleSheet(f" font-weight: bold; padding: 5px; border-radius: 3px;")

        controls_layout.addLayout(form_layout); controls_layout.addWidget(self.init_button); controls_layout.addWidget(self.run_step_button)
        controls_layout.addWidget(self.run_full_button); controls_layout.addWidget(self.inertia_label); controls_layout.addWidget(QLabel("<b>Log:</b>")); controls_layout.addWidget(self.log_text); controls_layout.addStretch()
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100); self.redraw_plot()
        layout.addLayout(controls_layout, 1); layout.addWidget(self.canvas, 4)
        self.init_button.clicked.connect(self.initialize_centroids); self.run_step_button.clicked.connect(self.run_one_step)
        self.run_full_button.clicked.connect(self.run_full_algorithm); self.k_spinner.valueChanged.connect(lambda: self.log_text.append(f"\n--- New Test for k={self.k_spinner.value()} ---"))
        self.run_step_button.setEnabled(False)

    def initialize_centroids(self):
        for el in self.dataset: el.cluster = -1
        k = self.k_spinner.value(); initial_points = random.sample(self.dataset, k)
        self.centroids = {i: pt.coords_scaled for i, pt in enumerate(initial_points)}
        self.run_step_button.setEnabled(True); self.update_clusters_and_inertia(assign_only=True)
    def run_one_step(self):
        new_centroids = update_centroids(self.dataset, self.k_spinner.value())
        if all(i in self.centroids and i in new_centroids and calculate_distance(self.centroids[i], new_centroids[i]) < 1e-6 for i in new_centroids):
            QMessageBox.information(self, "Converged!", "The centroids did not move significantly.")
        self.centroids.update(new_centroids); self.update_clusters_and_inertia()
    def run_full_algorithm(self):
        self.initialize_centroids()
        for i in range(30):
            old_centroids = self.centroids.copy(); assign_to_clusters(self.dataset, self.centroids); self.centroids.update(update_centroids(self.dataset, self.k_spinner.value()))
            if all(i in old_centroids and i in self.centroids and calculate_distance(old_centroids[i], self.centroids[i]) < 1e-7 for i in self.centroids): break
        inertia = calculate_inertia(self.dataset, self.centroids)
        self.log_text.append(f"Converged for k={self.k_spinner.value()} with final inertia: {inertia:.4f}")
        self.inertia_label.setText(f"<b>Total Inertia:</b> {inertia:.4f}"); self.redraw_plot()
    def update_clusters_and_inertia(self, assign_only=False):
        if not assign_only: self.centroids.update(update_centroids(self.dataset, self.k_spinner.value()))
        assign_to_clusters(self.dataset, self.centroids); inertia = calculate_inertia(self.dataset, self.centroids)
        self.inertia_label.setText(f"<b>Total Inertia:</b> {inertia:.4f}"); self.redraw_plot()
    def redraw_plot(self):
        self.canvas.axes.clear()
        for element in self.dataset:
            x, y = element.coords_scaled
            color = self.cluster_colors[element.cluster % len(self.cluster_colors)] if element.cluster != -1 else 'grey'
            self.canvas.axes.scatter(x, y, c=color, alpha=0.8, zorder=2)
            self.canvas.axes.text(x + 0.01, y, element.symbol, fontsize=9, zorder=3)
        for i, coords in self.centroids.items():
            self.canvas.axes.scatter(coords[0], coords[1], c=self.cluster_colors[i % len(self.cluster_colors)], marker='X', s=250, edgecolor='black', zorder=5)
        self.canvas.axes.set_title("p-Block Elements (Normalized Coordinates)"); self.canvas.axes.set_xlabel("Normalized Electronegativity"); self.canvas.axes.set_ylabel("Normalized Atomic Radius")
        self.canvas.axes.set_xlim(-0.1, 1.1); self.canvas.axes.set_ylim(-0.1, 1.1)
        self.canvas.axes.grid(True, linestyle='--', alpha=0.6); self.canvas.fig.tight_layout(); self.canvas.draw()

class KNNTab(QWidget):
    def __init__(self, dataset, main_window):
        super().__init__()
        self.main_window = main_window # Store reference for branding colors
        self.dataset = dataset; self.unknown_element = None
        self.family_colors = {"Halogen": "#1f77b4", "Metalloid": "#2ca02c", "Chalcogen": "#d62728", "Pnictogen": "#9467bd", "Carbon Group": "#8c564b", "Boron Group": "#e377c2"}
        layout = QHBoxLayout(self); controls_layout = QVBoxLayout(); form_layout = QFormLayout()
        self.k_spinner = QSpinBox(); self.k_spinner.setRange(1, 15); self.k_spinner.setValue(5)
        self.unknown_selector = QComboBox(); self.prediction_text = QTextEdit(); self.prediction_text.setReadOnly(True)
        self.test_all_button = QPushButton("Run Full Test")
        # --- BRANDING ---
        #self.test_all_button.setStyleSheet(f" font-weight: bold; padding: 5px; border-radius: 3px;")


        self.populate_selector(); form_layout.addRow("Select Unknown Element:", self.unknown_selector); form_layout.addRow("Number of Neighbors (k):", self.k_spinner)
        controls_layout.addLayout(form_layout); controls_layout.addWidget(QLabel("<b>Prediction Details:</b>")); controls_layout.addWidget(self.prediction_text)
        controls_layout.addWidget(self.test_all_button); controls_layout.addStretch()
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100); self.redraw_plot()
        layout.addLayout(controls_layout, 1); layout.addWidget(self.canvas, 4)
        self.unknown_selector.currentTextChanged.connect(self.set_unknown); self.k_spinner.valueChanged.connect(self.update_view); self.test_all_button.clicked.connect(self.run_full_test)
    def populate_selector(self):
        self.unknown_selector.blockSignals(True); self.unknown_selector.addItem("Select an Element...")
        for el in sorted(self.dataset, key=lambda x: x.name): self.unknown_selector.addItem(el.name)
        self.unknown_selector.blockSignals(False)
    def set_unknown(self, name):
        if name == "Select an Element...": self.unknown_element = None
        else: self.unknown_element = next((el for el in self.dataset if el.name == name), None)
        self.update_view()
    def update_view(self):
        if not self.unknown_element: self.redraw_plot(); self.prediction_text.setText(""); return
        training_data = [el for el in self.dataset if el.name != self.unknown_element.name]; k = self.k_spinner.value()
        neighbors = find_k_neighbors(training_data, self.unknown_element, k); prediction = predict_class(neighbors)
        vote_counts = Counter(n.family for n in neighbors)
        details = f"<b>Prediction for {self.unknown_element.name}:</b> {prediction}\n\n<b>Neighbor Votes:</b>\n"
        for family, count in vote_counts.most_common(): details += f"- {family}: {count} vote(s)\n"
        self.prediction_text.setHtml(details); self.redraw_plot(neighbors)
    def run_full_test(self):
        correct_predictions = 0; total_predictions = len(self.dataset); k = self.k_spinner.value()
        for element_to_test in self.dataset:
            training_data = [el for el in self.dataset if el.name != element_to_test.name]
            neighbors = find_k_neighbors(training_data, element_to_test, k); prediction = predict_class(neighbors)
            if prediction == element_to_test.family: correct_predictions += 1
        success_rate = (correct_predictions / total_predictions) * 100
        current_text = self.prediction_text.toHtml()
        current_text += f"<hr><b>Full Test Results (k={k}):</b><br>Correctly classified {correct_predictions} out of {total_predictions}.<br><b>Success Rate: {success_rate:.1f}%</b>"
        self.prediction_text.setHtml(current_text)
    def redraw_plot(self, neighbors=None):
        self.canvas.axes.clear()
        training_data = [el for el in self.dataset if not self.unknown_element or el.name != self.unknown_element.name]
        handles, labels = self.canvas.axes.get_legend_handles_labels()
        for el in training_data:
            x, y = el.coords_scaled; color = self.family_colors.get(el.family, 'grey')
            label = el.family if el.family not in labels else ""
            if label: labels.append(label)
            self.canvas.axes.scatter(x, y, c=color, label=label, alpha=0.8)
            self.canvas.axes.text(x + 0.01, y, el.symbol, fontsize=9)
        if self.unknown_element:
            ux, uy = self.unknown_element.coords_scaled
            self.canvas.axes.scatter(ux, uy, c='black', marker='X', s=200, zorder=5, label='Unknown')
            if neighbors:
                for n in neighbors: self.canvas.axes.scatter(n.coords_scaled[0], n.coords_scaled[1], s=150, facecolors='none', edgecolors='gold', linewidth=2, zorder=4)
                radius = calculate_distance(self.unknown_element.coords_scaled, neighbors[-1].coords_scaled)
                circle = Circle(self.unknown_element.coords_scaled, radius, facecolor='none', edgecolor='black', linestyle='--', zorder=3); self.canvas.axes.add_patch(circle)
        self.canvas.axes.set_title("p-Block Elements (Normalized Coordinates)"); self.canvas.axes.set_xlabel("Normalized Electronegativity"); self.canvas.axes.set_ylabel("Normalized Atomic Radius")
        self.canvas.axes.set_xlim(-0.1, 1.1); self.canvas.axes.set_ylim(-0.1, 1.1)
        self.canvas.axes.grid(True, linestyle='--', alpha=0.6); self.canvas.axes.legend(); self.canvas.fig.tight_layout(); self.canvas.draw()


# --- Main Application Class (RENAMED and BRANDED) ---
class CobberKApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # --- BRANDING ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberK")
        self.setGeometry(100, 100, 1000, 700)
        self.setFont(self.lato_font)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        # Pass self to the tab widgets so they can access branding colors
        tabs.addTab(KMeansTab(P_BLOCK_DATASET, self), "K-Means Clustering")
        tabs.addTab(KNNTab(P_BLOCK_DATASET, self), "K-Nearest Neighbors (KNN)")


# --- Standalone Execution Guard ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberKApp()
    window.show()
    sys.exit(app.exec())