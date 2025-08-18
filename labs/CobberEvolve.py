# CobberEvolve.py
# An application for exploring evolutionary algorithms by evolving a ligand
# to match a protein binding site.
# Refactored for the CobberLearnChem launcher.

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QMessageBox, QFrame
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QPainter, QFont


# --- Custom Widgets (Functionality Unchanged) ---

class ProteinLigandDisplay(QWidget):
    def __init__(self, protein_values, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.protein_values = protein_values
        self.ligand_values = None
        self.energy = None
        self.energy_color = QColor("black")
        self.colormap = plt.get_cmap('rainbow')

    def get_color(self, value):
        rgba = self.colormap(value)
        return QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

    def update_ligand(self, ligand_values, energy, color_str):
        self.ligand_values = ligand_values
        self.energy = energy
        self.energy_color = QColor(color_str)
        self.update()

    def clear_ligand(self):
        self.ligand_values = None
        self.energy = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        rect_height = 25
        num_segments = len(self.protein_values)
        segment_width = width / num_segments
        y_protein = height * 0.7
        font = painter.font()
        font.setPointSize(16)  # Or whatever size you prefer
        painter.setFont(font)
        painter.drawText(int(width / 2 - 50), int(y_protein - 10), "Protein Binding Site")
        for i, val in enumerate(self.protein_values):
            color = self.get_color(val)
            painter.fillRect(int(i * segment_width), int(y_protein), int(segment_width + 1), rect_height, color)
        if self.ligand_values is not None:
            y_ligand = height * 0.25
            # Draw label for ligand
            painter.setPen(Qt.GlobalColor.black)
            font = painter.font()
            font.setPointSize(16)  # Or whatever size you prefer
            painter.setFont(font)
            painter.drawText(int(width / 2 - 25), int(y_ligand - 10), "Ligand")
            for i, val in enumerate(self.ligand_values):
                color = self.get_color(val)
                painter.fillRect(int(i * segment_width), int(y_ligand), int(segment_width + 1), rect_height, color)
            if self.energy is not None:
                font = painter.font()
                font.setPointSize(16)
                painter.setFont(font)
                painter.setPen(self.energy_color)
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                                 f"Binding Energy of Ligand Below, E = {self.energy:.2f}")


class TopLigandDisplay(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 60)
        self.ligand_values = None
        self.energy = None
        self.colormap = plt.get_cmap('rainbow')

    def get_color(self, value):
        rgba = self.colormap(value)
        return QColor(int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))

    def set_ligand(self, ligand_values, energy):
        self.ligand_values = ligand_values
        self.energy = energy
        self.update()

    def clear(self):
        self.ligand_values = None
        self.energy = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        if self.ligand_values is None: return
        width, height = self.width(), self.height()
        rect_height = 20
        y0 = (height - rect_height) / 2 - 10
        num_segments = len(self.ligand_values)
        segment_width = width / num_segments
        for i, val in enumerate(self.ligand_values):
            color = self.get_color(val)
            painter.fillRect(int(i * segment_width), int(y0), int(segment_width + 1), rect_height, color)
        font = painter.font()
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                         f"E={self.energy:.2f}")


# --- RENAMED Main Application Class ---
class CobberEvolveApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- BRANDING ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberEvolve")
        self.setFixedSize(1400, 700)
        self.setFont(self.lato_font)

        # Core logic variables (unchanged)
        self.protein_segments = 100
        self.protein_values = np.array(
            [math.cos(2.7 * x / self.protein_segments) ** 2 for x in range(self.protein_segments)])
        self.protein_values = self.protein_values / np.max(self.protein_values)
        self.top_ligands = [];
        self.current_generation = 0
        self.plot_all_generations = [];
        self.plot_all_energies = []
        self.plot_best_generations = [];
        self.plot_best_energies = []
        self.blue_dots = None;
        self.red_line = None
        self.current_top_ligand = None
        self.timer = QTimer(self);
        self.ligands_to_process = 0
        self.is_seed_run = False;
        self.perturb_level = 0.05

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_frame = QFrame();
        left_layout = QVBoxLayout(left_frame)
        self.figure, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas_fig = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas_fig)
        self.protein_ligand_display = ProteinLigandDisplay(self.protein_values)
        left_layout.addWidget(self.protein_ligand_display)

        right_frame = QFrame();
        right_frame.setFixedWidth(300)
        right_layout = QVBoxLayout(right_frame);
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.present_btn = QPushButton("Present Single Ligand");
        self.present_btn.setMinimumHeight(40)
        self.seed_btn = QPushButton("Run Seed Generation");
        self.seed_btn.setMinimumHeight(40)
        self.next_gen_btn = QPushButton("Run Next Generation");
        self.next_gen_btn.setMinimumHeight(40)

        # --- BRANDING ---
        button_style = f"""
            QPushButton {{
                border: 2px solid {self.cobber_maroon.name()};
                font-size: 16px;
                font-weight: 600;
                padding: 10px 14px;
                border-radius: 6px;
                background-color: white;
            }}
            QPushButton:hover {{
                background-color: #FFD700;
            }}
        """
        self.present_btn.setStyleSheet(button_style)
        self.seed_btn.setStyleSheet(button_style)
        self.next_gen_btn.setStyleSheet(button_style)
        #self.present_btn.setStyleSheet("QPushButton { padding: 8px; }")

        self.present_btn.clicked.connect(self.present_single_ligand)
        self.seed_btn.clicked.connect(self.run_seed_generation)
        self.next_gen_btn.clicked.connect(self.run_next_generation)

        right_layout.addWidget(self.present_btn);
        right_layout.addWidget(self.seed_btn);
        right_layout.addWidget(self.next_gen_btn)

        ligands_layout = QHBoxLayout();
        ligands_layout.addWidget(QLabel("Ligands per Gen:"))
        self.ligands_input = QLineEdit("500");
        ligands_layout.addWidget(self.ligands_input)

        perturb_layout = QHBoxLayout();
        perturb_layout.addWidget(QLabel("Perturbation Level:"))
        self.perturb_input = QLineEdit("0.3");
        perturb_layout.addWidget(self.perturb_input)

        right_layout.addLayout(ligands_layout);
        right_layout.addLayout(perturb_layout)

        self.generation_label = QLabel("Generation: 0");
        self.generation_label.setStyleSheet("font-size: 14pt;");
        self.generation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.generation_label)

        top_ligands_label = QLabel("Top 3 Ligands:");
        top_ligands_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(top_ligands_label)

        self.top_ligands_displays = []
        for i in range(3):
            display = TopLigandDisplay();
            display.setFrameShape(QFrame.Shape.Panel);
            display.setFrameShadow(QFrame.Shadow.Sunken)
            right_layout.addWidget(display);
            self.top_ligands_displays.append(display)

        main_layout.addWidget(left_frame, 1);
        main_layout.addWidget(right_frame)

        self.initialize_plot()
        self.timer.timeout.connect(self.process_one_ligand_tick)

    # --- All other methods have UNCHANGED logic and are now indented correctly ---
    def set_buttons_enabled(self, enabled):
        self.present_btn.setEnabled(enabled); self.seed_btn.setEnabled(enabled); self.next_gen_btn.setEnabled(enabled)

    def calculate_energy(self, ligand_values):
        ligand_values = np.clip(ligand_values, 0, 1);
        rms = np.sqrt(np.mean((self.protein_values - ligand_values) ** 2))
        try:
            ratio = (0.5 - rms) / rms; E = -5 * math.log(ratio) if ratio > 0 else 20
        except (ValueError, ZeroDivisionError):
            E = 0
        color_str = 'red' if E > 1 else 'green' if E < -1 else 'blue';
        return E, color_str

    def initialize_plot(self):
        self.ax.cla();
        self.blue_dots, = self.ax.plot([], [], 'b.', markersize=5, alpha=0.4);
        self.red_line, = self.ax.plot([], [], 'ro-', label="Best Energy per Generation")
        self.ax.set_title("Best Match Energy vs Generation");
        self.ax.set_xlabel("Generation");
        self.ax.set_ylabel("Binding Energy")
        self.ax.grid(True);
        self.ax.legend();
        self.figure.tight_layout();
        self.canvas_fig.draw()

    def update_animation_and_data(self, ligand_values):
        energy, color_str = self.calculate_energy(ligand_values);
        new_ligand = (ligand_values, energy)
        self.protein_ligand_display.update_ligand(ligand_values, energy, color_str)
        self.plot_all_generations.append(self.current_generation);
        self.plot_all_energies.append(energy)
        self.blue_dots.set_data(self.plot_all_generations, self.plot_all_energies)
        if len(self.top_ligands) < 3:
            self.top_ligands.append(new_ligand); self.top_ligands.sort(key=lambda x: x[1])
        elif energy < self.top_ligands[-1][1]:
            self.top_ligands[-1] = new_ligand; self.top_ligands.sort(key=lambda x: x[1])
        current_best_energy = self.top_ligands[0][1]
        if not self.plot_best_generations or self.plot_best_generations[-1] != self.current_generation:
            self.plot_best_generations.append(self.current_generation);
            self.plot_best_energies.append(current_best_energy)
        else:
            self.plot_best_energies[-1] = current_best_energy
        self.red_line.set_data(self.plot_best_generations, self.plot_best_energies);
        self.ax.relim();
        self.ax.autoscale_view();
        self.canvas_fig.draw_idle()

    def finalize_generation(self):
        self.set_buttons_enabled(True)
        if self.top_ligands:
            self.current_top_ligand = self.top_ligands[0][0]
            for i, display in enumerate(self.top_ligands_displays):
                if i < len(self.top_ligands): ligand, energy = self.top_ligands[i]; display.set_ligand(ligand, energy)
        self.canvas_fig.draw()

    def present_single_ligand(self):
        if self.timer.isActive(): return
        self.start_generation(is_seed=True, num_ligands=1)

    def process_one_ligand_tick(self):
        if self.ligands_to_process <= 0: self.timer.stop(); self.finalize_generation(); return
        if self.is_seed_run:
            ligand_values = np.random.rand(self.protein_segments)
        else:
            perturbation = np.random.uniform(-self.perturb_level, self.perturb_level, self.protein_segments)
            ligand_values = self.current_top_ligand + perturbation
        self.update_animation_and_data(ligand_values);
        self.ligands_to_process -= 1
        if self.ligands_to_process == 0: self.timer.stop(); self.finalize_generation()

    def start_generation(self, is_seed, num_ligands=None):
        if self.timer.isActive():
            return

        if num_ligands is None:
            try:
                self.ligands_to_process = int(self.ligands_input.text())
                if self.ligands_to_process <= 0:
                    raise ValueError
            except ValueError:
                QMessageBox.critical(self, "Invalid Input", "Ligands per Gen must be a positive integer.")
                return
        else:
            self.ligands_to_process = num_ligands

        try:
            self.perturb_level = float(self.perturb_input.text())
        except ValueError:
            QMessageBox.critical(self, "Invalid Input", "Perturbation Level must be a number.")
            return

        self.is_seed_run = is_seed
        if self.is_seed_run:
            self.current_generation = 1
            self.top_ligands = []
            self.plot_all_generations, self.plot_all_energies = [], []
            self.plot_best_generations, self.plot_best_energies = [], []
            self.initialize_plot()
            for display in self.top_ligands_displays:
                display.clear()
        else:
            if not self.top_ligands:
                QMessageBox.warning(self, "No Ligands", "Run seed generation first.")
                return
            self.current_generation += 1
            self.plot_all_generations, self.plot_all_energies = [], []
            self.blue_dots.set_data([], [])

        self.generation_label.setText(f"Generation: {self.current_generation}")
        self.set_buttons_enabled(False)
        self.timer.start(10)

    # --- CORRECTLY INDENTED METHODS ---
    def run_seed_generation(self):
        self.start_generation(is_seed=True)

    def run_next_generation(self):
        self.start_generation(is_seed=False)


# --- Standalone Execution Guard ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = CobberEvolveApp()
    main_win.show()
    sys.exit(app.exec())