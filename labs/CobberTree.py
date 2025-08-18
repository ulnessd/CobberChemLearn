# cobber_tree.py
# An application for exploring Decision Tree models by manually and
# automatically sorting alkane data.
# Refactored for the CobberLearnChem launcher.

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import py3Dmol

# --- Add project root to path for robust imports ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('.')

# Scikit-learn imports for machine learning
from sklearn.tree import DecisionTreeRegressor, plot_tree

from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QListWidget, QAbstractItemView, QListWidgetItem,
    QTabWidget, QPushButton, QSpinBox, QTextEdit, QFormLayout, QMessageBox
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPixmap

from dataclasses import dataclass
from typing import List
from sklearn.tree import _tree


# --- Data Structures and Helper Functions (LOGIC UNCHANGED) ---
@dataclass
class Hydrocarbon:
    name: str
    formula: str
    num_carbons: int
    num_branches: int
    boiling_point: float
    sdf_path: str


def calculate_weighted_variance(group1: List[Hydrocarbon], group2: List[Hydrocarbon]) -> float:
    n1, n2 = len(group1), len(group2)
    n_total = n1 + n2
    if n_total == 0: return 0.0
    var1 = np.var([hc.boiling_point for hc in group1]) if n1 > 1 else 0.0
    var2 = np.var([hc.boiling_point for hc in group2]) if n2 > 1 else 0.0
    return (n1 / n_total) * var1 + (n2 / n_total) * var2


# --- Custom Widgets for the GUI ---
class ViewerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Molecule Viewer")
        self.setGeometry(200, 200, 500, 500)
        layout = QVBoxLayout(self)
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)
        self.setLayout(layout)

    def update_view(self, sdf_path):
        try:
            with open(sdf_path, 'r') as f:
                sdf_data = f.read()
            view = py3Dmol.view(width=480, height=480)
            view.addModel(sdf_data, 'sdf')
            view.setStyle({'stick': {}})
            view.zoomTo()
            self.web_view.setHtml(view.write_html())
        except FileNotFoundError:
            self.web_view.setHtml(f"<html><body><h2>Error: File not found</h2><p>{sdf_path}</p></body></html>")
        except Exception as e:
            self.web_view.setHtml(f"<html><body><h2>Error rendering molecule</h2><p>{e}</p></body></html>")


class MoleculeListWidget(QListWidget):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setAcceptDrops(True)
        self.setIconSize(QSize(80, 50))

    def dropEvent(self, event):
        source_widget = event.source()
        if not isinstance(source_widget, MoleculeListWidget):
            event.ignore()
            return
        source_name = source_widget.name
        target_name = self.name
        allowed_moves = {
            "deck": ["bin1", "bin2"],
            "bin1": ["deck", "bin1_1", "bin1_2", "bin2"],
            "bin2": ["deck", "bin2_1", "bin2_2", "bin1"],
            "bin1_1": ["bin1", "bin1_2"],
            "bin1_2": ["bin1", "bin1_1"],
            "bin2_1": ["bin2", "bin2_2"],
            "bin2_2": ["bin2", "bin2_1"],
        }
        if target_name in allowed_moves.get(source_name, []):
            item = source_widget.takeItem(source_widget.row(source_widget.currentItem()))
            self.addItem(item)
            event.accept()
        else:
            event.ignore()


class AutomatedTreeWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.tree_image_path = "decision_tree.png"
        self.model = None
        main_layout = QHBoxLayout(self)
        controls_layout = QVBoxLayout()
        display_layout = QVBoxLayout()
        form_layout = QFormLayout()
        self.depth_spinner = QSpinBox()
        self.depth_spinner.setRange(2, 10)
        self.depth_spinner.setValue(3)
        self.generate_button = QPushButton("Generate Decision Tree")
        self.predict_button = QPushButton("Predict Unknowns")
        self.generate_button.setStyleSheet(f"""
            QPushButton {{
                background-color: #F0F0F0; /* Light grey background */
                border: 1px solid #ADADAD; /* A subtle border */
                color: QColor(108, 29, 69);
                font-size: 14px; font-weight: bold; padding: 8px; border-radius: 5px;
            }}
            QPushButton:hover {{ background-color: #E0E0E0; }}
        """)
        self.predict_button.setStyleSheet("""
            QPushButton {
                background-color: #F0F0F0; /* Light grey background */
                border: 1px solid #ADADAD; /* A subtle border */
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #E0E0E0; /* A slightly darker grey on hover */
            }
        """)
        form_layout.addRow("Max Tree Depth:", self.depth_spinner)
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.generate_button)
        controls_layout.addWidget(self.predict_button)
        controls_layout.addStretch()
        self.tree_image_label = QLabel("Click 'Generate' to create and display the decision tree.")
        self.tree_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        display_layout.addWidget(self.tree_image_label, 3)
        display_layout.addWidget(QLabel("<b>Analysis & Predictions:</b>"))
        display_layout.addWidget(self.results_text, 1)
        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(display_layout, 4)
        self.generate_button.clicked.connect(self.generate_tree)
        self.predict_button.clicked.connect(self.predict_unknowns)
        self.predict_button.setEnabled(False)

    def generate_tree(self):
        # --- PATH IMPROVEMENT ---
        csv_path = os.path.join(project_root, "assets", "hydrocarbon_data.csv")
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", f"Data file not found at:\n{csv_path}")
            return

        features = ['Num_Carbons', 'Num_Branches', 'Molar_Mass', 'Density']
        target = 'Boiling_Point'

        X = df[features]
        y = df[target]

        max_depth = self.depth_spinner.value()
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        self.model.fit(X, y)

        plt.figure(figsize=(20, 10))
        plot_tree(self.model, feature_names=features, filled=False, rounded=True, fontsize=10)
        plt.title(f"Decision Tree for Alkane Boiling Points (Max Depth = {max_depth})", fontsize=16)
        plt.savefig(self.tree_image_path, dpi=150, bbox_inches='tight')
        plt.close()

        pixmap = QPixmap(self.tree_image_path)
        self.tree_image_label.setPixmap(pixmap.scaled(self.tree_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                      Qt.TransformationMode.SmoothTransformation))

        importances = self.model.feature_importances_
        feature_importance_text = "\n".join([f"  - {name}: {imp:.2%}" for name, imp in zip(features, importances)])
        results = (f"--- Model Training Results ---\n"
                   f"Dataset size: {len(df)} molecules\n"
                   f"Tree Depth: {max_depth}\n\n"
                   f"Feature Importances:\n{feature_importance_text}")
        self.results_text.setText(results)
        self.predict_button.setEnabled(True)

    def predict_unknowns(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "Please generate a tree before making predictions.")
            return
        unknowns = {
            "2,2,4-Trimethylpentane": pd.DataFrame([[8, 3, 114.23, 0.692]],
                                                   columns=['Num_Carbons', 'Num_Branches', 'Molar_Mass', 'Density']),
            "3-Ethyl-2,2-dimethylpentane": pd.DataFrame([[9, 3, 128.26, 0.729]],
                                                        columns=['Num_Carbons', 'Num_Branches', 'Molar_Mass',
                                                                 'Density']),
        }
        prediction_text = "\n\n--- Predictions for Unknowns ---\n"
        for name, data in unknowns.items():
            prediction = self.model.predict(data)[0]
            prediction_text += f"Predicted Boiling Point for {name}:\n  -> {prediction:.2f} °C\n"
        self.results_text.append(prediction_text)


class CobberTreeApp(QMainWindow):
    def __init__(self, manual_dataset):
        super().__init__()
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")
        self.setWindowTitle("CobberTree")
        self.setGeometry(100, 100, 1200, 800)
        self.setFont(self.lato_font)
        tabs = QTabWidget()
        self.setCentralWidget(tabs)
        manual_sorter_widget = self.create_manual_sorter_tab(manual_dataset)
        tabs.addTab(manual_sorter_widget, "Manual Sorter")
        automated_tree_widget = AutomatedTreeWidget()
        tabs.addTab(automated_tree_widget, "Automated Tree")

    def create_manual_sorter_tab(self, dataset):
        container = QWidget()
        self.manual_dataset = dataset
        self.hc_map = {hc.name: hc for hc in self.manual_dataset}
        self.viewer_window = ViewerWindow()
        main_layout = QVBoxLayout(container)
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedWidth(80)
        self.reset_button.setStyleSheet("background-color: #e0e0e0; font-weight: bold;")
        self.reset_button.clicked.connect(self.reset_manual_sorting)

        reset_layout = QHBoxLayout()
        reset_layout.addStretch()
        reset_layout.addWidget(self.reset_button)
        main_layout.addLayout(reset_layout)
        self.deck_list_widget = MoleculeListWidget("deck")
        self.bin1_list_widget = MoleculeListWidget("bin1")
        self.bin2_list_widget = MoleculeListWidget("bin2")
        self.bin1_1_list_widget = MoleculeListWidget("bin1_1")
        self.bin1_2_list_widget = MoleculeListWidget("bin1_2")
        self.bin2_1_list_widget = MoleculeListWidget("bin2_1")
        self.bin2_2_list_widget = MoleculeListWidget("bin2_2")
        level0_layout = QHBoxLayout()
        deck_frame = self.create_bin_section("Unsorted Deck", self.deck_list_widget)
        level0_layout.addStretch(1)
        level0_layout.addWidget(deck_frame, 2)
        level0_layout.addStretch(1)
        main_layout.addLayout(level0_layout)
        level1_layout = QHBoxLayout()
        bin1_frame = self.create_bin_section("Bin 1", self.bin1_list_widget)
        bin2_frame = self.create_bin_section("Bin 2", self.bin2_list_widget)
        level1_layout.addWidget(bin1_frame)
        level1_layout.addWidget(bin2_frame)
        main_layout.addLayout(level1_layout)
        level2_layout = QHBoxLayout()
        bin1_1_frame = self.create_bin_section("Bin 1,1", self.bin1_1_list_widget)
        bin1_2_frame = self.create_bin_section("Bin 1,2", self.bin1_2_list_widget)
        bin2_1_frame = self.create_bin_section("Bin 2,1", self.bin2_1_list_widget)
        bin2_2_frame = self.create_bin_section("Bin 2,2", self.bin2_2_list_widget)
        level2_layout.addWidget(bin1_1_frame)
        level2_layout.addWidget(bin1_2_frame)
        level2_layout.addWidget(bin2_1_frame)
        level2_layout.addWidget(bin2_2_frame)
        main_layout.addLayout(level2_layout, stretch=1)
        scoreboard_layout = QHBoxLayout()
        self.score_labels = {}

        # --- MODIFIED for Step 3: Create more detailed score frames ---
        score_frame_1 = self._create_score_frame("Split 1: Deck -> Lvl 1", "split1", "Bin 1", "Bin 2")
        score_frame_2 = self._create_score_frame("Split 2: Bin 1 -> Lvl 2", "split2", "Bin 1,1", "Bin 1,2")
        score_frame_3 = self._create_score_frame("Split 3: Bin 2 -> Lvl 2", "split3", "Bin 2,1", "Bin 2,2")

        scoreboard_layout.addWidget(score_frame_1)
        scoreboard_layout.addWidget(score_frame_2)
        scoreboard_layout.addWidget(score_frame_3)
        main_layout.addLayout(scoreboard_layout)
        self.populate_deck()
        self._connect_manual_signals()
        self.update_manual_calculations()  # Initial update
        return container

    def reset_manual_sorting(self):
        for list_widget in [
            self.deck_list_widget,
            self.bin1_list_widget, self.bin2_list_widget,
            self.bin1_1_list_widget, self.bin1_2_list_widget,
            self.bin2_1_list_widget, self.bin2_2_list_widget
        ]:
            list_widget.clear()
        self.populate_deck()
        self.update_manual_calculations()

    def _connect_manual_signals(self):
        for list_widget in [self.deck_list_widget, self.bin1_list_widget, self.bin2_list_widget,
                            self.bin1_1_list_widget, self.bin1_2_list_widget,
                            self.bin2_1_list_widget, self.bin2_2_list_widget]:
            list_widget.model().rowsInserted.connect(self.update_manual_calculations)
            list_widget.model().rowsRemoved.connect(self.update_manual_calculations)
            list_widget.itemClicked.connect(self.show_3d_viewer)

    def show_3d_viewer(self, item):
        molecule_name = item.text().split('|')[0].strip()
        hc = self.hc_map.get(molecule_name)
        if hc:
            self.viewer_window.setWindowTitle(f"3D View: {hc.name}")
            self.viewer_window.update_view(hc.sdf_path)
            self.viewer_window.show()
            self.viewer_window.raise_()

    # --- REWRITTEN for Step 3: Full Calculation Logic ---
    def update_manual_calculations(self):
        # Helper to get hydrocarbon objects from a list widget
        def _get_hcs_from_list(list_widget):
            return [self.hc_map[list_widget.item(i).text().split('|')[0].strip()] for i in range(list_widget.count())]

        # Helper to update the stats labels for a single bin
        def _update_bin_stats(hcs, key_prefix):
            n = len(hcs)
            self.score_labels[f'{key_prefix}_count'].setText(f"Count (n): {n}")
            if n > 0:
                mean = np.mean([hc.boiling_point for hc in hcs])
                var = np.var([hc.boiling_point for hc in hcs]) if n > 1 else 0.0
                self.score_labels[f'{key_prefix}_mean'].setText(f"Mean BP: {mean:.2f}")
                self.score_labels[f'{key_prefix}_var'].setText(f"Variance: {var:.2f}")
            else:
                self.score_labels[f'{key_prefix}_mean'].setText("Mean BP: N/A")
                self.score_labels[f'{key_prefix}_var'].setText("Variance: N/A")

        # Get hydrocarbon lists for all bins
        hcs_bin1 = _get_hcs_from_list(self.bin1_list_widget)
        hcs_bin2 = _get_hcs_from_list(self.bin2_list_widget)
        hcs_bin1_1 = _get_hcs_from_list(self.bin1_1_list_widget)
        hcs_bin1_2 = _get_hcs_from_list(self.bin1_2_list_widget)
        hcs_bin2_1 = _get_hcs_from_list(self.bin2_1_list_widget)
        hcs_bin2_2 = _get_hcs_from_list(self.bin2_2_list_widget)

        # --- Update Split 1 ---
        _update_bin_stats(hcs_bin1, "split1_binA")
        _update_bin_stats(hcs_bin2, "split1_binB")
        cost1 = calculate_weighted_variance(hcs_bin1, hcs_bin2)
        self.score_labels['split1_cost'].setText(f"<b>TOTAL COST: {cost1:.2f}</b>")

        # --- Update Split 2 ---
        _update_bin_stats(hcs_bin1_1, "split2_binA")
        _update_bin_stats(hcs_bin1_2, "split2_binB")
        cost2 = calculate_weighted_variance(hcs_bin1_1, hcs_bin1_2)
        self.score_labels['split2_cost'].setText(f"<b>TOTAL COST: {cost2:.2f}</b>")

        # --- Update Split 3 ---
        _update_bin_stats(hcs_bin2_1, "split3_binA")
        _update_bin_stats(hcs_bin2_2, "split3_binB")
        cost3 = calculate_weighted_variance(hcs_bin2_1, hcs_bin2_2)
        self.score_labels['split3_cost'].setText(f"<b>TOTAL COST: {cost3:.2f}</b>")

    def populate_deck(self):
        for hc in self.manual_dataset:
            item_text = (
                f"{hc.name:<20s} | BP: {hc.boiling_point:>5.1f}°C | #C: {hc.num_carbons} | #Br: {hc.num_branches}")
            list_item = QListWidgetItem(item_text)
            font = QFont("Lato");
            font.setPointSize(9)
            list_item.setFont(font)
            self.deck_list_widget.addItem(list_item)

    def create_bin_section(self, title, list_widget):
        layout = QVBoxLayout()

        label = QLabel(f"<h3>{title}</h3>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(list_widget)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setLayout(layout)

        frame.setStyleSheet("""
            QFrame {
                border: 3px solid #6C1D45;  /* Outer frame: Cobber Maroon */
                border-radius: 6px;
                background-color: #FAFAFA;
            }
            QLabel {
                color: #6C1D45;             /* Label text color: Cobber Maroon */
                border: none;
            }
            QListWidget {
                border: 1px solid #333333; /* Inner list: thin charcoal border */
                border-radius: 4px;
                background-color: white;
            }
        """)

        return frame

    # --- REWRITTEN for Step 3: Detailed Scoreboard Creation ---
    def _create_score_frame(self, title, key_prefix, binA_name, binB_name):
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)

        title_label = QLabel(f"<b>{title}</b>")
        layout.addWidget(title_label)

        # Create layouts for Bin A and Bin B stats side-by-side
        bins_layout = QHBoxLayout()

        # --- Bin A Stats ---
        binA_layout = QVBoxLayout()
        binA_label = QLabel(f"<i>{binA_name}</i>")
        self.score_labels[f'{key_prefix}_binA_count'] = QLabel("Count (n): 0")
        self.score_labels[f'{key_prefix}_binA_mean'] = QLabel("Mean BP: N/A")
        self.score_labels[f'{key_prefix}_binA_var'] = QLabel("Variance: N/A")
        binA_layout.addWidget(binA_label)
        binA_layout.addWidget(self.score_labels[f'{key_prefix}_binA_count'])
        binA_layout.addWidget(self.score_labels[f'{key_prefix}_binA_mean'])
        binA_layout.addWidget(self.score_labels[f'{key_prefix}_binA_var'])

        # --- Bin B Stats ---
        binB_layout = QVBoxLayout()
        binB_label = QLabel(f"<i>{binB_name}</i>")
        self.score_labels[f'{key_prefix}_binB_count'] = QLabel("Count (n): 0")
        self.score_labels[f'{key_prefix}_binB_mean'] = QLabel("Mean BP: N/A")
        self.score_labels[f'{key_prefix}_binB_var'] = QLabel("Variance: N/A")
        binB_layout.addWidget(binB_label)
        binB_layout.addWidget(self.score_labels[f'{key_prefix}_binB_count'])
        binB_layout.addWidget(self.score_labels[f'{key_prefix}_binB_mean'])
        binB_layout.addWidget(self.score_labels[f'{key_prefix}_binB_var'])

        bins_layout.addLayout(binA_layout)
        bins_layout.addLayout(binB_layout)
        layout.addLayout(bins_layout)

        # --- Total Cost ---
        self.score_labels[f'{key_prefix}_cost'] = QLabel("<b>TOTAL COST: N/A</b>")
        layout.addWidget(self.score_labels[f'{key_prefix}_cost'])

        return frame


if __name__ == "__main__":
    def get_asset_path(filename):
        return os.path.join(project_root, "assets", filename)


    sdf = lambda name: get_asset_path(f"{name.lower().replace('-', '')}.sdf")
    manual_dataset = [
        Hydrocarbon("Butane", "C4H10", 4, 0, -0.5, sdf("Butane")),
        Hydrocarbon("Pentane", "C5H12", 5, 0, 36.1, sdf("Pentane")),
        Hydrocarbon("Hexane", "C6H14", 6, 0, 68.7, sdf("Hexane")),
        Hydrocarbon("Heptane", "C7H16", 7, 0, 98.4, sdf("Heptane")),
        Hydrocarbon("Octane", "C8H18", 8, 0, 125.7, sdf("Octane")),
        Hydrocarbon("Isobutane", "C4H10", 4, 1, -11.7, sdf("Isobutane")),
        Hydrocarbon("Isopentane", "C5H12", 5, 1, 27.8, sdf("Isopentane")),
        Hydrocarbon("3-Methylpentane", "C6H14", 6, 1, 63.3, sdf("3-Methylpentane")),
        Hydrocarbon("2-Methylheptane", "C8H18", 8, 1, 117.6, sdf("2-Methylheptane")),
        Hydrocarbon("Neopentane", "C5H12", 5, 2, 9.5, sdf("Neopentane")),
        Hydrocarbon("2,2-Dimethylhexane", "C8H18", 8, 2, 106.8, sdf("2,2-Dimethylhexane")),
    ]
    app = QApplication(sys.argv)
    window = CobberTreeApp(sorted(manual_dataset, key=lambda hc: (hc.num_carbons, hc.num_branches)))
    window.show()
    sys.exit(app.exec())