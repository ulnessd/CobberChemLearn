# CobberHbond.py
# An application for discovering hydrogen bonding patterns using K-Means clustering.
# Refactored for the CobberLearnChem launcher.

import sys
import os
import numpy as np
import pandas as pd
import random
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QFileDialog, QPushButton, QLabel, QComboBox, QHBoxLayout,
    QMessageBox, QCheckBox, QListWidget, QListWidgetItem,
    QGroupBox, QSlider
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor  # Added for branding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Add project root to path for robust imports ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('.')


# --- Core Engine (LOGIC UNCHANGED) ---

class MoleculeVisualizer:
    @staticmethod
    def plot_molecules(ax, molecule1_type, molecule2_type, distance, angle, line_length=2):
        ax.clear()

        # Keep original angle for positioning calculations
        angle_rad = np.deg2rad(angle)
        x2 = distance * np.cos(angle_rad)
        y2 = distance * np.sin(angle_rad)

        # --- Draw Molecule 1 ---
        ax.plot([0, -line_length], [0, 0], 'k-', lw=2)
        ax.plot(0, 0, 'ro', markersize=10, label='Terminal Atom for Molecule 1')
        ax.text(-line_length, 0.2, molecule1_type, fontsize=12, ha='center')

        # --- Draw Molecule 2 ---
        if distance != 0:
            u_x, u_y = x2 / distance, y2 / distance
        else:
            u_x, u_y = 0, 0
        dx2, dy2 = x2 + line_length * u_x, y2 + line_length * u_y
        ax.plot([x2, dx2], [y2, dy2], 'b-', lw=2)
        ax.plot(x2, y2, 'go', markersize=10, label='Terminal Atom for Molecule 2')
        ax.text(dx2, dy2 + 0.2, molecule2_type, fontsize=12, ha='center')

        # --- Draw the distance and angle labels correctly ---
        ax.plot([0, x2], [0, y2], 'r--', lw=1)
        ax.text(x2 / 2, y2 / 2 - 0.3, f'{distance:.2f} Å', fontsize=10, color='red', ha='center')

        # NEW: Draw the arc from the negative x-axis (pi) to the molecule's angle
        angle_arc = np.linspace(np.pi, angle_rad, 100)
        arc_radius = distance / 4
        ax.plot(arc_radius * np.cos(angle_arc), arc_radius * np.sin(angle_arc), 'r-', lw=1)

        # NEW: Calculate the chemical angle for display and position the text correctly
        chemical_angle_display = 180 - angle
        text_angle_rad = (np.pi + angle_rad) / 2
        ax.text(arc_radius * np.cos(text_angle_rad) + 0.2, arc_radius * np.sin(text_angle_rad) + 0.2,
                f'{chemical_angle_display:.2f}°',
                fontsize=10, color='red')

        # --- Set plot limits and labels ---
        ax.set_xlim(-3.5, 7.5);
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal', 'box');
        ax.set_xlabel('X (Å)');
        ax.set_ylabel('Y (Å)')
        ax.set_title('Molecular Interaction Visualization');
        ax.grid(True);
        ax.legend()


# --- GUI Components (LOGIC UNCHANGED, branding added) ---

class DataTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # For branding
        self.layout = QVBoxLayout(self);
        self.dataset = None;
        self.timer = QTimer(self)
        self.features = ['Physical_Distance (Å)', 'Physical_Angle (°)', 'Energy']
        top_controls_layout = QHBoxLayout();
        self.load_button = QPushButton('Load Dataset')
       # self.load_button.setStyleSheet(
        #    f"background-color: {self.main_window.cobber_gold.name()}; color: {self.main_window.cobber_maroon.name()}; font-weight: bold; padding: 5px; border-radius: 3px;")
        self.load_button.clicked.connect(self.load_data);
        top_controls_layout.addWidget(self.load_button)
        self.data_info_label = QLabel('No dataset loaded.');
        self.data_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter);
        top_controls_layout.addWidget(self.data_info_label, 1)
        self.layout.addLayout(top_controls_layout);
        main_layout = QHBoxLayout();
        left_column_layout = QVBoxLayout()
        plotting_groupbox = QGroupBox("Data Visualization");
        plotting_layout = QVBoxLayout(plotting_groupbox);
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(['Histogram', '3D Scatter Plot', '2D Scatter Plot']);
        self.plot_combo.currentTextChanged.connect(self.update_plot_controls);
        plotting_layout.addWidget(self.plot_combo)
        self.plot_button = QPushButton('Plot Histogram');
        self.plot_button.clicked.connect(self.plot_data);
        plotting_layout.addWidget(self.plot_button)
        self.scatter_controls = QWidget();
        scatter_layout = QHBoxLayout(self.scatter_controls);
        scatter_layout.setContentsMargins(0, 0, 0, 0)
        self.x_axis_combo = QComboBox();
        self.y_axis_combo = QComboBox();
        scatter_layout.addWidget(QLabel("X-Axis:"));
        scatter_layout.addWidget(self.x_axis_combo);
        scatter_layout.addWidget(QLabel("Y-Axis:"));
        scatter_layout.addWidget(self.y_axis_combo)
        plotting_layout.addWidget(self.scatter_controls);
        left_column_layout.addWidget(plotting_groupbox)
        interaction_groupbox = QGroupBox("Interaction Geometry")
        interaction_layout = QVBoxLayout(interaction_groupbox)

        self.explore_button = QPushButton('Show Single Random Pair')
        self.explore_button.clicked.connect(self.explore_pair_interaction)
        interaction_layout.addWidget(self.explore_button)

        self.animate_button = QPushButton('Animate Pair Geometries')
        self.animate_button.clicked.connect(self.animate_geometries)
        interaction_layout.addWidget(self.animate_button)

        # Add the animation speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Slower"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(10, 1000)  # Delay in milliseconds (10ms = fast, 200ms = slow)
        self.speed_slider.setValue(750)  # Start slower
        self.speed_slider.setInvertedAppearance(True)  # Makes left side "faster"
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(QLabel("Faster"))
        interaction_layout.addLayout(speed_layout)

        left_column_layout.addWidget(interaction_groupbox)
        left_column_layout.addStretch();
        main_layout.addLayout(left_column_layout)
        self.plot_figure = Figure(figsize=(7, 7));
        self.plot_canvas = FigureCanvas(self.plot_figure);
        main_layout.addWidget(self.plot_canvas, 1)
        self.layout.addLayout(main_layout);
        self.update_plot_controls(self.plot_combo.currentText())
        self.timer.timeout.connect(self._animation_step)
        self.speed_slider.valueChanged.connect(lambda value: self.timer.setInterval(value))

    def update_plot_controls(self, text):
        if text == "2D Scatter Plot":
            self.scatter_controls.setVisible(True); self.plot_button.setText("Plot 2D Scatter")
        else:
            self.scatter_controls.setVisible(False); self.plot_button.setText(f"Plot {text}")

    def load_data(self):
        assets_dir = os.path.join(project_root, "assets")
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Dataset", assets_dir, "CSV Files (*.csv)")
        if file_name:
            try:
                self.dataset = pd.read_csv(file_name);
                #self.dataset['Physical_Angle (°)'] += 180
                self.x_axis_combo.clear();
                self.x_axis_combo.addItems(self.features)
                self.y_axis_combo.clear();
                self.y_axis_combo.addItems(self.features);
                self.data_info_label.setText(f'Loaded: {os.path.basename(file_name)}');
                self.main_window.on_data_loaded()
            except Exception as e:
                self.dataset = None; QMessageBox.warning(self, 'Error', f'Failed to load dataset: {e}')

    def plot_data(self):
        if self.dataset is None: QMessageBox.warning(self, 'Error', 'Please load a dataset first.'); return
        self.plot_figure.clear();
        plot_type = self.plot_combo.currentText()
        if plot_type == 'Histogram':
            self.plot_histogram()
        elif plot_type == '2D Scatter Plot':
            self.plot_scatter()
        elif plot_type == '3D Scatter Plot':
            self.plot_3d_scatter()

    def plot_histogram(self):
        df_plot = self.dataset.copy()
        df_plot['Physical_Angle (°)'] += 180
        ax = self.plot_figure.add_subplot(111);
        #self.dataset[self.features].hist(ax=ax);
        df_plot[self.features].hist(ax=ax);
        self.plot_figure.tight_layout();
        self.plot_canvas.draw()

    def plot_scatter(self):
        ax = self.plot_figure.add_subplot(111)
        x_feature = self.x_axis_combo.currentText()
        y_feature = self.y_axis_combo.currentText()

        # --- Create a temporary, transformed copy for plotting ---
        df_plot = self.dataset.copy()
        df_plot['Physical_Angle (°)'] += 180

        reply = QMessageBox.question(self, 'Normalize Features', 'Plot normalized features?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            scaler = MinMaxScaler()
            df_plot[self.features] = scaler.fit_transform(df_plot[self.features])

        ax.scatter(df_plot[x_feature], df_plot[y_feature], alpha=0.5, s=1)
        ax.set_xlabel(x_feature);
        ax.set_ylabel(y_feature)
        ax.set_title(f'{y_feature} vs. {x_feature}');
        ax.grid(True)
        self.plot_canvas.draw()

    def plot_3d_scatter(self):
        ax = self.plot_figure.add_subplot(111, projection='3d')

        # --- Create a temporary, transformed copy for plotting ---
        df_plot = self.dataset.copy()
        df_plot['Physical_Angle (°)'] += 180

        reply = QMessageBox.question(self, 'Normalize Features', 'Plot normalized features?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            scaler = MinMaxScaler()
            df_plot[self.features] = scaler.fit_transform(df_plot[self.features])

        ax.scatter(df_plot[self.features[0]], df_plot[self.features[1]], df_plot[self.features[2]], alpha=0.5, s=1)
        ax.set_xlabel(self.features[0]);
        ax.set_ylabel('Interaction Angle (°)')
        ax.set_zlabel(self.features[2])
        self.plot_canvas.draw()

    def explore_pair_interaction(self):
        if self.dataset is None: return
        self.plot_figure.clear();
        ax = self.plot_figure.add_subplot(111);
        random_row = self.dataset.sample(n=1).iloc[0]
        MoleculeVisualizer.plot_molecules(ax, random_row['Molecule_1'], random_row['Molecule_2'],
                                          random_row['Physical_Distance (Å)'], random_row['Physical_Angle (°)'])
        self.plot_canvas.draw()

    def animate_geometries(self):
        if self.dataset is None:
            QMessageBox.warning(self, 'Error', 'Please load a dataset first.')
            return

        self.plot_figure.clear()
        self.animation_ax = self.plot_figure.add_subplot(111)
        self.molecule_indices = random.sample(range(len(self.dataset)), 100)
        self.current_index = 0

        # --- MODIFIED: Use the instance timer and get speed from the slider ---
        self.timer.start(self.speed_slider.value())

    def _animation_step(self):
        if self.current_index < len(self.molecule_indices):
            idx = self.molecule_indices[self.current_index];
            row = self.dataset.iloc[idx]
            MoleculeVisualizer.plot_molecules(self.animation_ax, row['Molecule_1'], row['Molecule_2'],
                                              row['Physical_Distance (Å)'], row['Physical_Angle (°)'])
            self.plot_canvas.draw();
            self.current_index += 1
        else:
            self.timer.stop(); QMessageBox.information(self, "Animation Complete",
                                                       "Finished showing 100 random pair geometries.")


class ModelTab(QWidget):
    def __init__(self, data_tab, main_window):
        super().__init__()
        self.main_window = main_window;
        self.data_tab = data_tab;
        self.results_tab = None;
        self.trained_model = None
        self.layout = QVBoxLayout(self);
        groupbox = QGroupBox("K-Means Clustering Setup");
        box_layout = QVBoxLayout(groupbox)
        params_layout = QHBoxLayout();
        params_layout.addWidget(QLabel('Number of Clusters (k):'));
        self.num_clusters_combo = QComboBox()
        self.num_clusters_combo.addItems([str(i) for i in range(2, 11)]);
        params_layout.addWidget(self.num_clusters_combo);
        box_layout.addLayout(params_layout)
        self.scaling_checkbox = QCheckBox('Normalize Features for Clustering');
        self.scaling_checkbox.setChecked(True)
        self.scaling_checkbox.setToolTip(
            "It is highly recommended to keep this checked for distance-based algorithms.");
        box_layout.addWidget(self.scaling_checkbox)
        self.run_button = QPushButton('Run K-Means Clustering')
       # self.run_button.setStyleSheet(
        #    f"background-color: {self.main_window.cobber_gold.name()}; color: {self.main_window.cobber_maroon.name()}; font-weight: bold; padding: 5px; border-radius: 3px;")
        self.run_button.clicked.connect(self.run_clustering);
        box_layout.addWidget(self.run_button);
        self.layout.addWidget(groupbox);
        self.layout.addStretch()

    def run_clustering(self):
        if self.data_tab.dataset is None: QMessageBox.warning(self, 'Error', 'Please load a dataset first.'); return
        n_clusters = int(self.num_clusters_combo.currentText());
        features = self.data_tab.dataset[['Physical_Distance (Å)', 'Physical_Angle (°)', 'Energy']]
        if self.scaling_checkbox.isChecked():
            scaler = MinMaxScaler(); features_scaled = scaler.fit_transform(features)
        else:
            features_scaled = features.values
        try:
            model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
            self.data_tab.dataset['Cluster'] = model.fit_predict(features_scaled);
            self.trained_model = model
            QMessageBox.information(self, 'Success',
                                    f'K-Means clustering completed successfully.\nData is now ready for analysis in the Results tab.')
            if self.results_tab: self.results_tab.on_clustering_complete()
        except Exception as e:
            QMessageBox.critical(self, "Clustering Error", f"An error occurred: {e}")


class ResultsTab(QWidget):
    def __init__(self, data_tab, model_tab, main_window):
        super().__init__()
        self.main_window = main_window;
        self.data_tab = data_tab;
        self.model_tab = model_tab
        self.layout = QVBoxLayout(self);
        self.upper_layout = QHBoxLayout()
        self.figure = Figure();
        self.canvas = FigureCanvas(self.figure);
        self.upper_layout.addWidget(self.canvas)
        self.heatmap_figure = Figure();
        self.heatmap_canvas = FigureCanvas(self.heatmap_figure);
        self.upper_layout.addWidget(self.heatmap_canvas)
        self.layout.addLayout(self.upper_layout);
        plot_options_layout = QHBoxLayout();
        plot_options_layout.addWidget(QLabel('Color by:'));
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(['Cluster', 'Molecule Pair']);
        plot_options_layout.addWidget(self.color_by_combo)
        self.centroids_checkbox = QCheckBox("Show Centroids Only");
        plot_options_layout.addWidget(self.centroids_checkbox);
        plot_options_layout.addStretch()
        self.plot_button = QPushButton('Plot Clusters')
        #self.plot_button.setStyleSheet(
         #   f"background-color: {self.main_window.cobber_gold.name()}; color: {self.main_window.cobber_maroon.name()}; font-weight: bold; padding: 5px; border-radius: 3px;")
        self.plot_button.clicked.connect(self.plot_clusters);
        plot_options_layout.addWidget(self.plot_button);
        self.layout.addLayout(plot_options_layout)
        self.color_by_combo.currentTextChanged.connect(self.plot_clusters);
        self.centroids_checkbox.toggled.connect(self.plot_clusters)
        self.cluster_list_widget = QListWidget();
        self.cluster_list_widget.itemClicked.connect(self.display_cluster_histogram);
        self.layout.addWidget(self.cluster_list_widget)
        self.show_cluster_map_button = QPushButton('Show Cluster Heatmap');
        self.show_cluster_map_button.clicked.connect(self.show_cluster_map);
        self.layout.addWidget(self.show_cluster_map_button)
        self.histogram_figure = Figure();
        self.histogram_canvas = FigureCanvas(self.histogram_figure);
        self.layout.addWidget(self.histogram_canvas)
        self.on_clustering_complete(clear_only=True)

    def on_clustering_complete(self, clear_only=False):
        self.figure.clear();
        self.canvas.draw();
        self.heatmap_figure.clear();
        self.heatmap_canvas.draw();
        self.histogram_figure.clear();
        self.histogram_canvas.draw()
        if clear_only:
            self.plot_button.setEnabled(False);
            self.cluster_list_widget.setEnabled(False);
            self.show_cluster_map_button.setEnabled(False);
            self.centroids_checkbox.setEnabled(False);
            self.color_by_combo.setEnabled(False);
            self.cluster_list_widget.clear();
            return
        if 'Cluster' in self.data_tab.dataset.columns:
            self.cluster_list_widget.clear();
            clusters = sorted(self.data_tab.dataset['Cluster'].unique())
            for cluster in clusters: item = QListWidgetItem(f'Cluster {cluster}'); item.setData(
                Qt.ItemDataRole.UserRole, cluster); self.cluster_list_widget.addItem(item)
            self.plot_button.setEnabled(True);
            self.cluster_list_widget.setEnabled(True);
            self.show_cluster_map_button.setEnabled(True);
            self.centroids_checkbox.setEnabled(True);
            self.color_by_combo.setEnabled(True);
            self.plot_clusters()

    def plot_clusters(self):
        if self.data_tab.dataset is None or 'Cluster' not in self.data_tab.dataset.columns: return
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')

        # --- Create a temporary, transformed copy for plotting ---
        data_to_plot = self.data_tab.dataset.copy()
        data_to_plot['Physical_Angle (°)'] += 180

        was_normalized = self.model_tab.scaling_checkbox.isChecked()
        if was_normalized:
            scaler = MinMaxScaler()
            data_to_plot[self.data_tab.features] = scaler.fit_transform(data_to_plot[self.data_tab.features])

        if self.centroids_checkbox.isChecked():
            model = self.model_tab.trained_model
            if model and isinstance(model, KMeans):
                centroids = model.cluster_centers_;
                ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=200, c='red',
                           label='Centroids');
                ax.set_title("K-Means Cluster Centroids")
            else:
                ax.text(0.5, 0.5, 0.5, "Centroids only available for K-Means.", ha='center')
        else:
            color_by = self.color_by_combo.currentText();
            group_field = 'Cluster' if color_by == 'Cluster' else 'Molecule Pair'
            if group_field == 'Molecule Pair': data_to_plot['Molecule Pair'] = data_to_plot['Molecule_1'] + ' - ' + \
                                                                               data_to_plot['Molecule_2']
            groups = data_to_plot[group_field].unique();
            cmap = plt.get_cmap('viridis');
            colors = cmap(np.linspace(0, 1, len(groups)));
            group_to_color = dict(zip(groups, colors))
            for name, group in data_to_plot.groupby(group_field): ax.scatter(group[self.data_tab.features[0]],
                                                                             group[self.data_tab.features[1]],
                                                                             group[self.data_tab.features[2]],
                                                                             label=name, color=group_to_color[name],
                                                                             alpha=0.6, s=1)
            #if len(groups) < 15: ax.legend(fontsize='small')
        ax.set_xlabel('Distance');
        ax.set_ylabel('Interaction Angle');
        ax.set_zlabel('Energy');
        self.canvas.draw()

    def display_cluster_histogram(self, item):
        self.selected_cluster = item.data(Qt.ItemDataRole.UserRole);
        data = self.data_tab.dataset;
        cluster_data = data[data['Cluster'] == self.selected_cluster].copy()
        cluster_data['Molecule Pair'] = cluster_data['Molecule_1'] + ' - ' + cluster_data['Molecule_2'];
        pair_counts = cluster_data['Molecule Pair'].value_counts().sort_index()
        self.histogram_figure.clear();
        ax = self.histogram_figure.add_subplot(111);
        pair_counts.plot(kind='bar', ax=ax)
        ax.set_xlabel('Molecule Pair');
        ax.set_ylabel('Count');
        ax.set_title(f'Molecule Pair Distribution in Cluster {self.selected_cluster}');
        self.histogram_figure.tight_layout();
        self.histogram_canvas.draw()

    def show_cluster_map(self):
        if not hasattr(self, 'selected_cluster') or self.cluster_list_widget.currentItem() is None: QMessageBox.warning(
            self, 'Error', 'Please select a cluster from the list first.'); return
        self.display_heatmap()

    def display_heatmap(self):
        self.heatmap_figure.clear();
        ax = self.heatmap_figure.add_subplot(111)
        cluster_data = self.data_tab.dataset[self.data_tab.dataset['Cluster'] == self.selected_cluster];
        x_positions, y_positions = [], []
        for _, row in cluster_data.iterrows():
            angle_rad = np.deg2rad(row['Physical_Angle (°)']);
            x_positions.append(row['Physical_Distance (Å)'] * np.cos(angle_rad));
            y_positions.append(row['Physical_Distance (Å)'] * np.sin(angle_rad))
        heatmap, xedges, yedges = np.histogram2d(x_positions, y_positions, bins=50, range=[[-3.5, 7.5], [-6, 6]]);
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='auto');
        self.heatmap_figure.colorbar(im, ax=ax, label='Frequency')
        ax.plot(0, 0, 'wo', markersize=10, markeredgecolor='black', label='Term. Atom Mol. 1');
        ax.plot([0, -2], [0, 0], 'w-', lw=3, label='Molecule 1')
        ax.set_title(f'Heatmap of Molecule 2 Positions in Cluster {self.selected_cluster}');
        ax.set_xlabel('X Position (Å)');
        ax.set_ylabel('Y Position (Å)');
        ax.set_aspect('equal', 'box');
        ax.legend();
        self.heatmap_canvas.draw()


# --- Main Application Class (RENAMED and BRANDED) ---
class CobberHbondApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # --- BRANDING ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle('CobberHbond')
        self.setGeometry(50, 50, 1200, 750)
        self.setFont(self.lato_font)

        self.tabs = QTabWidget()
        self.data_tab = DataTab(self)
        self.model_tab = ModelTab(self.data_tab, self)
        self.results_tab = ResultsTab(self.data_tab, self.model_tab, self)
        self.model_tab.results_tab = self.results_tab

        self.tabs.addTab(self.data_tab, 'Data')
        self.tabs.addTab(self.model_tab, 'Model')
        self.tabs.addTab(self.results_tab, 'Results')
        self.setCentralWidget(self.tabs)

    def on_data_loaded(self):
        """A method that can be called by the DataTab to signal the main window."""
        QMessageBox.information(self, "Dataset Loaded",
                                "Dataset has loaded successfully!")
        #self.tabs.setCurrentIndex(1)  # Switch to model tab automatically


# --- Standalone Execution Guard ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = CobberHbondApp()
    main_window.show()
    sys.exit(app.exec())