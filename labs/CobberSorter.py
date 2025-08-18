# CobberSorter.py
# An application to classify synthetic polymer electron microscopy images
# using a pre-trained Convolutional Neural Network (CNN).
# Refactored for the CobberLearnChem launcher.

import sys
import os
import cv2
import numpy as np
import pickle
import random
import time

# Add project root to path for robust imports
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('.')

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox,
    QProgressBar, QTextEdit, QStackedWidget
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QTextCursor
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QRunnable, QThreadPool

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns


# --- Core Engine & Worker Threads (LOGIC UNCHANGED) ---

def generate_bead_image(image_size=(256, 256), small_bead_radius=(8, 10), large_bead_radius=(18, 20), num_beads=50,
                        category='mixed', small_prob=0.5):
    image = np.zeros(image_size, dtype=np.uint8);
    existing_beads = []
    for _ in range(num_beads):
        if category == 'pure_small':
            radius = random.randint(*small_bead_radius)
        elif category == 'pure_large':
            radius = random.randint(*large_bead_radius)
        else:
            radius = random.randint(*small_bead_radius) if random.random() < small_prob else random.randint(
                *large_bead_radius)
        for _ in range(10):
            x = random.randint(radius, image_size[1] - radius);
            y = random.randint(radius, image_size[0] - radius)
            if not any(np.sqrt((x - b['x']) ** 2 + (y - b['y']) ** 2) < (radius + b['radius'] + 2) for b in
                       existing_beads): break
        else:
            continue
        existing_beads.append({'x': x, 'y': y, 'radius': radius});
        bead_canvas = np.zeros((2 * radius, 2 * radius), dtype=np.uint8)
        base_intensity = 50;
        cv2.circle(bead_canvas, (radius, radius), radius, base_intensity, -1)
        for i in range(1, radius):
            intensity = min(base_intensity + int((i / radius) * 200), 255)
            cv2.circle(bead_canvas, (radius, radius), radius - i, intensity, 1)
        roi = image[y - radius:y + radius, x - radius:x + radius]
        if roi.shape[0] != 2 * radius or roi.shape[1] != 2 * radius: continue
        image[y - radius:y + radius, x - radius:x + radius] = cv2.max(roi, bead_canvas)
    noise = np.random.normal(0, 10, image_size)
    return np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)


class WorkerSignals(QObject):
    progress = pyqtSignal(int);
    finished = pyqtSignal(float, list, list);
    error = pyqtSignal(str)
    log = pyqtSignal(str, QColor);
    display_image = pyqtSignal(QImage);
    single_result = pyqtSignal(str, str)


class SorterWorker(QRunnable):
    def __init__(self, model, le, num_iterations=200):
        super().__init__();
        self.model = model;
        self.le = le;
        self.signals = WorkerSignals()
        self.num_iterations = num_iterations;
        self.is_running = True;
        self.y_true = [];
        self.y_pred = []

    def run(self):
        bad_errors = 0
        try:
            for i in range(self.num_iterations):
                if not self.is_running: break
                rand = random.random()
                if rand < 0.25:
                    category, small_prob, actual_bin = 'pure_small', 0.0, 'pure_small'
                elif rand < 0.50:
                    category, small_prob, actual_bin = 'pure_large', 0.0, 'pure_large'
                else:
                    category, small_prob = 'mixed', random.uniform(0.05, 0.95)
                    if small_prob >= 0.92:
                        actual_bin = 'acceptable_small'
                    elif small_prob <= 0.08:
                        actual_bin = 'acceptable_large'
                    else:
                        actual_bin = 'reject'
                img_np = generate_bead_image(category=category, small_prob=small_prob, num_beads=random.randint(40, 90))
                prediction = self.predict_class(img_np)
                if self.num_iterations == 1:
                    self.signals.single_result.emit(actual_bin, prediction)
                else:
                    self.y_true.append(actual_bin); self.y_pred.append(prediction)
                log_color = QColor("green")
                if actual_bin != prediction:
                    if self.is_bad_error(actual_bin, prediction):
                        bad_errors += 1; log_color = QColor("red")
                    else:
                        log_color = QColor("blue")
                self.signals.log.emit(f"Image {i + 1}: Actual='{actual_bin}', Predicted='{prediction}'", log_color)
                height, width = img_np.shape
                q_img = QImage(img_np.data, width, height, width, QImage.Format.Format_Grayscale8)
                self.signals.display_image.emit(q_img.copy())
                if self.num_iterations > 1: self.signals.progress.emit(
                    int(((i + 1) / self.num_iterations) * 100)); time.sleep(0.05)
            if self.num_iterations > 1:
                error_rate = (bad_errors / self.num_iterations) * 100
                self.signals.finished.emit(error_rate, self.y_true, self.y_pred)
        except Exception as e:
            self.signals.error.emit(f"An error occurred in the sorting thread: {e}")

    def predict_class(self, img):
        img_resized = cv2.resize(img, (256, 256));
        img_normalized = img_resized / 255.0
        img_expanded = np.expand_dims(img_normalized, axis=-1);
        img_input = np.expand_dims(img_expanded, axis=0)
        preds = self.model.predict(img_input, verbose=0);
        pred_class_idx = np.argmax(preds, axis=1)[0]
        return self.le.inverse_transform([pred_class_idx])[0]

    def is_bad_error(self, actual, predicted):
        small_types = {'pure_small', 'acceptable_small'};
        large_types = {'pure_large', 'acceptable_large'}
        if (actual == 'reject' and predicted != 'reject') or (predicted == 'reject' and actual != 'reject'): return True
        if (actual in small_types and predicted in large_types) or (
                actual in large_types and predicted in small_types): return True
        return False

    def stop(self):
        self.is_running = False


# --- RENAMED Main Application Class ---
class CobberSorterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # --- BRANDING ---
        self.cobber_maroon = QColor(108, 29, 69);
        self.cobber_gold = QColor(234, 170, 0);
        self.lato_font = QFont("Lato")
        self.setWindowTitle("CobberSorter");
        self.setGeometry(100, 100, 1100, 700);
        self.setFont(self.lato_font)

        self.model, self.le = None, None;
        self.cm_y_true, self.cm_y_pred = [], [];
        self.threadpool = QThreadPool()
        main_widget = QWidget();
        self.setCentralWidget(main_widget);
        main_layout = QVBoxLayout(main_widget)
        controls_layout = QHBoxLayout()
        self.load_model_button = QPushButton("Load Trained Model");
        self.inspect_batch_button = QPushButton("Inspect Single Batch");
        self.run_sort_button = QPushButton("Run Full Sort (200 Images)")
        self.inspect_batch_button.setEnabled(False);
        self.run_sort_button.setEnabled(False)
        # --- BRANDING ---
        button_style = f"background-color: {self.cobber_gold.name()}; color: {self.cobber_maroon.name()}; font-weight: bold; padding: 5px; border-radius: 3px;"
        #self.load_model_button.setStyleSheet(button_style);
        #self.run_sort_button.setStyleSheet(button_style)

        controls_layout.addWidget(self.load_model_button);
        controls_layout.addWidget(self.inspect_batch_button);
        controls_layout.addWidget(self.run_sort_button)
        display_layout = QHBoxLayout();
        image_area_layout = QVBoxLayout()
        self.display_stack = QStackedWidget();
        self.image_label = QLabel("Load a model to begin.");
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(512, 512);
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.cm_canvas = FigureCanvas(Figure(figsize=(6, 6)));
        self.display_stack.addWidget(self.image_label);
        self.display_stack.addWidget(self.cm_canvas)
        image_area_layout.addWidget(self.display_stack)
        self.cm_button = QPushButton("Show Confusion Matrix");
        self.cm_button.setEnabled(False);
        image_area_layout.addWidget(self.cm_button)
        log_v_layout = QVBoxLayout();
        log_label = QLabel("<b>Classification Log:</b>");
        self.log_text = QTextEdit();
        self.log_text.setReadOnly(True);
        self.log_text.setFont(QFont("Courier", 9))
        log_v_layout.addWidget(log_label);
        log_v_layout.addWidget(self.log_text)
        display_layout.addLayout(image_area_layout);
        display_layout.addLayout(log_v_layout)
        self.progress_bar = QProgressBar();
        self.progress_bar.setVisible(False)
        main_layout.addLayout(controls_layout);
        main_layout.addLayout(display_layout);
        main_layout.addWidget(self.progress_bar)

        self.load_model_button.clicked.connect(self.load_trained_model);
        self.inspect_batch_button.clicked.connect(self.inspect_batch)
        self.run_sort_button.clicked.connect(self.run_sort);
        self.cm_button.clicked.connect(self.toggle_cm_view)

    # --- All other methods have UNCHANGED logic ---
    def load_trained_model(self):
        assets_dir = os.path.join(project_root, "assets")
        model_file, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", assets_dir,
                                                    "Keras Model Files (*.keras *.h5)")
        if not model_file: return
        le_file, _ = QFileDialog.getOpenFileName(self, "Select Label Encoder", os.path.dirname(model_file),
                                                 "Pickle Files (*.pkl)")
        if not le_file: QMessageBox.warning(self, "Load Canceled",
                                            "A label encoder file (.pkl) is required to proceed."); return
        try:
            self.model = load_model(model_file)
            with open(le_file, 'rb') as f:
                self.le = pickle.load(f)
            self.log_text.setText(f"Successfully loaded model:\n{os.path.basename(model_file)}\n\n")
            self.inspect_batch_button.setEnabled(True);
            self.run_sort_button.setEnabled(True);
            self.cm_button.setEnabled(False)
            self.cm_y_true, self.cm_y_pred = [], [];
            self.display_stack.setCurrentWidget(self.image_label)
            self.cm_button.setText("Show Confusion Matrix")
            QMessageBox.information(self, "Success", "Model and label encoder loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Loading Error", f"Failed to load files: {e}"); self.model, self.le = None, None

    def inspect_batch(self):
        if not self.model or not self.le: return
        self.display_stack.setCurrentWidget(self.image_label);
        self.cm_button.setText("Show Confusion Matrix")
        self.set_buttons_enabled(False);
        worker = SorterWorker(self.model, self.le, num_iterations=1)
        worker.signals.log.connect(self.log_message);
        worker.signals.display_image.connect(self.display_image)
        worker.signals.single_result.connect(self.handle_single_result)
        worker.signals.finished.connect(lambda: self.set_buttons_enabled(True));
        worker.signals.error.connect(lambda e: self.set_buttons_enabled(True))
        self.threadpool.start(worker)

    def handle_single_result(self, actual, predicted):
        self.cm_y_true.append(actual);
        self.cm_y_pred.append(predicted);
        self.cm_button.setEnabled(True);
        self.set_buttons_enabled(True)

    def run_sort(self):
        if not self.model or not self.le: return
        self.log_text.clear();
        self.cm_y_true, self.cm_y_pred = [], [];
        self.progress_bar.setValue(0);
        self.progress_bar.setVisible(True)
        self.set_buttons_enabled(False);
        self.display_stack.setCurrentWidget(self.image_label);
        self.cm_button.setText("Show Confusion Matrix")
        self.sorter_worker = SorterWorker(self.model, self.le)
        self.sorter_worker.signals.progress.connect(self.update_progress);
        self.sorter_worker.signals.log.connect(self.log_message)
        self.sorter_worker.signals.display_image.connect(self.display_image);
        self.sorter_worker.signals.finished.connect(self.sorting_finished)
        self.sorter_worker.signals.error.connect(self.sorting_error)
        self.threadpool.start(self.sorter_worker)

    def sorting_finished(self, error_rate, y_true, y_pred):
        self.progress_bar.setVisible(False);
        self.set_buttons_enabled(True);
        self.cm_y_true = y_true;
        self.cm_y_pred = y_pred
        self.cm_button.setEnabled(True);
        final_message = f"<b>Sorting Complete. Final Percent Identification Error: {error_rate:.2f}%</b>"
        self.log_text.append(final_message);
        QMessageBox.information(self, "Sorting Complete", final_message)

    def toggle_cm_view(self):
        if not self.cm_y_true: QMessageBox.information(self, "No Data",
                                                       "Please inspect at least one batch to generate data for the matrix."); return
        if self.display_stack.currentWidget() == self.image_label:
            self.plot_confusion_matrix();
            self.display_stack.setCurrentWidget(self.cm_canvas);
            self.cm_button.setText("Show Image")
        else:
            self.display_stack.setCurrentWidget(self.image_label); self.cm_button.setText("Show Confusion Matrix")

    def plot_confusion_matrix(self):
        if not self.cm_y_true or not self.le: return
        labels = self.le.classes_;
        cm = confusion_matrix(self.cm_y_true, self.cm_y_pred, labels=labels)
        fig = self.cm_canvas.figure;
        fig.clear();
        ax = fig.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
        ax.set_title('Confusion Matrix', fontsize=14);
        ax.set_ylabel('Actual Class', fontsize=12);
        ax.set_xlabel('Predicted Class', fontsize=12)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor");
        fig.tight_layout();
        self.cm_canvas.draw()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def log_message(self, text, color):
        self.log_text.setTextColor(color); self.log_text.append(text); self.log_text.moveCursor(
            QTextCursor.MoveOperation.End)

    def display_image(self, q_img):
        self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio))

    def sorting_error(self, error_message):
        self.progress_bar.setVisible(False);
        self.set_buttons_enabled(True);
        self.log_text.append(f"<font color='red'><b>CRITICAL ERROR:</b> {error_message}</font>")
        QMessageBox.critical(self, "Sorting Error", f"A critical error occurred: {error_message}")

    def set_buttons_enabled(self, enabled):
        is_model_loaded = self.model is not None
        self.load_model_button.setEnabled(enabled);
        self.inspect_batch_button.setEnabled(enabled and is_model_loaded);
        self.run_sort_button.setEnabled(enabled and is_model_loaded)

    def closeEvent(self, event):
        if hasattr(self, 'sorter_worker'): self.sorter_worker.stop()
        self.threadpool.waitForDone();
        event.accept()


# --- Standalone Execution Guard ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CobberSorterApp()
    window.show()
    sys.exit(app.exec())