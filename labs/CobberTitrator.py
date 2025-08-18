# CobberTitrator.py
# A PyQt6 application for training a reinforcement learning agent to perform titrations.
# Refactored for the CobberLearnChem launcher.

import sys
import numpy as np
import gym
from gym import spaces
import random
import tensorflow as tf
from tensorflow.keras import layers
from collections import deque
import matplotlib.pyplot as plt
import time
import os

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QFileDialog, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Add project root to path for robust imports ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('.')


# ----------------------------
# Environment Definition (LOGIC UNCHANGED)
# ----------------------------
class RobotTrackEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(RobotTrackEnv, self).__init__()
        self.max_velocity = 1.25;
        self.min_velocity = 0.0;
        self.max_acceleration = 0.5;
        self.min_acceleration = -0.5;
        self.dt = 0.1
        self.acceleration_values = [-0.5, 0.0, 0.5]
        self.action_space = spaces.Discrete(len(self.acceleration_values))
        low_obs = np.array([0.0, self.min_velocity, 0.0, 9.5], dtype=np.float32)
        high_obs = np.array([np.inf, self.max_velocity, 1.0, 24.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.reset()

    def reset(self):
        if not getattr(self, 'equivalence_point_set', False):
            self.d_true = np.random.uniform(10.0, 24.0)
            self.d_estimated = self.d_true + np.random.uniform(-0.5, 0.5)
        else:
            self.equivalence_point_set = False
        self.position = 0.0;
        self.velocity = 0.0;
        self.done = False;
        self.total_time = 0.0;
        self.warning = 0.0
        return self._get_state()

    def step(self, action):
        acceleration = self.acceleration_values[action]
        self.velocity = np.clip(self.velocity + acceleration * self.dt, self.min_velocity, self.max_velocity)
        self.position += self.velocity * self.dt
        self.total_time += self.dt;
        self.warning = self._calculate_warning();
        reward = self.velocity * self.dt - 0.1
        if self.velocity == 0.0 and self.warning == 0.0 and self.position < self.d_true - 0.5:
            self.done = True;
            reward += -50.0;
            return self._get_state(), reward, self.done, {}
        if self.d_true - 0.1 <= self.position < self.d_true:
            self.done = True;
            error = 2 * (self.position - self.d_true)
            reward += 100.0 / self.total_time + 225.0 * np.exp(200.0 * error)
        elif self.position > self.d_true:
            error = 2 * (self.position - self.d_true)
            self.done = True; reward += -1 * np.exp(100.0 * error) +1
        return self._get_state(), reward, self.done, {}

    def _calculate_warning(self):
        x, d_est = self.position, self.d_estimated
        if x >= d_est - 0.40: w = 1 - np.exp(-20 * (x - d_est + 0.4) ** 2); return np.clip(w, 0.0, 1.0)
        return 0.0

    def _get_state(self):
        return np.array([self.position, self.velocity, self.warning, self.d_estimated], dtype=np.float32)

    def set_equivalence_point(self, d_true):
        self.d_true = d_true;
        self.d_estimated = self.d_true + np.random.uniform(-0.5, 0.5);
        self.equivalence_point_set = True


# ----------------------------
# DQN & Replay Memory (LOGIC UNCHANGED)
# ----------------------------
class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu', input_shape=(state_size,));
        self.dense2 = layers.Dense(64, activation='relu');
        self.output_layer = layers.Dense(action_size)

    def call(self, x): x = self.dense1(x); x = self.dense2(x); return self.output_layer(x)


class ReplayMemory:
    def __init__(self, capacity): self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done): self.memory.append(
        (state, action, reward, next_state, done))

    def sample(self, batch_size): return random.sample(self.memory, batch_size)

    def __len__(self): return len(self.memory)


# ----------------------------
# Worker Threads (LOGIC UNCHANGED)
# ----------------------------
class TrainingThread(QThread):
    update_log = pyqtSignal(str);
    update_animation = pyqtSignal(dict);
    training_finished = pyqtSignal()

    def __init__(self, num_episodes):
        super().__init__();
        self.num_episodes = num_episodes;
        self.env = RobotTrackEnv();
        self.model = None;
        self.target_model = None
        self.memory = ReplayMemory(10000);
        self.gamma = 0.99;
        self.epsilon_start = 1.0;
        self.epsilon_end = 0.01
        self.epsilon_decay = 1500;
        self.batch_size = 64;
        self.target_update = 10;
        self.steps_done = 0

    # Add this entire new method inside the EvaluationTab class

    def _format_inputs(self):
        """Formats the numbers in the input boxes to the correct precision."""
        try:
            # Format concentrations to 4 decimal places
            conc_hcl = float(self.conc_hcl_input.text())
            self.conc_hcl_input.setText(f"{conc_hcl:.4f}")

            conc_naoh = float(self.conc_naoh_input.text())
            self.conc_naoh_input.setText(f"{conc_naoh:.4f}")

            # Format volume to 2 decimal places
            vol_hcl = float(self.volume_hcl_input.text())
            self.volume_hcl_input.setText(f"{vol_hcl:.2f}")
        except ValueError:
            # Don't do anything if the input is not a valid number
            pass

    def select_action(self, state, action_size):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1. * self.steps_done / self.epsilon_decay);
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(action_size)
        else:
            q_values = self.model(np.array([state], dtype=np.float32)); return np.argmax(q_values[0])

    def run(self):
        try:
            state_size = self.env.observation_space.shape[0];
            action_size = self.env.action_space.n
            self.model = DQN(state_size, action_size);
            self.target_model = DQN(state_size, action_size)
            dummy_input = tf.constant([[0.0] * state_size], dtype=tf.float32);
            self.model(dummy_input);
            self.target_model(dummy_input)
            self.target_model.set_weights(self.model.get_weights());
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            for episode in range(1, self.num_episodes + 1):
                state = self.env.reset();
                total_reward, t = 0, 0;
                positions = [self.env.position]
                for t in range(1, 1501):
                    action = self.select_action(state, action_size);
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward;
                    self.memory.push(state, action, reward, next_state, done);
                    state = next_state;
                    positions.append(self.env.position)
                    if len(self.memory) >= self.batch_size:
                        transitions = self.memory.sample(self.batch_size);
                        states, actions, rewards, next_states, dones = map(np.array, zip(*transitions))
                        next_q_values = self.target_model(next_states.astype(np.float32));
                        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
                        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
                        with tf.GradientTape() as tape:
                            q_values = self.model(states.astype(np.float32));
                            action_masks = tf.one_hot(actions.astype(np.int32), action_size)
                            q_values = tf.reduce_sum(q_values * action_masks, axis=1);
                            loss = tf.keras.losses.MSE(target_q_values, q_values)
                        grads = tape.gradient(loss, self.model.trainable_variables);
                        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                    if done: break
                if episode % self.target_update == 0: self.target_model.set_weights(self.model.get_weights())
                if episode % 5 == 0:
                    log_message = (
                        f"Ep {episode}: Reward={total_reward:.2f}, Steps={t}, True Endpoint={self.env.d_true:.2f}, Stop={self.env.position:.2f}")
                    self.update_log.emit(log_message);
                    self.update_animation.emit({'positions': positions.copy(), 'd_true': self.env.d_true})
            self.update_log.emit("Training complete. Model is ready to be saved.")
        except Exception as e:
            self.update_log.emit(f"Error during training: {e}")
        finally:
            self.training_finished.emit()


class EvaluationThread(QThread):
    update_log = pyqtSignal(str);
    update_animation = pyqtSignal(list, float);
    update_results = pyqtSignal(str);
    evaluation_finished = pyqtSignal()

    def __init__(self, model, d_true, conc_naoh, volume_hcl):
        super().__init__();
        self.model = model;
        self.conc_naoh = conc_naoh;
        self.volume_hcl = volume_hcl
        self.env = RobotTrackEnv();
        self.env.set_equivalence_point(d_true)

    # Replace the entire run method in the EvaluationThread class with this one

    def run(self):
        try:
            self.update_log.emit(
                f"Starting Eval: True End Point={self.env.d_true:.2f}, Estimated End Point={self.env.d_estimated:.2f}")

            header = f"{'Run':<8}{'Stop Position (mL)':<25}{'Error (mL)':<20}"
            self.update_log.emit(header)
            self.update_log.emit("-" * 53)

            # --- MODIFIED: Create lists to store results from the visible runs ---
            final_errors = []
            final_positions = []

            for run in range(1, 4):
                final_pos, final_error, total_reward, t = self.run_single_titration()

                # --- MODIFIED: Store both error and position ---
                final_errors.append(final_error)
                final_positions.append(final_pos)

                log_message = f"{run:<8}{final_pos:<25.2f}{final_error:<20.2f}"
                self.update_log.emit(log_message)

                time.sleep(2)  # Keep the pause for visual effect

            if np.std(final_errors) >= 0.1:
                self.update_log.emit("Runs not consistent. Performing a fourth run.")
                final_pos, final_error, _, _ = self.run_single_titration()

                # --- MODIFIED: Store the fourth run's results if it happens ---
                final_errors.append(final_error)
                final_positions.append(final_pos)

                # Use the same column formatting for the 4th run log
                log_message = f"{4:<8}{final_pos:<25.2f}{final_error:<20.2f}"
                self.update_log.emit(log_message)

            # --- CORRECTED: Calculate average from the stored, visible results ---
            avg_titrant_vol = np.mean(final_positions)

            conc_hcl_final = (self.conc_naoh * avg_titrant_vol) / self.volume_hcl
            self.update_results.emit(
                f"""
                <p style="font-family: Lato; font-size: 12pt;">
                    Average Titrant Volume: <b>{avg_titrant_vol:.2f} mL</b>
                </p>
                <p style="font-family: Lato; font-size: 11pt;">
                    Final Concentration of HCl: <b>{conc_hcl_final:.4f} M</b>
                </p>
                """
            )
        except Exception as e:
            self.update_log.emit(f"Error during evaluation: {e}")
        finally:
            self.evaluation_finished.emit()



    def run_single_titration(self):
        self.env.set_equivalence_point(self.env.d_true);
        state = self.env.reset();
        positions = [self.env.position];
        total_reward, t = 0, 0
        for t in range(1, 1501):
            state = np.array([state], dtype=np.float32);
            q_values = self.model(state);
            action = np.argmax(q_values[0])
            next_state, reward, done, _ = self.env.step(action);
            state = next_state;
            total_reward += reward;
            positions.append(self.env.position)
            if done: break
        self.update_animation.emit(positions, self.env.d_true)
        return self.env.position, (self.env.position - self.env.d_true), total_reward, t


# ----------------------------
# GUI Tabs (Styling Added)
# ----------------------------
class TrainingTab(QWidget):
    # In the TrainingTab Class

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        layout = QVBoxLayout(self)
        controls_layout = QHBoxLayout()

        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 100000)
        self.episodes_input.setValue(50)
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)

        controls_layout.addWidget(QLabel("Number of Episodes:"))
        controls_layout.addWidget(self.episodes_input)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.save_button)

        # --- NEW: Create a horizontal layout for the display area ---
        display_layout = QHBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # --- NEW: Create a label to hold the chemist image ---
        self.chemist_label = QLabel()
        self.chemist_label.setFixedSize(250, 250)
        self.chemist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- NEW: Load the pixmaps for the chemists ---
        self.approving_pixmap = QPixmap(os.path.join(project_root, "assets", "approvingchemist.jpg"))
        self.disapproving_pixmap = QPixmap(os.path.join(project_root, "assets", "disapprovingchemist.jpg"))

        # --- NEW: Add canvas and label to the horizontal layout ---
        display_layout.addWidget(self.canvas)
        display_layout.addWidget(self.chemist_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("lato", 11))

        layout.addLayout(controls_layout)
        layout.addWidget(QLabel("Training Animation (updated every 5 episodes):"))
        layout.addLayout(display_layout)  # Add the new horizontal layout
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.log)

    def start_training(self):
        self.start_button.setEnabled(False);
        self.save_button.setEnabled(False);
        self.log.append(f"Starting training for {self.episodes_input.value()} episodes...")
        self.chemist_label.clear()
        self.training_thread = TrainingThread(self.episodes_input.value());
        self.training_thread.update_log.connect(self.log.append)
        self.training_thread.update_animation.connect(self.update_animation);
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.start()

    def save_model(self):
        if self.training_thread and self.training_thread.model:
            assets_dir = os.path.join(project_root, "assets");
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Trained Model", assets_dir,
                                                      "H5 Weights (*.weights.h5)")
            if filepath:
                if not filepath.endswith(".weights.h5"): filepath += ".weights.h5"
                self.training_thread.model.save_weights(filepath);
                self.log.append(f"Model saved to {filepath}")
        else:
            QMessageBox.warning(self, "Save Model", "No trained model to save.")

    def update_animation(self, frame_data):
        self.figure.clear();
        ax = self.figure.add_subplot(111);
        positions = frame_data['positions'];
        d_true = frame_data['d_true']
        t = np.arange(len(positions)) * 0.1;
        #colors = np.exp(-200 * (np.array(positions) - d_true - 0.05)**2 );
        colors = (np.tanh(20 * (np.array(positions) - d_true + 0.002) ) + 1) / 2.0
        cmap = plt.cm.viridis_r
        sc = ax.scatter(t, positions, c=colors, cmap=cmap, marker='s', s=100, vmin=0.0, vmax=1.0)
        ax.set_xlim(0, max(5, t.max() * 1.1));
        ax.set_ylim(0, d_true * 1.1)
        ax.set_xlabel('Time (s)');
        ax.set_ylabel('Titrant Added (mL)');
        cbar = self.figure.colorbar(sc, ax=ax);
        cbar.set_label('Indicator Color');
        cbar.set_ticks([]);
        self.canvas.draw()

        # --- NEW: Add logic to display chemist image ---
        final_pos = frame_data['positions'][-1]
        ep = frame_data['d_true']
        if ep - 0.075 <= final_pos <= ep + 0.01:
            pixmap = self.approving_pixmap
        else:
            pixmap = self.disapproving_pixmap

        self.chemist_label.setPixmap(pixmap.scaled(250, 250,
                                                   Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation))

    def on_training_finished(self):
        self.start_button.setEnabled(True); self.save_button.setEnabled(True)


class EvaluationTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model = None
        layout = QVBoxLayout(self)
        inputs_layout = QHBoxLayout()

        self.conc_hcl_input = QLineEdit("0.1")
        self.volume_hcl_input = QLineEdit("20")
        self.conc_naoh_input = QLineEdit("0.1")

        #self.conc_hcl_input.editingFinished.connect(self._format_inputs)
        #self.volume_hcl_input.editingFinished.connect(self._format_inputs)
        #self.conc_naoh_input.editingFinished.connect(self._format_inputs)


        inputs_layout.addWidget(QLabel("Conc. HCl (M):"));
        inputs_layout.addWidget(self.conc_hcl_input)
        inputs_layout.addWidget(QLabel("Volume HCl (mL):"));
        inputs_layout.addWidget(self.volume_hcl_input)
        inputs_layout.addWidget(QLabel("Conc. NaOH (M):"));
        inputs_layout.addWidget(self.conc_naoh_input)

        buttons_layout = QHBoxLayout()
        self.load_model_button = QPushButton("Load Model")
        self.sacrificial_run_button = QPushButton("Sacrificial Run")
        self.perform_titration_button = QPushButton("Perform Titration")

        self.load_model_button.clicked.connect(self.load_model)
        self.sacrificial_run_button.clicked.connect(self.sacrificial_run)
        self.sacrificial_run_button.setEnabled(False)
        self.perform_titration_button.clicked.connect(self.perform_titration)
        self.perform_titration_button.setEnabled(False)

        buttons_layout.addWidget(self.load_model_button)
        buttons_layout.addWidget(self.sacrificial_run_button)
        buttons_layout.addWidget(self.perform_titration_button)

        # --- NEW: Create horizontal layout for display ---
        display_layout = QHBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        # --- NEW: Create label and load chemist images ---
        self.chemist_label = QLabel()
        self.chemist_label.setFixedSize(250, 250)
        self.chemist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.approving_pixmap = QPixmap(os.path.join(project_root, "assets", "approvingchemist.jpg"))
        self.disapproving_pixmap = QPixmap(os.path.join(project_root, "assets", "disapprovingchemist.jpg"))

        display_layout.addWidget(self.canvas)
        display_layout.addWidget(self.chemist_label)

        self.results_display = QTextEdit();
        self.results_display.setReadOnly(True)
        self.log = QTextEdit();
        self.log.setReadOnly(True)
        self.log.setFont(QFont("lato", 11))

        layout.addLayout(inputs_layout)
        layout.addLayout(buttons_layout)
        layout.addWidget(QLabel("Titration Animation:"))
        layout.addLayout(display_layout)  # Add new horizontal layout
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_display)
        layout.addWidget(QLabel("Evaluation Log:"))
        layout.addWidget(self.log)

    def _format_inputs(self):
        """Formats the numbers in the input boxes to the correct precision."""
        try:
            # Format concentrations to 4 decimal places
            conc_hcl = float(self.conc_hcl_input.text())
            self.conc_hcl_input.setText(f"{conc_hcl:.4f}")

            conc_naoh = float(self.conc_naoh_input.text())
            self.conc_naoh_input.setText(f"{conc_naoh:.4f}")

            # Format volume to 2 decimal places
            vol_hcl = float(self.volume_hcl_input.text())
            self.volume_hcl_input.setText(f"{vol_hcl:.2f}")
        except ValueError:
            # Don't do anything if the input is not a valid number
            pass

    def load_model(self):
        assets_dir = os.path.join(project_root, "assets");
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Trained Model", assets_dir, "H5 Weights (*.weights.h5)")
        if filepath:
            try:
                state_size = 4;
                action_size = 3;
                self.model = DQN(state_size, action_size);
                self.model(tf.constant([[0.0] * state_size], dtype=tf.float32));
                self.model.load_weights(filepath)
                self.log.append(f"Model loaded from {os.path.basename(filepath)}");
                self.sacrificial_run_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Load Model Error", f"Error: {e}")

    def sacrificial_run(self):


        self._format_inputs()

        try:
            conc_hcl = float(self.conc_hcl_input.text());
            volume_hcl = float(self.volume_hcl_input.text());
            conc_naoh = float(self.conc_naoh_input.text())
            if not (conc_hcl > 0 and volume_hcl > 0 and conc_naoh > 0): raise ValueError("Inputs must be positive.")
            self.d_true = (conc_hcl * volume_hcl) / conc_naoh
            self.log.append(
                f"Sacrificial Run Complete. True End Point calculated: {self.d_true:.2f} mL");
            self.perform_titration_button.setEnabled(True)
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {e}")

    def perform_titration(self):
        if not self.model or not hasattr(self, 'd_true'): QMessageBox.warning(self, "Setup Error",
                                                                              "Load a model and perform a sacrificial run first."); return
        self.chemist_label.clear()
        self.perform_titration_button.setEnabled(False);
        self.results_display.clear()
        conc_naoh = float(self.conc_naoh_input.text());
        volume_hcl = float(self.volume_hcl_input.text())
        self.evaluation_thread = EvaluationThread(self.model, self.d_true, conc_naoh, volume_hcl)
        self.evaluation_thread.update_log.connect(self.log.append);
        self.evaluation_thread.update_animation.connect(self.update_animation)
        self.evaluation_thread.update_results.connect(self.results_display.append);
        self.evaluation_thread.evaluation_finished.connect(lambda: self.perform_titration_button.setEnabled(True))
        self.evaluation_thread.start()

    def update_animation(self, positions, d_true):
        self.figure.clear();
        ax = self.figure.add_subplot(111);
        t = np.arange(len(positions)) * 0.1
        colors = np.exp(-200 * (np.array(positions) - d_true - 0.05) ** 2);
        cmap = plt.cm.viridis_r
        sc = ax.scatter(t, positions, c=colors, cmap=cmap, marker='s', s=100, vmin=0.0, vmax=1.0)
        ax.axhline(y=d_true, color='b', linestyle='--', label=f'Endpoint ({d_true:.2f} mL)')
        ax.set_xlim(0, max(5, t.max() * 1.1));
        ax.set_ylim(0, d_true * 1.1)
        ax.set_xlabel('Time (s)');
        ax.set_ylabel('Titrant Added (mL)');
        cbar = self.figure.colorbar(sc, ax=ax);
        cbar.set_label('Indicator Color');
        cbar.set_ticks([]);
        ax.legend();
        self.canvas.draw()

        final_pos = positions[-1]
        ep = d_true
        if ep - 0.075 <= final_pos <= ep + 0.01:
            pixmap = self.approving_pixmap
        else:
            pixmap = self.disapproving_pixmap

        self.chemist_label.setPixmap(pixmap.scaled(250, 250,
                                                   Qt.AspectRatioMode.KeepAspectRatio,
                                                   Qt.TransformationMode.SmoothTransformation))

# --- RENAMED Main Application Class ---
class CobberTitratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # --- BRANDING ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberTitrator")
        self.setGeometry(100, 100, 1300, 700)
        self.setFont(self.lato_font)

        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()
        # Pass self to tabs for branding colors
        self.training_tab = TrainingTab(self)
        self.evaluation_tab = EvaluationTab(self)
        self.tabs.addTab(self.training_tab, "Model Training")
        self.tabs.addTab(self.evaluation_tab, "Model Testing")

        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)


# --- Standalone Execution Guard ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberTitratorApp()  # Use new class name
    window.show()
    sys.exit(app.exec())