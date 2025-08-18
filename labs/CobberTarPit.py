# CobberTarPit.py
# A PyQt6 application for exploring Q-learning by manually and automatically training an agent.
# Refactored for the CobberLearnChem launcher.

import sys
import numpy as np
import os
import random
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QMessageBox, QFrame
)
from PyQt6.QtGui import QFont, QPixmap, QColor
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer

# --- Add project root to path for robust imports ---
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    project_root = os.path.abspath('.')


class AutoTrainerWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    q_table_updated = pyqtSignal(np.ndarray)
    animation_step = pyqtSignal(int)
    episode_finished = pyqtSignal(bool)
    update_reward = pyqtSignal(float)

    def __init__(self, q_table, rewards, gamma, start_pos, goal_pos, trap_pos,
                 max_steps=100, epsilon=0.1):
        super().__init__()
        self.q_table = q_table.copy()
        self.rewards = rewards
        self.gamma = gamma
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.trap_pos = trap_pos
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.num_actions = 4
        self._is_running = True

    def run(self):
        episode_count = 0
        while self._is_running:
            episode_count += 1
            state = self.start_pos

            # --- NEW: Reset and update reward for the new episode ---
            episode_reward = 0.0
            self.update_reward.emit(episode_reward)

            self.animation_step.emit(state)
            QThread.msleep(100)

            for _ in range(self.max_steps):
                if not self._is_running: break

                if random.uniform(0, 1) < self.epsilon:
                    action = random.randint(0, self.num_actions - 1)
                else:
                    action = np.argmax(self.q_table[state, :])

                if action == 0:
                    s_prime = state - 4 if state > 3 else state
                elif action == 1:
                    s_prime = state + 4 if state < 12 else state
                elif action == 2:
                    s_prime = state - 1 if state % 4 != 0 else state
                else:
                    s_prime = state + 1 if state % 4 != 3 else state

                reward = self.rewards[s_prime]

                # --- NEW: Update and emit live reward ---
                episode_reward += reward
                self.update_reward.emit(episode_reward)

                self.animation_step.emit(s_prime)
                QThread.msleep(100)

                next_max = np.max(self.q_table[s_prime, :])
                new_value = reward + self.gamma * next_max
                self.q_table[state, action] = new_value

                state = s_prime
                if state == self.goal_pos or state == self.trap_pos:
                    break

            if not self._is_running: break

            self.progress.emit(episode_count)
            self.q_table_updated.emit(self.q_table.copy())
            self.episode_finished.emit(state == self.goal_pos)
            QThread.msleep(500)

        self.finished.emit()

    def stop(self):
        self._is_running = False


class CobberTarPitApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")
        self.setWindowTitle("CobberTarPit")
        self.setGeometry(100, 100, 650, 725)
        self.setFont(self.lato_font)
        self.num_states = 16
        self.num_actions = 4
        self.gamma = 0.9
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.cumulative_reward = 0.0
        self.grid_is_hidden = True
        self.start_pos = 12
        self.goal_pos = 3
        self.trap_pos = 7
        self.agent_pos = self.start_pos
        self.goal_image = os.path.join(project_root, "assets", "pure_product.png")
        self.trap_images = [os.path.join(project_root, "assets", f"tar_pit{i}.png") for i in [1, 2, 3]]
        self.rewards = np.full(self.num_states, -1.0)
        self.rewards[self.goal_pos] = 100.0
        self.rewards[self.trap_pos] = -100.0
        self.last_state = -1
        self.last_action = -1
        self.grid_labels = []
        self.q_value_labels = {}
        self.worker = None
        self.thread = None
        self._setup_ui()
        self.full_reset()

    # Add this new method to the CobberTarPitApp class
    def _on_reward_updated(self, reward_value):
        self.reward_label.setText(f"{reward_value:.2f}")

    def _setup_ui(self):
        main_widget = QWidget();
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        grid_groupbox = QGroupBox("The Lab Bench");
        grid_layout = QGridLayout();
        grid_layout.setSpacing(5)
        for i in range(16):
            label = QLabel();
            label.setFixedSize(80, 80);
            label.setAlignment(Qt.AlignmentFlag.AlignCenter);
            label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            self.grid_labels.append(label);
            grid_layout.addWidget(label, i // 4, i % 4)
        grid_groupbox.setLayout(grid_layout);
        main_layout.addWidget(grid_groupbox)
        right_panel_layout = QVBoxLayout()
        q_table_groupbox = QGroupBox("The Agent's Brain (Q-Table)");
        q_table_layout = QGridLayout();
        q_table_layout.setSpacing(1)
        actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        q_table_layout.addWidget(QLabel(""), 0, 0)
        for c, a in enumerate(actions):
            h = QLabel(a);
            h.setFont(QFont("Arial", 10, QFont.Weight.Bold));
            h.setAlignment(Qt.AlignmentFlag.AlignCenter);
            q_table_layout.addWidget(h, 0, c + 1)
        for r in range(16):
            sh = QLabel(f"S{r}");
            sh.setFont(QFont("Arial", 10, QFont.Weight.Bold));
            sh.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter);
            q_table_layout.addWidget(sh, r + 1, 0)
            self.q_value_labels[r] = []
            for c in range(4):
                ql = QLabel("0.00");
                ql.setAlignment(Qt.AlignmentFlag.AlignCenter);
                ql.setStyleSheet("border:1px solid #ccc;background-color:white;");
                ql.setFixedSize(60, 19)
                self.q_value_labels[r].append(ql);
                q_table_layout.addWidget(ql, r + 1, c + 1)
        q_table_groupbox.setLayout(q_table_layout);
        right_panel_layout.addWidget(q_table_groupbox)
        controls_groupbox = QGroupBox("Controls");
        controls_layout = QVBoxLayout(controls_groupbox)

        # --- NEW: Create a dedicated status label ---
        self.status_label = QLabel("Select an action to begin.")
        self.status_label.setWordWrap(True)  # Allows text to wrap to a new line
        self.status_label.setStyleSheet("color: #333; font-style: italic; padding-bottom: 5px;")
        controls_layout.addWidget(self.status_label)

        # --- NEW: Add a separator line for visual clarity ---
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(separator)

        reward_layout = QHBoxLayout();
        reward_layout.addWidget(QLabel("<b>Current Episode Reward:</b>"))
        self.reward_label = QLabel("0.0");
        self.reward_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        reward_layout.addWidget(self.reward_label);
        reward_layout.addStretch();
        controls_layout.addLayout(reward_layout)
        action_layout = QHBoxLayout();
        action_layout.addWidget(QLabel("Choose Action:"))
        self.action_combo = QComboBox();
        self.action_combo.addItems(actions);
        action_layout.addWidget(self.action_combo)
        controls_layout.addLayout(action_layout)

        self.execute_button = QPushButton("Execute Action");
        self.update_q_button = QPushButton("Update Q-value");
        self.reset_button = QPushButton("Reset Episode")
        self.full_reset_button = QPushButton("Reset All & Clear Brain");
        self.reveal_button = QPushButton("Reveal Lab Bench");
        self.run_animation_button = QPushButton("Run Animation");
        self.stop_animation_button = QPushButton("Stop Animation")
        self.stop_animation_button.setEnabled(False)
        animation_control_layout = QHBoxLayout()
        animation_control_layout.addWidget(self.run_animation_button)
        animation_control_layout.addWidget(self.stop_animation_button)

        self.execute_button.clicked.connect(self.execute_action);
        self.update_q_button.clicked.connect(self.update_q_value)
        self.reveal_button.clicked.connect(self.reveal_grid);
        self.reset_button.clicked.connect(self.reset_episode)
        self.full_reset_button.clicked.connect(self.full_reset);
        self.run_animation_button.clicked.connect(self.start_animation)
        self.stop_animation_button.clicked.connect(self.stop_animation)

        # --- New Code with Separators ---
        # Manual Action Buttons
        controls_layout.addWidget(self.execute_button)
        controls_layout.addWidget(self.update_q_button)

        # Episode Control Buttons
        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.reveal_button)  # Moved here

        # --- First Separator Line ---
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(separator1)

        # Animation Buttons
        controls_layout.addLayout(animation_control_layout)

        # --- Second Separator Line ---
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(separator2)

        # Full Reset Button
        controls_layout.addWidget(self.full_reset_button)

        right_panel_layout.addWidget(controls_groupbox);
        main_layout.addLayout(right_panel_layout)
        self.status_label.setText("Select an action to begin.")

    def full_reset(self):
        self.stop_animation()  # Ensure any running animation is stopped
        self.q_table.fill(0);
        self.update_q_table_ui();
        self.reset_episode()
        self.status_label.setText("New session started. Agent's brain cleared.")

    def reset_episode(self):
        self.grid_is_hidden = True;
        self.agent_pos = self.start_pos;
        self.cumulative_reward = 0.0
        self.update_grid_ui();
        self.reward_label.setText(f"{self.cumulative_reward:.2f}")
        self.set_controls_enabled(True)
        self.update_q_button.setEnabled(False);
        self.reveal_button.setEnabled(False)
        self.status_label.setText("New episode started. Explore the hidden environment.")

    def execute_action(self):
        s = self.agent_pos;
        action_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3};
        a = action_map[self.action_combo.currentText()]
        if a == 0:
            s_prime = s - 4 if s > 3 else s
        elif a == 1:
            s_prime = s + 4 if s < 12 else s
        elif a == 2:
            s_prime = s - 1 if s % 4 != 0 else s
        else:
            s_prime = s + 1 if s % 4 != 3 else s
        self.agent_pos = s_prime;
        R = self.rewards[s_prime];
        self.cumulative_reward += R;
        self.reward_label.setText(f"{self.cumulative_reward:.2f}")
        self.last_state, self.last_action, self.s_prime, self.reward = s, a, s_prime, R
        self.update_grid_ui();
        max_q_s_prime = np.max(self.q_table[s_prime])
        status_text = f"Action in S{s} -> Reward={R:.1f}, New State=S{s_prime}. Formula: Q = {R:.1f} + {self.gamma} * max(Q(S{s_prime}))"
        self.status_label.setText(status_text);
        self.action_combo.setEnabled(False);
        self.execute_button.setEnabled(False);
        self.update_q_button.setEnabled(True)

    def update_q_value(self):
        s, a, s_prime, R = self.last_state, self.last_action, self.s_prime, self.reward
        new_q_value = R + self.gamma * np.max(self.q_table[s_prime, :])
        self.q_table[s, a] = new_q_value;
        self.update_q_table_ui()
        self.status_label.setText(
            f"Q(S{s}, Action: {self.action_combo.currentText()}) updated to {new_q_value:.2f}. Choose next action.")
        is_terminal = s_prime == self.goal_pos or s_prime == self.trap_pos
        if is_terminal:
            self.action_combo.setEnabled(False);
            self.execute_button.setEnabled(False);
            self.update_q_button.setEnabled(False);
            self.reveal_button.setEnabled(True)
            if s_prime == self.goal_pos:
                self.show_popup("Success!", "You have synthesized the pure product!", self.goal_image)
            else:
                self.show_popup("Failure!", "Oh no! You made sludge ... a chemical tar pit!",
                                random.choice(self.trap_images))
        else:
            self.action_combo.setEnabled(True);
            self.execute_button.setEnabled(True);
            self.update_q_button.setEnabled(False)

    def start_animation(self):
        self.set_controls_enabled(False)
        self.status_label.setText("Starting continuous animation...")
        self.thread = QThread();
        self.worker = AutoTrainerWorker(q_table=self.q_table, rewards=self.rewards, gamma=self.gamma,
                                        start_pos=self.start_pos, goal_pos=self.goal_pos, trap_pos=self.trap_pos)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit);
        self.worker.finished.connect(self.worker.deleteLater);
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.finished.connect(self._on_animation_finished)
        self.worker.progress.connect(self._on_auto_train_progress)
        self.worker.q_table_updated.connect(self._on_q_table_updated)
        self.worker.animation_step.connect(self._on_animation_step)
        self.worker.episode_finished.connect(self._on_episode_finished)
        self.worker.update_reward.connect(self._on_reward_updated)
        self.thread.start()

    def stop_animation(self):
        if self.worker: self.worker.stop()
        self.status_label.setText("Stopping animation...")
        # --- FIX: Immediately update button state for responsiveness ---
        self.set_controls_enabled(True)

    def _on_auto_train_progress(self, episode_num):
        self.status_label.setText(f"Running... Episode: {episode_num}")

    def _on_q_table_updated(self, new_q_table):
        self.q_table = new_q_table;
        self.update_q_table_ui()

    def _on_animation_step(self, position):
        self.agent_pos = position;
        self.update_grid_ui()

    def _on_episode_finished(self, is_success):
        if is_success:
            self.show_popup("Success!", "The agent has found the pure product!", self.goal_image, is_animated=True)
        else:
            self.show_popup("Failure!", "The agent fell into the tar pit!", random.choice(self.trap_images),
                            is_animated=True)
        # --- FIX: Don't call reset_episode from here. The worker handles the loop. ---

    def _on_animation_finished(self):
        self.status_label.setText("Animation stopped.")
        self.set_controls_enabled(True)
        self.reward_label.setText("0.00")

    def set_controls_enabled(self, enabled):
        self.execute_button.setEnabled(enabled);
        self.update_q_button.setEnabled(enabled);
        self.reset_button.setEnabled(enabled)
        self.full_reset_button.setEnabled(enabled);
        self.reveal_button.setEnabled(enabled)
        self.run_animation_button.setEnabled(enabled)
        self.stop_animation_button.setEnabled(not enabled)
        self.action_combo.setEnabled(enabled)

    def reveal_grid(self):
        self.grid_is_hidden = False;
        self.update_grid_ui();
        self.reveal_button.setEnabled(False)

    # --- MODIFIED to handle both blocking and auto-closing pop-ups ---
    def show_popup(self, title, message, image_path, is_animated=False):
        msg_box = QMessageBox(self);
        msg_box.setWindowTitle(title);
        msg_box.setText(f"<h3>{message}</h3>")

        if is_animated:
            msg_box.setInformativeText("The agent will now start the next episode.")
            icon_size = 200
        else:
            msg_box.setInformativeText("Click 'Reveal Lab Bench' to see the layout, or 'Reset Episode' to start again.")
            icon_size = 500

        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            msg_box.setIconPixmap(pixmap.scaled(icon_size, icon_size, Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))
        else:
            print(f"Warning: Image not found at {image_path}")

        if is_animated:
            QTimer.singleShot(500, msg_box.close)
            msg_box.show()  # Use show() for non-blocking
        else:
            msg_box.exec()  # Use exec() for blocking

    def update_grid_ui(self):
        for i, label in enumerate(self.grid_labels):
            label.setText(f"S{i}");
            style = "border: 2px solid #555; border-radius: 5px;"
            if self.grid_is_hidden:
                style += "background-color: #ddd; color: #888;"
            else:
                if i == self.start_pos:
                    style += "background-color: #aaffaa;"; label.setText("S")
                elif i == self.goal_pos:
                    style += "background-color: #aaddff;"; label.setText("G")
                elif i == self.trap_pos:
                    style += "background-color: #444; color: white;"; label.setText("X")
                else:
                    style += "background-color: #f0f0f0;"; label.setText("")
            if i == self.agent_pos: style += "border: 4px solid #ff5555;"
            label.setStyleSheet(style)

    def update_q_table_ui(self):
        for r in range(self.num_states):
            for c in range(self.num_actions):
                q_val = self.q_table[r, c];
                label = self.q_value_labels[r][c];
                label.setText(f"{q_val:.2f}")
                if q_val > 0.01:
                    label.setStyleSheet(
                        f"background-color: rgba(100, 100, 255, {int(min(255, q_val * 3))}); border: 1px solid #ccc;")
                elif q_val < -0.01:
                    label.setStyleSheet(
                        f"background-color: rgba(255, 100, 100, {int(min(255, abs(q_val * 3)))}); border: 1px solid #ccc;")
                else:
                    label.setStyleSheet("background-color: white; border: 1px solid #ccc;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CobberTarPitApp()
    window.show()
    sys.exit(app.exec())