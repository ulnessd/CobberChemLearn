import tensorflow as tf

import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTreeWidget, QTreeWidgetItem, QLabel, QPushButton, QFrame,
    QSplitter, QScrollArea, QStackedWidget, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap, QColor, QIcon

from labs.CobberAlpha import CobberAlphaApp
from labs.CobberLand import CobberLandApp
from labs.CobberNeuron import CobberNeuronApp

# --- Import the refactored lab applications ---
try:
    from labs.CobberFetcher import CobberFetcherApp
except ImportError:
    print("Warning: Could not import CobberFetcherApp.");
    CobberFetcherApp = None

try:
    from labs.CobberCluster import CobberClusterApp
except ImportError:
    print("Warning: Could not import CobberClusterApp.");
    CobberClusterApp = None

try:
    from labs.CobberImpute import CobberImputeApp
except ImportError:
    print("Warning: Could not import CobberImputeApp.");
    CobberImputeApp = None

try:
    from labs.CobberResidue import CobberResidueApp
except ImportError:
    print("Warning: Could not import CobberResidueApp.");
    CobberResidueApp = None

# === NEW: Import CobberTree and its dataclass ===
try:
    from labs.CobberTree import CobberTreeApp, Hydrocarbon
except ImportError as e:
    print(f"Warning: Could not import CobberTreeApp. Error: {e}")
    CobberTreeApp = None
    Hydrocarbon = None

try:
    from labs.CobberK import CobberKApp
except ImportError as e:
    print(f"Warning: Could not import CobberKApp. Error: {e}")
    CobberKApp = None

try:
    from labs.CobberDescender import CobberDescenderApp
except ImportError as e:
    print(f"Warning: Could not import CobberDescenderApp. Error: {e}")
    CobberDescenderApp = None

try:
    from labs.CobberEvolve import CobberEvolveApp
except ImportError as e:
    print(f"Warning: Could not import CobberEvolveApp. Error: {e}")
    CobberEvolveApp = None

try:
    from labs.CobberNeuron import CobberNeuronApp
except ImportError as e:
    print(f"Warning: Could not import CobberNeuronApp. Error: {e}")
    CobberNeuronApp = None

try:
    from labs.CobberCNN import CobberCNNApp
except ImportError as e:
    print(f"Warning: Could not import CobberCNNApp. Error: {e}")
    CobberCNNApp = None

try:
    from labs.CobberTarPit import CobberTarPitApp
except ImportError as e:
    print(f"Warning: Could not import CobberTarPitApp. Error: {e}")
    CobberTarPitApp = None

try:
    from labs.CobberLand import CobberLandApp
except ImportError as e:
    print(f"Warning: Could not import CobberLandApp. Error: {e}")
    CobberLandApp = None

try:
    from labs.CobberHbond import CobberHbondApp
except ImportError as e:
    print(f"Warning: Could not import CobberHbondApp. Error: {e}")
    CobberHbondApp = None

try:
    from labs.CobberTitrator import CobberTitratorApp
except ImportError as e:
    print(f"Warning: Could not import CobberTitratorApp. Error: {e}")
    CobberTitratorApp = None

try:
    from labs.CobberSorter import CobberSorterApp
except ImportError as e:
    print(f"Warning: Could not import CobberSorterApp. Error: {e}")
    CobberSorterApp = None

try:
    from labs.CobberAlpha import CobberAlphaApp
except ImportError as e:
    print(f"Warning: Could not import CobberAlphaApp. Error: {e}")
    CobberAlphaApp = None

# =============================================================================
# DATA STRUCTURE - Complete and Unchanged from your working version
# =============================================================================
# =============================================================================
# DATA STRUCTURE (Updated with all completed labs)
# =============================================================================
LABS_DATA = [
    {
        "part": "Part II: The 'Data' Domain", "chapter": 4, "title": "Acquiring Chemical Data from Public Databases",
        "program_name": "CobberFetcher",
        "description": "This chapter will introduce you to two of the most important databases in the chemical and biological sciences: PubChem for small molecules and the Protein Data Bank (PDB) for large macromolecules. You will learn how to navigate these resources and use the CobberFetcher application to programmatically retrieve data.",
        "app_class": CobberFetcherApp
    },
    {
        "part": "Part II: The 'Data' Domain", "chapter": 5, "title": "Exploring Similarity and Parameter Space",
        "program_name": "CobberCluster",
        "description": "This activity explores the concept of 'similarity'. We will use the CobberCluster software to get an intuitive feeling for parameter space by comparing molecules using different criteria, acting as a 'human' clustering algorithm.",
        "app_class": CobberClusterApp
    },
    {
        "part": "Part II: The 'Data' Domain", "chapter": 6, "title": "An Introduction to Imputation",
        "program_name": "CobberImpute",
        "description": "This activity introduces data imputation, the process of intelligently filling in missing values. We will explore why simply ignoring missing data can be dangerous and use the CobberImpute application to compare several machine learning strategies for creating a complete, high-quality dataset.",
        "app_class": CobberImputeApp
    },
    {
        "part": "Part II: The 'Data' Domain", "chapter": 7, "title": "An Introduction to Error Analysis",
        "program_name": "CobberResidue",
        "description": "This activity introduces the fundamental concepts of model evaluation. We will use the CobberResidue application to analyze the pre-computed results from several hypothetical models, allowing us to focus entirely on the art and science of evaluation.",
        "app_class": CobberResidueApp
    },
    {
        "part": "Part III: The 'Models' Domain", "chapter": 8, "title": "An Introduction to Decision Trees",
        "program_name": "CobberTree",
        "description": "This activity introduces a fundamental and intuitive machine learning algorithm known as a decision tree. You will use the CobberTree application to become the 'human' decision tree algorithm, manually sorting a small dataset of molecules to discover the best splits for yourself.",
        "app_class": CobberTreeApp
    },
    {
        "part": "Part III: The 'Models' Domain", "chapter": 9, "title": "Similarity, Clustering, & Classification",
        "program_name": "CobberK",
        "description": "This activity introduces two foundational machine learning algorithms: K-Means Clustering and K-Nearest Neighbors (KNN). We will explore how these distance-based methods can be used to discover groups and classify elements within the periodic table.",
        "app_class": CobberKApp
    },
    {
        "part": "Part III: The 'Models' Domain", "chapter": 10, "title": "An Introduction to Gradient Descent",
        "program_name": "CobberDescender",
        "description": "This activity introduces gradient descent, the fundamental engine that powers modern machine learning. We will visually and interactively explore how a computer can 'descend' a complex error surface to find the optimal parameters for fitting a line to a set of noisy data points.",
        "app_class": CobberDescenderApp
    },
    {
        "part": "Part III: The 'Models' Domain", "chapter": 11, "title": "An Introduction to Evolutionary Algorithms",
        "program_name": "CobberEvolve",
        "description": "This chapter introduces a powerful class of optimization algorithms inspired by Darwinian evolution. We will use our chemoinformatics application, CobberEvolve, to discover a molecule with an optimal drug-binding energy through a process of simulated natural selection.",
        "app_class": CobberEvolveApp
    },
    {
        "part": "Part III: The 'Models' Domain", "chapter": 12,
        "title": "An Introduction to Artificial Neural Networks",
        "program_name": "CobberNeuron",
        "description": "This activity will demystify the 'black box' of AI by showing that a neural network is simply a collection of the basic components we have already studied. We will explore how a single neuron works and then see how a network of them can be trained to learn complex, non-linear chemical relationships.",
        "app_class": CobberNeuronApp
    },
    {
        "part": "Part III: The 'Models' Domain", "chapter": 13,
        "title": "Introduction to Convolutional Neural Networks",
        "program_name": "CobberCNN",
        "description": "This activity introduces the Convolutional Neural Network (CNN). We will build one by hand, visually exploring the mathematical operations that allow a computer to 'see' and recognize features like lines and junctions within an image of a molecule.",
        "app_class": CobberCNNApp
    },
    {
        "part": "Part III: The 'Models' Domain", "chapter": 14, "title": "An Introduction to Reinforcement Learning",
        "program_name": "CobberTarPit",
        "description": "In RL, there is no answer key. Instead, we create an agent that learns by taking actions and observing the consequences. Your task is to explore the 'Lab Bench' environment and manually build the agent’s 'brain'—the Q-table—one step at a time.",
        "app_class": CobberTarPitApp
    },
    {
        "part": "Part IV: The 'Results' Domain", "chapter": 15, "title": "Exploring Regression Models in Cobberland",
        "program_name": "CobberLand",
        "description": "In this lab, you will use the CobberLand application to predict a fictional chemical property, 'STALKiness'. By training and evaluating five different common regression models, you will learn how to assess their performance, interpret their results, and understand critical trade-offs.",
        "app_class": CobberLandApp
    },
    {
        "part": "Part IV: The 'Results' Domain", "chapter": 16, "title": "Discovering Hydrogen Bonding",
        "program_name": "CobberHbond",
        "description": "We will utilize unsupervised machine learning to discover the existence and nature of hydrogen bonding. We will use a clustering algorithm to find natural groups within the data, effectively having the machine 'discover' this fundamental chemical principle.",
        "app_class": CobberHbondApp
    },
    {
        "part": "Part IV: The 'Results' Domain", "chapter": 17, "title": "Training a Titrating Robot",
        "program_name": "CobberTitrator",
        "description": "This lab explores Reinforcement Learning (RL). We will train a simulated robot agent—the 'CobberTitrator'—to perform titrations autonomously. It will learn entirely through trial and error, guided by a system of rewards and penalties.",
        "app_class": CobberTitratorApp
    },
    {
        "part": "Part IV: The 'Results' Domain", "chapter": 18, "title": "Application: Quality Control with CNNs",
        "program_name": "CobberSorter",
        "description": "This activity introduces a practical application of CNNs: automated quality control. You will use the CobberSorter application to classify synthetic images of polymer beads, similar to those seen under an electron microscope.",
        "app_class": CobberSorterApp
    },
    {
        "part": "Part IV: The 'Results' Domain", "chapter": 19, "title": "Understanding AlphaFold",
        "program_name": "CobberAlpha",
        "description": "In this lab, you will use the CobberAlpha application to walk through the process of predicting the structure of insulin’s A-chain using the revolutionary AlphaFold model and comparing it to the known experimental structure.",
        "app_class": CobberAlphaApp
    },
]


class CobberLearnChem(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- NEW: Define project_root as an instance attribute ---
        try:
            self.project_root = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            self.project_root = os.path.abspath('.')
        self.setWindowTitle("CobberLearnChem: A Machine Learning Suite for Chemistry")
        self.setGeometry(100, 100, 1000, 600)  # Increased height slightly

        self.current_lab_window = None  # To hold a reference to an open lab

        # --- Color and Font Definitions ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        # --- Main widget and layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Pane: Curriculum Navigator ---
        self.navigator_tree = QTreeWidget()
        self.navigator_tree.setHeaderLabel("Curriculum Labs")
        self.populate_navigator()
        splitter.addWidget(self.navigator_tree)
        self.navigator_tree.setStyleSheet(f"""
            QTreeWidget {{ font-size: 11pt; }}
            QHeaderView::section {{
                background-color: {self.cobber_gold.name()}; color: black;
                padding: 4px; font-size: 12pt; font-weight: bold;
                border: 1px solid #EAAA00;
            }}
            QTreeWidget::item {{ padding: 5px; }}
        """)

        # --- Right Pane: Managed by a Stacked Widget ---
        self.stacked_widget = QStackedWidget()

        # --- Page 1: The Welcome Page ---
        self.welcome_page = QWidget()
        welcome_layout = QVBoxLayout(self.welcome_page)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        try:
            if getattr(sys, 'frozen', False):
                script_dir = os.path.dirname(sys.executable)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))

        icon_path = os.path.join(script_dir, "ChapterTitleIcon.png")
        self.icon_pixmap = QPixmap(icon_path)
        welcome_icon_label = QLabel()
        welcome_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if not self.icon_pixmap.isNull():
            welcome_icon_label.setPixmap(
                self.icon_pixmap.scaledToHeight(256, Qt.TransformationMode.SmoothTransformation))
        welcome_title = QLabel("Welcome to CobberLearnChem!")
        w_title_font = QFont(self.lato_font);
        w_title_font.setPointSize(22);
        w_title_font.setBold(True)
        welcome_title.setFont(w_title_font)
        welcome_title.setStyleSheet(f"color: {self.cobber_maroon.name()};")
        welcome_subtitle = QLabel("Select a lab from the navigator on the left to begin.")
        w_subtitle_font = QFont(self.lato_font);
        w_subtitle_font.setPointSize(14);
        w_subtitle_font.setItalic(True)
        welcome_subtitle.setFont(w_subtitle_font)
        welcome_layout.addWidget(welcome_icon_label)
        welcome_layout.addSpacing(20)
        welcome_layout.addWidget(welcome_title)
        welcome_layout.addWidget(welcome_subtitle)
        self.stacked_widget.addWidget(self.welcome_page)

        # --- Page 2: The Lab Detail Page ---
        self.lab_detail_page = QWidget()
        details_layout = QVBoxLayout(self.lab_detail_page)
        details_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        details_layout.setContentsMargins(15, 15, 15, 15)
        details_layout.setSpacing(10)
        self.lab_icon_label = QLabel()
        self.lab_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lab_icon_label.setMinimumHeight(72)
        self.title_label = QLabel()
        self.title_label.setWordWrap(True)
        title_font = QFont(self.lato_font);
        title_font.setPointSize(18);
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet(f"color: {self.cobber_maroon.name()};")
        self.program_label = QLabel()
        program_font = QFont(self.lato_font);
        program_font.setPointSize(12);
        program_font.setItalic(True)
        self.program_label.setFont(program_font)
        self.description_label = QLabel()
        desc_font = QFont(self.lato_font);
        desc_font.setPointSize(11)
        self.description_label.setFont(desc_font)
        self.description_label.setWordWrap(True)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.description_label)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.launch_button = QPushButton("Launch Lab")
        self.launch_button.setFont(QFont("Arial", 12))
        self.launch_button.setMinimumHeight(50)
        details_layout.addWidget(self.lab_icon_label)
        details_layout.addSpacing(1)
        details_layout.addWidget(self.title_label)
        details_layout.addWidget(self.program_label)
        details_layout.addSpacing(10)
        details_layout.addWidget(scroll_area, 1)
        details_layout.addWidget(self.launch_button)
        self.stacked_widget.addWidget(self.lab_detail_page)

        # --- Final Layout Assembly and Signal Connections ---
        splitter.addWidget(self.stacked_widget)
        splitter.setSizes([400, 600])
        main_layout.addWidget(splitter)
        self.navigator_tree.currentItemChanged.connect(self.update_details_panel)
        self.launch_button.clicked.connect(self.launch_selected_lab)  # Connect launch button
        self.stacked_widget.setCurrentWidget(self.welcome_page)

    def populate_navigator(self):
        self.navigator_tree.clear()
        parts = {}
        part_font = QFont(self.lato_font);
        part_font.setPointSize(11);
        part_font.setBold(True)
        chapter_font = QFont(self.lato_font);
        chapter_font.setPointSize(10)
        for lab in LABS_DATA:
            part_name = lab["part"]
            if part_name not in parts:
                parts[part_name] = QTreeWidgetItem(self.navigator_tree, [part_name])
                parts[part_name].setFont(0, part_font)
            parent_item = parts[part_name]
            chapter_text = f"Ch {lab['chapter']}: {lab['title']}"
            child_item = QTreeWidgetItem(parent_item, [chapter_text])
            child_item.setData(0, Qt.ItemDataRole.UserRole, lab)
            child_item.setFont(0, chapter_font)
            child_item.setForeground(0, self.cobber_maroon)
        for part_item in parts.values():
            part_item.setExpanded(True)

    def update_details_panel(self, current_item, previous_item):
        if current_item is None or current_item.childCount() > 0:
            self.stacked_widget.setCurrentWidget(self.welcome_page)
            return
        lab_data = current_item.data(0, Qt.ItemDataRole.UserRole)
        if lab_data:
            if not self.icon_pixmap.isNull():
                self.lab_icon_label.setPixmap(
                    self.icon_pixmap.scaledToHeight(72, Qt.TransformationMode.SmoothTransformation))
                self.lab_icon_label.setVisible(True)
            else:
                self.lab_icon_label.setVisible(False)
            self.title_label.setText(f"Chapter {lab_data['chapter']}: {lab_data['title']}")
            self.program_label.setText(f"Program: {lab_data['program_name']}")
            self.description_label.setText(lab_data['description'])
            # Enable launch button only if the lab class is available
            self.launch_button.setEnabled(lab_data.get("app_class") is not None)
            self.stacked_widget.setCurrentWidget(self.lab_detail_page)

    def launch_selected_lab(self):
        """Instantiates and shows the main window for the selected lab."""
        current_item = self.navigator_tree.currentItem()
        if not current_item or not current_item.data(0, Qt.ItemDataRole.UserRole):
            return
        lab_data = current_item.data(0, Qt.ItemDataRole.UserRole)
        AppClass = lab_data.get("app_class")

        if not AppClass:
            QMessageBox.information(self, "Under Construction",
                                    f"The '{lab_data['program_name']}' application has not been integrated yet.")
            return

        # Check if the AppClass is the special CobberTreeApp
        if AppClass is CobberTreeApp:
            # First, check if the necessary components were imported correctly
            if not Hydrocarbon:
                QMessageBox.critical(self, "Error", "CobberTree components could not be imported.")
                return

            # Helper function to create full paths to asset files
            def get_asset_path(filename):
                # Use the instance attribute defined in __init__
                return os.path.join(self.project_root, "assets", filename)

            # Lambda function to make creating paths cleaner
            sdf = lambda name: get_asset_path(f"{name.lower().replace('-', '')}.sdf")

            # --- THIS IS THE FULL, UNABBREVIATED LIST ---
            # Recreate the dataset needed by CobberTree's constructor
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
            # Pass the constructed dataset when creating the window
            self.current_lab_window = AppClass(sorted(manual_dataset, key=lambda hc: (hc.num_carbons, hc.num_branches)))
        else:
            # For all other labs (like CobberFetcher, etc.), create them normally
            self.current_lab_window = AppClass()

        self.current_lab_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("ProgramIcon.ico"))
    window = CobberLearnChem()
    window.show()
    sys.exit(app.exec())