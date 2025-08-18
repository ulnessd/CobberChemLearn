# CobberLearnChem.py
# Main launcher application for the suite of lab programs.

import tensorflow as tf

import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTreeWidget, QTreeWidgetItem, QLabel, QFrame,
    QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap, QColor, QIcon

# --- Import all lab applications ---
# NOTE: It's okay if some of these fail during development.
try:
    from labs.CobberFetcher import CobberFetcherApp
except ImportError:
    CobberFetcherApp = None
try:
    from labs.CobberCluster import CobberClusterApp
except ImportError:
    CobberClusterApp = None
try:
    from labs.CobberImpute import CobberImputeApp
except ImportError:
    CobberImputeApp = None
try:
    from labs.CobberResidue import CobberResidueApp
except ImportError:
    CobberResidueApp = None
try:
    from labs.CobberTree import CobberTreeApp, Hydrocarbon
except ImportError:
    CobberTreeApp, Hydrocarbon = None, None
try:
    from labs.CobberK import CobberKApp
except ImportError:
    CobberKApp = None
try:
    from labs.CobberDescender import CobberDescenderApp
except ImportError:
    CobberDescenderApp = None
try:
    from labs.CobberEvolve import CobberEvolveApp
except ImportError:
    CobberEvolveApp = None
try:
    from labs.CobberNeuron import CobberNeuronApp
except ImportError:
    CobberNeuronApp = None
try:
    from labs.CobberCNN import CobberCNNApp
except ImportError:
    CobberCNNApp = None
try:
    from labs.CobberTarPit import CobberTarPitApp
except ImportError:
    CobberTarPitApp = None
try:
    from labs.CobberLand import CobberLandApp
except ImportError:
    CobberLandApp = None
try:
    from labs.CobberHbond import CobberHbondApp
except ImportError:
    CobberHbondApp = None
try:
    from labs.CobberTitrator import CobberTitratorApp
except ImportError:
    CobberTitratorApp = None
try:
    from labs.CobberSorter import CobberSorterApp
except ImportError:
    CobberSorterApp = None
try:
    from labs.CobberAlpha import CobberAlphaApp
except ImportError:
    CobberAlphaApp = None

# --- UPDATED: Finalized LABS_DATA structure ---
LABS_DATA = [
    {
        "part": "Part II: The Data Domain", "chapter": 5,
        "title": "From Click to Code: Accessing Chemical Data with Confidence",
        "program_name": "CobberFetcher", "app_class": CobberFetcherApp
    },
    {
        "part": "Part II: The Data Domain", "chapter": 6,
        "title": "Patterns Among Molecules: Exploring Chemical Similarity",
        "program_name": "CobberCluster", "app_class": CobberClusterApp
    },
    {
        "part": "Part II: The Data Domain", "chapter": 7, "title": "Mind the Gaps: Learning to Impute with Insight",
        "program_name": "CobberImpute", "app_class": CobberImputeApp
    },
    {
        "part": "Part II: The Data Domain", "chapter": 8,
        "title": "Judging the Fit: A Practical Guide to Error Analysis",
        "program_name": "CobberResidue", "app_class": CobberResidueApp
    },
    {
        "part": "Part III: The Models Domain", "chapter": 9,
        "title": "Branch by Branch: Teaching Trees to Make Predictions",
        "program_name": "CobberTree", "app_class": CobberTreeApp
    },
    {
        "part": "Part III: The Models Domain", "chapter": 10,
        "title": "Clustering and Classification: Mapping Meaning",
        "program_name": "CobberK", "app_class": CobberKApp
    },
    {
        "part": "Part III: The Models Domain", "chapter": 11,
        "title": "Step by Step: Learning Through Gradient Descent",
        "program_name": "CobberDescender", "app_class": CobberDescenderApp
    },
    {
        "part": "Part III: The Models Domain", "chapter": 12, "title": "An Introduction to Evolutionary Algorithms",
        "program_name": "CobberEvolve", "app_class": CobberEvolveApp
    },
    {
        "part": "Part III: The Models Domain", "chapter": 13,
        "title": "An Introduction to Artificial Neural Networks",
        "program_name": "CobberNeuron", "app_class": CobberNeuronApp
    },
    {
        "part": "Part III: The Models Domain", "chapter": 14, "title": "How do computers see? Intro to CNNs",
        "program_name": "CobberCNN", "app_class": CobberCNNApp
    },
    {
        "part": "Part III: The Models Domain", "chapter": 15, "title": "An Introduction to Reinforcement Learning",
        "program_name": "CobberTarPit", "app_class": CobberTarPitApp
    },
    {
        "part": "Part IV: The Results Domain", "chapter": 16, "title": "Exploring Regression Models in Cobberland",
        "program_name": "CobberLand", "app_class": CobberLandApp
    },
    {
        "part": "Part IV: The Results Domain", "chapter": 17,
        "title": "Discovering Hydrogen Bonding with Unsupervised Learning",
        "program_name": "CobberHbond", "app_class": CobberHbondApp
    },
    {
        "part": "Part IV: The Results Domain", "chapter": 18,
        "title": "Training a Titrating Robot with Reinforcement Learning",
        "program_name": "CobberTitrator", "app_class": CobberTitratorApp
    },
    {
        "part": "Part IV: The Results Domain", "chapter": 19, "title": "Application: Quality Control with CNNs",
        "program_name": "CobberSorter", "app_class": CobberSorterApp
    },
    {
        "part": "Part IV: The Results Domain", "chapter": 20,
        "title": "The AI Revolution in BioChemistry: Understanding AlphaFold",
        "program_name": "CobberAlpha", "app_class": CobberAlphaApp
    },
]


class CobberLearnChem(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.project_root = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            self.project_root = os.path.abspath('.')
        self.setWindowTitle("CobberLearnChem: A Machine Learning Suite for Chemistry")
        self.setGeometry(100, 100, 1000, 700)
        self.current_lab_window = None
        self.cobber_maroon = QColor(108, 29, 69);
        self.cobber_gold = QColor(234, 170, 0);
        self.lato_font = QFont("Lato")
        self.setWindowIcon(QIcon("ProgramIcon.ico"))

        central_widget = QWidget();
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget);
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Pane: Curriculum Navigator ---
        self.navigator_tree = QTreeWidget();
        self.navigator_tree.setHeaderLabel("Curriculum Labs")
        self.populate_navigator();
        splitter.addWidget(self.navigator_tree)
        self.navigator_tree.setStyleSheet(f"""
            QTreeWidget {{ font-size: 11pt; }} QHeaderView::section {{
                background-color: {self.cobber_maroon.name()}; color: white; padding: 4px;
                font-size: 12pt; font-weight: bold; border: 1px solid #6C1D45;
            }} QTreeWidget::item {{ padding: 5px; }}
        """)

        # --- REWRITTEN: Right Pane is now the static Welcome Page ---
        welcome_page = QWidget()
        welcome_layout = QVBoxLayout(welcome_page)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Define path for the icon robustly
        try:
            if getattr(sys, 'frozen', False):
                script_dir = os.path.dirname(sys.executable)
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
        icon_path = os.path.join(script_dir, "ProgramIcon.png")

        icon_pixmap = QPixmap(icon_path)
        welcome_icon_label = QLabel()
        welcome_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if not icon_pixmap.isNull():
            welcome_icon_label.setPixmap(icon_pixmap.scaledToHeight(256, Qt.TransformationMode.SmoothTransformation))

        welcome_title = QLabel("Welcome to CobberLearnChem!");
        w_title_font = QFont(self.lato_font);
        w_title_font.setPointSize(22);
        w_title_font.setBold(True);
        welcome_title.setFont(w_title_font)
        welcome_title.setStyleSheet(f"color: {self.cobber_maroon.name()};")

        welcome_subtitle = QLabel("Double-click a lab from the navigator to launch.");
        w_subtitle_font = QFont(self.lato_font);
        w_subtitle_font.setPointSize(14);
        w_subtitle_font.setItalic(True);
        welcome_subtitle.setFont(w_subtitle_font)

        welcome_layout.addWidget(welcome_icon_label);
        welcome_layout.addSpacing(20);
        welcome_layout.addWidget(welcome_title);
        welcome_layout.addWidget(welcome_subtitle)

        splitter.addWidget(welcome_page)  # Add the welcome page directly

        # --- Final Layout Assembly and Signal Connections ---
        splitter.setSizes([400, 600]);
        main_layout.addWidget(splitter)

        # --- REWRITTEN: Connect itemActivated to launch the lab directly ---
        self.navigator_tree.itemActivated.connect(self.launch_lab)

    def populate_navigator(self):
        self.navigator_tree.clear();
        parts = {}
        part_font = QFont(self.lato_font);
        part_font.setPointSize(11);
        part_font.setBold(True)
        chapter_font = QFont(self.lato_font);
        chapter_font.setPointSize(10)
        for lab in LABS_DATA:
            part_name = lab["part"]
            if part_name not in parts:
                parts[part_name] = QTreeWidgetItem(self.navigator_tree, [part_name]);
                parts[part_name].setFont(0, part_font)
            parent_item = parts[part_name]
            chapter_text = f"Chapter: {lab['chapter']}: {lab['program_name']}"
            child_item = QTreeWidgetItem(parent_item, [chapter_text])
            child_item.setData(0, Qt.ItemDataRole.UserRole, lab);
            child_item.setFont(0, chapter_font)
            child_item.setForeground(0, self.cobber_maroon)
        for part_item in parts.values(): part_item.setExpanded(True)

    # --- REWRITTEN: Combined launch logic into a single event handler ---
    def launch_lab(self, item, column):
        """Instantiates and shows the main window for the selected lab."""
        if item is None or item.childCount() > 0: return  # Ignore clicks on part headers

        lab_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not lab_data: return

        AppClass = lab_data.get("app_class")
        if not AppClass:
            QMessageBox.information(self, "Under Construction",
                                    f"The '{lab_data['program_name']}' application has not been integrated yet.")
            return

        try:
            if AppClass is CobberTreeApp:
                if not Hydrocarbon:
                    raise ImportError("CobberTree's Hydrocarbon class not available.")

                def get_asset_path(filename):
                    return os.path.join(self.project_root, "assets", filename)

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
                self.current_lab_window = AppClass(
                    sorted(manual_dataset, key=lambda hc: (hc.num_carbons, hc.num_branches)))
            else:
                self.current_lab_window = AppClass()

            self.current_lab_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Could not launch the lab application.\nError: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        # Robust path for icon
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "assets", "ProgramIcon.ico")
        if os.path.exists(icon_path): app.setWindowIcon(QIcon(icon_path))
    except Exception as e:
        print(f"Could not set window icon: {e}")

    window = CobberLearnChem()
    window.show()
    sys.exit(app.exec())