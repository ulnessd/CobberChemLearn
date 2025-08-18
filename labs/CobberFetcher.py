# cobber_fetcher.py
# A PyQt6 application for fetching chemical data from PubChem and PDB.
#
# Refactored to be a standalone module that can be run directly for testing
# or imported by the main CobberLearnChem launcher.
#
# Dependencies:
# pip install pubchempy py3Dmol PyQt6-WebEngine requests PyQt6

import sys
import requests
from PyQt6.QtCore import (Qt, QObject, pyqtSignal, QRunnable, QThreadPool)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QTextEdit, QPushButton, QComboBox, QTabWidget,
    QMessageBox, QStatusBar, QDialog, QDialogButtonBox
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
import pubchempy as pcp
import py3Dmol


# --- Worker Signals ---
# Used to communicate from the background thread to the main GUI thread.
class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)


# --- Fetcher Worker ---
# A QRunnable worker that performs the network requests in the background.
# This prevents the GUI from freezing during network activity.
class FetcherWorker(QRunnable):
    def __init__(self, db_choice, identifiers):
        super().__init__()
        self.db_choice = db_choice
        self.identifiers = identifiers
        self.signals = WorkerSignals()

    def run(self):
        """The main work of the thread. Fetches data for each identifier."""
        try:
            for identifier in self.identifiers:
                identifier = identifier.strip()
                if not identifier:
                    continue

                self.signals.progress.emit(f"Fetching '{identifier}' from {self.db_choice}...")

                if self.db_choice == "PubChem":
                    data = self._fetch_pubchem(identifier)
                else:  # PDB
                    data = self._fetch_pdb(identifier)

                if data.get("error"):
                    self.signals.error.emit(data["error"])
                else:
                    self.signals.result.emit(data)
        except Exception as e:
            self.signals.error.emit(f"A critical error occurred: {e}")
        finally:
            self.signals.finished.emit()

    def _fetch_pubchem(self, identifier):
        """Fetches data for a small molecule from PubChem."""
        try:
            # This initial call is still useful for getting the CID
            compound = pcp.get_compounds(identifier, 'name')[0]

            # --- NEW: Directly fetch the SMILES string via API call ---
            smiles = "N/A"  # Default value
            try:
                smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/property/CanonicalSMILES/TXT"
                smiles_response = requests.get(smiles_url)
                if smiles_response.status_code == 200:
                    smiles = smiles_response.text.strip()
            except requests.RequestException:
                # If this specific request fails, we can log it or just use the default "N/A"
                pass
            # --- End of new code ---

            # Download 2D image
            img_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/PNG?image_size=512x512"
            img_response = requests.get(img_url)
            img_response.raise_for_status()

            # Download SDF file content for 3D viewing
            sdf_response = requests.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/SDF?record_type=3d")
            sdf_response.raise_for_status()

            # Safely format molecular weight
            weight = "N/A"
            if compound.molecular_weight:
                try:
                    weight_val = float(compound.molecular_weight)
                    weight = f"{weight_val:.2f} g/mol"
                except (ValueError, TypeError):
                    weight = str(compound.molecular_weight)

            return {
                "type": "PubChem", "identifier": identifier, "cid": compound.cid,
                "formula": compound.molecular_formula, "weight": weight,
                "smiles": smiles,  # Use the SMILES string we fetched directly
                "image_data": img_response.content,
                "sdf_data": sdf_response.text,
            }
        except (IndexError, pcp.PubChemHTTPError):
            return {"error": f"Error: Could not find '{identifier}' in PubChem."}
        except requests.RequestException as e:
            return {"error": f"Error fetching data for '{identifier}': {e}"}

    def _fetch_pdb(self, identifier):
        """Fetches data for a macromolecule from RCSB PDB."""
        try:
            pdb_id = identifier.upper()
            # Fetch metadata
            meta_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            meta_response = requests.get(meta_url)
            meta_response.raise_for_status()
            meta_data = meta_response.json()

            # Fetch FASTA sequence
            fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_id}/display"
            fasta_response = requests.get(fasta_url)
            fasta_response.raise_for_status()
            fasta_lines = fasta_response.text.splitlines()
            #clean_fasta = "".join(fasta_lines[1:]) # Remove header line

            # Fetch PDB file content
            pdb_file_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            pdb_file_response = requests.get(pdb_file_url)
            pdb_file_response.raise_for_status()

            # Safely format resolution
            resolution = "N/A"
            res_val = meta_data["rcsb_entry_info"].get("resolution_combined", [None])[0]
            if res_val:
                try:
                    resolution = f"{float(res_val):.2f} Å"
                except (ValueError, TypeError):
                    resolution = str(res_val)

            return {
                "type": "PDB", "identifier": pdb_id, "title": meta_data["struct"]["title"],
                "method": meta_data["exptl"][0]["method"], "resolution": resolution,
                "fasta": "\n".join(fasta_lines), "pdb_data": pdb_file_response.text,
            }
        except requests.RequestException:
            return {"error": f"Error: Could not find PDB ID '{identifier}' or failed to fetch data."}


# --- 3D Molecule Viewer Dialog ---
class MoleculeViewerDialog(QDialog):
    def __init__(self, title, sdf_content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 500, 400)
        layout = QVBoxLayout(self)
        self.browser = QWebEngineView()

        # Setup py3Dmol view
        view = py3Dmol.view(width=480, height=380)
        view.addModel(sdf_content, 'sdf')
        view.setStyle({'stick': {}})
        view.zoomTo()

        self.browser.setHtml(view._make_html())
        layout.addWidget(self.browser)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


# --- Raw File Viewer Dialog ---
class FileViewerDialog(QDialog):
    def __init__(self, title, content, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 600, 500)
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit(self)
        self.text_edit.setPlainText(content)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Courier", 10))
        layout.addWidget(self.text_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


# --- Main Application Class ---
# RENAMED from MainWindow to CobberFetcherApp for clarity
class CobberFetcherApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- BRANDING: Define colors and fonts ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberFetcher")
        self.setGeometry(100, 100, 1200, 800)
        self.setFont(self.lato_font) # Apply base font to the whole window

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self.threadpool = QThreadPool()

        # --- Left Control Panel ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMinimumWidth(300)
        left_panel.setMaximumWidth(400)

        form_layout = QFormLayout()
        self.db_combo = QComboBox()
        self.db_combo.addItems(["PubChem", "PDB"])
        form_layout.addRow(QLabel("Select Database:"), self.db_combo)

        input_label = QLabel("Enter Identifiers (one per line):")
        self.input_console = QTextEdit()
        self.input_console.setPlaceholderText("e.g., Aspirin\nIbuprofen\n\nor\n\n1MBO\n4E7V")

        # --- BRANDING: Style the main action buttons ---
        button_layout = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch Data")
        self.fetch_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.cobber_maroon.name()};
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }}
            #QPushButton:hover {{
             #   background-color: #FFD700; /* A brighter gold */
            }}
        """)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.setStyleSheet("""
            QPushButton { padding: 8px; border-radius: 5px; }
        """)

        button_layout.addWidget(self.fetch_btn)
        button_layout.addWidget(self.clear_btn)

        report_label = QLabel("Report Log:")
        self.report_console = QTextEdit()
        self.report_console.setReadOnly(True)

        left_layout.addLayout(form_layout)
        left_layout.addWidget(input_label)
        left_layout.addWidget(self.input_console, 1)
        left_layout.addLayout(button_layout)
        left_layout.addWidget(report_label)
        left_layout.addWidget(self.report_console, 1)

        # --- Right Results Panel ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 1)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready. Select a database and enter identifiers.")

        # --- Connect signals to slots ---
        self.fetch_btn.clicked.connect(self.start_fetching)
        self.clear_btn.clicked.connect(self.clear_all)

    def start_fetching(self):
        """Initiates the background worker to fetch data."""
        raw_text = self.input_console.toPlainText()
        identifiers = [line for line in raw_text.splitlines() if line.strip()]
        if not identifiers:
            QMessageBox.warning(self, "Input Error", "Please enter at least one identifier.")
            return

        self.fetch_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.report_console.clear()

        worker = FetcherWorker(self.db_combo.currentText(), identifiers)
        worker.signals.result.connect(self.add_result_tab)
        worker.signals.error.connect(self.log_error)
        worker.signals.progress.connect(self.statusBar().showMessage)
        worker.signals.finished.connect(self.on_fetching_finished)
        self.threadpool.start(worker)

    def on_fetching_finished(self):
        """Called when the worker thread is done."""
        self.statusBar().showMessage("Fetching complete.", 5000)
        self.fetch_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

    def log_error(self, error_message):
        """Displays an error message in the report log."""
        self.report_console.append(f"<font color='red'>{error_message}</font>")

    def add_result_tab(self, data):
        """Adds a new tab with the fetched data."""
        if data["type"] == "PubChem":
            self.create_pubchem_tab(data)
        else:
            self.create_pdb_tab(data)
        self.report_console.append(f"<font color='green'>Success: Fetched '{data['identifier']}'.</font>")

    def create_pubchem_tab(self, data):
        """Creates and populates a tab for a PubChem result."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title_label = QLabel(f"<b>{data['identifier']} (CID: {data['cid']})</b>")
        title_label.setFont(QFont("Lato", 14, QFont.Weight.Bold)) # BRANDING

        image_label = QLabel()
        q_img = QImage.fromData(data["image_data"])
        pixmap = QPixmap(QImage.fromData(data["image_data"]))
        image_label.setPixmap(
            pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        info_layout = QFormLayout()
        info_layout.addRow(QLabel("<b>Formula:</b>"), QLabel(data['formula']))
        info_layout.addRow(QLabel("<b>Mol. Weight:</b>"), QLabel(data['weight']))

        smiles_label = QTextEdit(data['smiles'])
        smiles_label.setReadOnly(True)
        smiles_label.setMaximumHeight(50)
        info_layout.addRow(QLabel("<b>SMILES:</b>"), smiles_label)

        view_3d_button = QPushButton("View 3D Structure")
        view_3d_button.clicked.connect(
            lambda: self.show_molecule_viewer(f"3D Viewer: {data['identifier']}", data['sdf_data']))

        layout.addWidget(title_label)
        layout.addWidget(image_label)
        layout.addLayout(info_layout)
        layout.addWidget(view_3d_button)
        layout.addStretch()
        self.tabs.addTab(tab, data['identifier'])

    def create_pdb_tab(self, data):
        """Creates and populates a tab for a PDB result."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        title_label = QLabel(f"<b>{data['identifier']}</b>")
        title_label.setFont(QFont("Lato", 14, QFont.Weight.Bold)) # BRANDING

        info_layout = QFormLayout()
        info_layout.addRow(QLabel("<b>Title:</b>"), QLabel(data['title']))
        info_layout.addRow(QLabel("<b>Method:</b>"), QLabel(data['method']))
        info_layout.addRow(QLabel("<b>Resolution:</b>"), QLabel(data['resolution']))

        fasta_label = QLabel("<b>FASTA Sequence:</b>")
        fasta_text = QTextEdit(data['fasta'])
        fasta_text.setReadOnly(True)
        fasta_text.setFont(QFont("Courier", 10))

        pdb_button = QPushButton("View PDB File")
        pdb_button.clicked.connect(lambda: self.show_file_viewer(f"PDB: {data['identifier']}", data['pdb_data']))

        layout.addWidget(title_label)
        layout.addLayout(info_layout)
        layout.addWidget(fasta_label)
        layout.addWidget(fasta_text)
        layout.addWidget(pdb_button)
        layout.addStretch()
        self.tabs.addTab(tab, data['identifier'])

    def show_molecule_viewer(self, title, sdf_content):
        """Shows the 3D molecule viewer dialog."""
        dialog = MoleculeViewerDialog(title, sdf_content, self)
        dialog.exec()

    def show_file_viewer(self, title, content):
        """Shows the raw text file viewer dialog."""
        dialog = FileViewerDialog(title, content, self)
        dialog.exec()

    def clear_all(self):
        """Clears all inputs and results."""
        self.input_console.clear()
        self.report_console.clear()
        self.tabs.clear()
        self.statusBar().showMessage("Ready. Select a database and enter identifiers.")


# --- Standalone Execution Guard ---
# This block allows the script to be run directly for testing.
# It will NOT execute when this file is imported by another script.
if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Use the renamed class
    window = CobberFetcherApp()
    window.show()
    sys.exit(app.exec())
