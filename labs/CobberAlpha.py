# CobberAlpha.py
# An application to assist in running AlphaFold and comparing the results
# to experimental PDB structures.
# Refactored for the CobberLearnChem launcher.

import sys
import os
import webbrowser
import py3Dmol
import tempfile
import zipfile
import json
import re
import numpy as np
import shutil

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGroupBox, QPlainTextEdit,
    QMessageBox, QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from Bio.PDB import PDBList, PDBParser, PDBIO, Select, PPBuilder

# --- NEW: Data structure for selectable proteins ---
PROTEIN_MAP = {
    "Insulin (A-chain)": ("1TRZ", "A"),
    "Ubiquitin": ("1UBQ", "A"),
    "Green Fluorescent Protein": ("1EMA", "A"),
    "Lysozyme": ("1LYZ", "A"),
    "Villin's Headpiece Subdomain": ("1VII", "A"),
    "Myoglobin": ("1MBN", "A")
}


class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return 1 if chain.get_id() == self.chain_id else 0

    def accept_residue(self, residue):
        return 1 if residue.get_id()[0] == " " else 0


class FetchWorker(QThread):
    finished = pyqtSignal(str, str, str)

    def __init__(self, pdb_id, chain_id):
        super().__init__();
        self.pdb_id = pdb_id;
        self.chain_id = chain_id

    def run(self):
        try:
            pdb_list = PDBList()
            original_pdb_path = pdb_list.retrieve_pdb_file(self.pdb_id, pdir='.', file_format='pdb', overwrite=True)
            if not os.path.exists(original_pdb_path): raise FileNotFoundError(f"Could not download {self.pdb_id}.")
            cleaned_pdb_path = f"{self.pdb_id}_chain_{self.chain_id}_cleaned.pdb"
            parser = PDBParser(QUIET=True);
            structure = parser.get_structure(self.pdb_id, original_pdb_path)
            io = PDBIO();
            io.set_structure(structure);
            io.save(cleaned_pdb_path, select=ChainSelect(self.chain_id))
            if os.path.exists(original_pdb_path): os.remove(original_pdb_path)
            chain_structure = parser.get_structure("chain", cleaned_pdb_path)
            fasta_sequence = "".join([str(pp.get_sequence()) for pp in PPBuilder().build_peptides(chain_structure)])
            if not fasta_sequence: raise ValueError(f"No sequence found for Chain {self.chain_id}.")
            self.finished.emit(fasta_sequence, cleaned_pdb_path, "")
        except Exception as e:
            self.finished.emit("", "", str(e))


class CobberAlphaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cobber_maroon = QColor(108, 29, 69);
        self.cobber_gold = QColor(234, 170, 0);
        self.lato_font = QFont("Lato")
        self.setWindowTitle("CobberAlpha");
        self.setGeometry(100, 100, 600, 600);
        self.setFont(self.lato_font)
        self.fasta_sequence = "";
        self.cleaned_pdb_path = None;
        self.predicted_pdb_path = None
        self.current_pdb_id = None;
        self.current_chain_id = None
        self._setup_ui()
        # --- REMOVED automatic fetch ---

    def _setup_ui(self):
        main_widget = QWidget();
        self.setCentralWidget(main_widget);
        main_layout = QVBoxLayout(main_widget)

        # --- NEW: Add protein selection dropdown and fetch button ---
        selection_group = QGroupBox("Select a Protein");
        selection_layout = QHBoxLayout(selection_group)
        self.protein_combo = QComboBox();
        self.protein_combo.addItems(PROTEIN_MAP.keys())
        self.fetch_button = QPushButton("Fetch Selected Protein")
        selection_layout.addWidget(QLabel("Protein:"));
        selection_layout.addWidget(self.protein_combo, 1)
        selection_layout.addWidget(self.fetch_button)
        main_layout.addWidget(selection_group)

        self.copy_button = QPushButton("Copy Raw Sequence");
        self.copy_button.setEnabled(False)
        self.view_exp_button = QPushButton("View Experimental Structure");
        self.view_exp_button.setEnabled(False)
        self.launch_colab_button = QPushButton("Launch ColabFold in Browser")
        self.process_zip_button = QPushButton("Process ColabFold .zip File");
        self.process_zip_button.setEnabled(False)
        self.view_pred_button = QPushButton("View Predicted Structure in Browser");
        self.view_pred_button.setEnabled(False)

        # --- MODIFIED: More generic title ---
        seq_group = QGroupBox("Step 1: Get Protein Sequence");
        seq_layout = QVBoxLayout(seq_group)
        self.fasta_display = QPlainTextEdit();
        self.fasta_display.setReadOnly(True)
        self.fasta_display.setPlaceholderText("Select a protein and click 'Fetch' to begin...")
        seq_layout.addWidget(self.fasta_display);
        seq_layout.addWidget(self.copy_button, 0, Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(seq_group)

        view_exp_group = QGroupBox("Step 2: View Experimental Structure");
        view_exp_layout = QHBoxLayout(view_exp_group)
        view_exp_layout.addWidget(self.view_exp_button);
        main_layout.addWidget(view_exp_group)

        colab_group = QGroupBox("Step 3: Launch ColabFold to Predict Structure");
        colab_layout = QHBoxLayout(colab_group)
        colab_layout.addWidget(self.launch_colab_button);
        main_layout.addWidget(colab_group)

        process_group = QGroupBox("Step 4: Process ColabFold Result");
        process_layout = QHBoxLayout(process_group)
        process_layout.addWidget(self.process_zip_button);
        main_layout.addWidget(process_group)

        view_pred_group = QGroupBox("Step 5: View Predicted Structure");
        view_pred_layout = QHBoxLayout(view_pred_group)
        view_pred_layout.addWidget(self.view_pred_button);
        main_layout.addWidget(view_pred_group)

        log_group = QGroupBox("Log");
        log_layout = QVBoxLayout(log_group)
        self.log_console = QPlainTextEdit();
        self.log_console.setReadOnly(True)
        log_layout.addWidget(self.log_console);
        main_layout.addWidget(log_group)

        # --- MODIFIED: Connect new fetch button ---
        self.fetch_button.clicked.connect(self.start_manual_fetch)
        self.copy_button.clicked.connect(self.copy_sequence);
        self.view_exp_button.clicked.connect(self.view_experimental_structure)
        self.launch_colab_button.clicked.connect(self.launch_colabfold);
        self.process_zip_button.clicked.connect(self.process_zip_file)
        self.view_pred_button.clicked.connect(self.view_predicted_structure)

    def log(self, m):
        self.log_console.appendPlainText(m);
        QApplication.processEvents()

    # --- NEW: Replaces start_automatic_fetch ---
    def start_manual_fetch(self):
        self.set_buttons_enabled(False)
        selection = self.protein_combo.currentText()
        pdb_id, chain_id = PROTEIN_MAP[selection]

        self.fasta_display.setPlaceholderText(f"Fetching sequence for PDB: {pdb_id}, Chain: {chain_id}...")
        self.fasta_display.clear()

        self.log(f"Fetch initiated for PDB {pdb_id}, Chain {chain_id}...");
        self.current_pdb_id = pdb_id
        self.current_chain_id = chain_id

        self.fetch_worker = FetchWorker(pdb_id, chain_id)
        self.fetch_worker.finished.connect(self.on_fetch_finished)
        self.fetch_worker.start()

    def on_fetch_finished(self, fasta_seq, cleaned_path, error_str):
        self.fetch_button.setEnabled(True)  # Always re-enable fetch button
        if error_str:
            self.log(f"Error: {error_str}");
            QMessageBox.critical(self, "Error", f"An error occurred:\n{error_str}");
            return

        self.fasta_sequence = fasta_seq;
        self.cleaned_pdb_path = cleaned_path
        self.fasta_display.setPlainText(f">{self.current_pdb_id}|Chain {self.current_chain_id}\n{fasta_seq}")
        self.view_exp_button.setText(f"View Experimental Structure in Browser")
        self.set_buttons_enabled(True)
        self.log("Success! Sequence ready. Cleaned PDB available.")

    def set_buttons_enabled(self, enabled):
        """Helper to enable/disable buttons post-fetch."""
        self.copy_button.setEnabled(enabled)
        self.view_exp_button.setEnabled(enabled)
        self.process_zip_button.setEnabled(enabled)
        # Prediction button is handled separately

    def copy_sequence(self):
        if not self.fasta_sequence: return
        QApplication.clipboard().setText(self.fasta_sequence);
        self.log(f"Sequence ({len(self.fasta_sequence)} residues) copied.")

    def _create_and_launch_3d_view(self, pdb_path):
        if not pdb_path or not os.path.exists(pdb_path): QMessageBox.warning(self, "File Not Found",
                                                                             f"{pdb_path} not found."); return
        self.log(f"Generating 3D view for {os.path.basename(pdb_path)}...")
        try:
            with open(pdb_path, 'r') as f:
                pdb_data = f.read()
            view = py3Dmol.view(width=800, height=600);
            view.addModel(pdb_data, 'pdb');
            view.setStyle({'cartoon': {'color': 'spectrum'}});
            view.zoomTo()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding='utf-8') as hf:
                view.write_html(hf.name);
                webbrowser.open(f"file://{os.path.abspath(hf.name)}", new=1)
            self.log("Launched 3D view in browser.")
        except Exception as e:
            self.log(f"Error creating view: {e}");
            QMessageBox.critical(self, "View Error", f"Could not generate view:\n{e}")

    def view_experimental_structure(self):
        self._create_and_launch_3d_view(self.cleaned_pdb_path)

    def view_predicted_structure(self):
        self._create_and_launch_3d_view(self.predicted_pdb_path)

    def launch_colabfold(self):
        url = "https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2.ipynb#scrollTo=kOblAo-xetgx";
        webbrowser.open(url);
        self.log(f"Opening ColabFold in browser...")

    def process_zip_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "Select ColabFold Output Zip File", "", "Zip Files (*.zip)");
        if not fp: return
        self.log(f"Processing {os.path.basename(fp)}...")
        try:
            with tempfile.TemporaryDirectory(prefix="colabfold_") as temp_dir:
                with zipfile.ZipFile(fp, 'r') as zr:
                    zr.extractall(temp_dir)
                self.log(f"Extracted zip to temporary directory.");
                scores, pdb_files = [], {}
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(".json") and "scores" in file:
                            try:
                                with open(os.path.join(root, file)) as f:
                                    data = json.load(f)
                                rank = int(re.search(r"rank_(\d+)", file).group(1));
                                scores.append({'rank': rank, 'plddt': np.mean(data.get('plddt', [0]))})
                            except:
                                continue
                        elif file.endswith(".pdb"):
                            rank_match = re.search(r"rank_(\d+)", file)
                            if rank_match: pdb_files[int(rank_match.group(1))] = os.path.join(root, file)
                if not scores: raise FileNotFoundError("No valid score files found.")
                best_model = sorted(scores, key=lambda x: x['rank'])[0];
                best_rank_num = best_model['rank'];
                best_plddt = best_model['plddt']
                self.log(f"Found best model: Rank {best_rank_num} with mean pLDDT: {best_plddt:.2f}")
                source_pdb_path = pdb_files.get(best_rank_num)
                if not source_pdb_path: raise FileNotFoundError(f"PDB file for Rank {best_rank_num} not found.")
                self.predicted_pdb_path = os.path.join(os.getcwd(), f"{self.current_pdb_id}_alphafold_prediction.pdb")
                shutil.copy(source_pdb_path, self.predicted_pdb_path);
                self.log(f"Saved predicted structure to: {self.predicted_pdb_path}")
                self.view_pred_button.setEnabled(True)
                QMessageBox.information(self, "Success",
                                        f"Processed AlphaFold result.\nBest model (Rank {best_rank_num}) saved.")
        except Exception as e:
            self.log(f"Error processing zip file: {e}");
            QMessageBox.critical(self, "Zip Error", f"Error processing zip:\n{e}")

    def closeEvent(self, event):
        for path in [self.cleaned_pdb_path, self.predicted_pdb_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CobberAlphaApp()
    window.show()
    sys.exit(app.exec())