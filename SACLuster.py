import sys
import requests
import traceback
import numpy as np
import pandas as pd
from urllib.parse import quote
import base64
from urllib.request import urlopen, HTTPError

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QMainWindow,
    QMessageBox, QComboBox, QPushButton, QLabel, QStatusBar
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices

# --- Data Science Imports ---
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, TapTool, LinearColorMapper, ColorBar, BasicTicker, CustomJS
from bokeh.resources import CDN
from bokeh.embed import file_html
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Descriptors3D
from rdkit import __version__ as rdkit_version

# --- Local Imports ---
try:
    import molecule_packs
except ImportError:
    print("FATAL ERROR: molecule_packs.py not found.")
    print("Please make sure the file containing the molecule lists is in the same directory.")
    sys.exit(1)

# Print the RDKit version for debugging purposes
print(f"Using RDKit version: {rdkit_version}")


################################################################################
#
# PART 1: CHEMINFO ENGINE
#
################################################################################

class CheminfoEngine:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self._mol_cache = {}

    def _emit_progress(self, message):
        if self.progress_callback:
            self.progress_callback.emit(message)

    def _fetch_scalar_properties(self, molecules, property_name):
        self._emit_progress(f"Fetching {property_name} for {len(molecules)} molecules...")
        results = {}
        for i, name in enumerate(molecules):
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/property/{property_name}/JSON"
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                data = r.json()
                value = data['PropertyTable']['Properties'][0][property_name]
                results[name] = float(value)
                self._emit_progress(f"Fetched {property_name} for {name} ({i + 1}/{len(molecules)})")
            except Exception as e:
                print(f"Could not fetch {property_name} for '{name}': {e}")
        return results

    def _get_mol_from_smiles(self, name):
        if name in self._mol_cache:
            return self._mol_cache[name]

        self._emit_progress(f"Fetching 3D structure for {name}...")
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/property/CanonicalSMILES/JSON"
        try:
            r = requests.get(smiles_url, timeout=10)
            r.raise_for_status()
            smiles = r.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=1)
            AllChem.MMFFOptimizeMolecule(mol)
            self._mol_cache[name] = mol
            return mol
        except Exception as e:
            print(f"Could not fetch SMILES/3D for '{name}': {e}")
            return None

    def _calculate_scalar_similarity(self, values_dict):
        names = list(values_dict.keys())
        values = np.array([values_dict[n] for n in names])
        n = len(names)
        sim_matrix = np.zeros((n, n))
        max_diff = np.max(np.abs(values[:, None] - values[None, :]))
        if max_diff == 0: max_diff = 1.0

        for i in range(n):
            for j in range(n):
                diff = abs(values[i] - values[j])
                sim_matrix[i, j] = 1 - (diff / max_diff)
        return sim_matrix, names

    def get_similarity_matrix(self, molecules, criterion):
        self._emit_progress(f"Starting calculation for criterion: {criterion}...")

        scalar_properties = {
            "TPSA": "TPSA", "XLogP (Solubility)": "XLogP", "Molecular Weight": "MolecularWeight",
            "Rotatable Bonds (Flexibility)": "RotatableBondCount", "H-Bond Donors": "HBondDonorCount",
            "H-Bond Acceptors": "HBondAcceptorCount"
        }
        if criterion in scalar_properties:
            prop_name = scalar_properties[criterion]
            data = self._fetch_scalar_properties(molecules, prop_name)
            if not data: return None, None
            return self._calculate_scalar_similarity(data)

        mols_dict = {name: self._get_mol_from_smiles(name) for name in molecules}
        valid_mols = {name: mol for name, mol in mols_dict.items() if mol is not None}
        if not valid_mols: return None, None
        names = list(valid_mols.keys())
        n = len(names)

        # --- THIS ENTIRE BLOCK IS THE NEW, ROBUST FIX ---
        if criterion == "Ellipsoid Volume (3D Size)":
            self._emit_progress("Calculating Ellipsoid Volumes...")
            volumes = {}
            for name, mol in valid_mols.items():
                try:
                    mass = Descriptors.MolWt(mol)
                    # Get principal moments of inertia
                    pmi1, pmi2, pmi3 = Descriptors3D.PMI1(mol), Descriptors3D.PMI2(mol), Descriptors3D.PMI3(mol)

                    # From the moments of inertia of a uniform ellipsoid, we can solve
                    # for the semi-axes lengths (a, b, c).
                    # I_1 = (m/5)*(b^2+c^2) etc.
                    # This leads to a^2 = (5/2m)*(I_2+I_3-I_1)
                    a2 = (2.5 / mass) * (pmi2 + pmi3 - pmi1)
                    b2 = (2.5 / mass) * (pmi1 + pmi3 - pmi2)
                    c2 = (2.5 / mass) * (pmi1 + pmi2 - pmi3)

                    # Ensure non-negative before sqrt, guards against floating point errors
                    if a2 < 0: a2 = 0
                    if b2 < 0: b2 = 0
                    if c2 < 0: c2 = 0

                    # Volume of an ellipsoid is (4/3)*pi*a*b*c
                    vol = (4.0 / 3.0) * np.pi * np.sqrt(a2 * b2 * c2)
                    volumes[name] = vol
                except:
                    print(f"Could not calculate Ellipsoid Volume for {name}")
            return self._calculate_scalar_similarity(volumes)

        if criterion == "Shape (NPR)":
            self._emit_progress("Calculating NPR shape descriptors...")
            npr_values = {name: (Descriptors3D.NPR1(mol), Descriptors3D.NPR2(mol)) for name, mol in valid_mols.items()}
            points = np.array(list(npr_values.values()))
            dist_matrix = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1))
            max_dist = np.max(dist_matrix)
            if max_dist == 0: max_dist = 1.0
            sim_matrix = 1 - (dist_matrix / max_dist)
            return sim_matrix, names

        if criterion == "Tanimoto (Overall Structure)":
            self._emit_progress("Calculating Tanimoto similarities...")
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in valid_mols.values()]
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
            return sim_matrix, names

        raise ValueError(f"Unknown criterion: {criterion}")


################################################################################
#
# PART 2: PYQT6 WORKER THREAD
#
################################################################################

class Worker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, molecules, criterion):
        super().__init__()
        self.molecules = molecules
        self.criterion = criterion

    def run(self):
        try:
            engine = CheminfoEngine(progress_callback=self.progress)
            result = engine.get_similarity_matrix(self.molecules, self.criterion)
            self.finished.emit(result)
        except Exception as e:
            error_str = f"An error occurred: {e}\n\n" + traceback.format_exc()
            self.error.emit(error_str)


################################################################################
#
# PART 3: 3D MOLECULE VIEWER HELPERS
#
################################################################################

def fetch_sdf_base64(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/SDF?record_type=3d"
    try:
        sdf_bytes = urlopen(url, timeout=10).read()
        return base64.b64encode(sdf_bytes).decode("utf-8")
    except HTTPError as e:
        print(f"⚠️ Failed to fetch 3D SDF for {name}: {e}")
        return None


def generate_3dmol_html(mol_name, sdf_b64):
    return f"""
    <!DOCTYPE html><html><head>
    <meta charset="UTF-8"><title>3D Viewer - {mol_name}</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol.js"></script>
    </head><body>
    <div id="viewer" style="width: 100%; height: 100vh; margin: 0; padding: 0;"></div>
    <script>
        (function() {{
            let viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
            let sdf = atob("{sdf_b64}");
            viewer.addModel(sdf, "sdf");
            viewer.setStyle({{}}, {{stick:{{}}}});
            viewer.zoomTo();
            viewer.render();
        }})();
    </script></body></html>
    """


class MoleculeWindow(QMainWindow):
    def __init__(self, mol_name, sdf_b64, x, y, w, h):
        super().__init__()
        self.setWindowTitle(f"3D Viewer: {mol_name}")
        self.setGeometry(x, y, w, h)
        view = QWebEngineView()
        html_content = generate_3dmol_html(mol_name, sdf_b64)
        view.setHtml(html_content)
        self.setCentralWidget(view)


################################################################################
#
# PART 4: BOKEH PLOTTING
#
################################################################################

def create_bokeh_plot(sim_matrix, names):
    df = pd.DataFrame(sim_matrix, index=names, columns=names)
    df_flat = df.stack().reset_index()
    df_flat.columns = ['mol1', 'mol2', 'value']
    source = ColumnDataSource(df_flat)

    low_val, high_val = df_flat['value'].min(), df_flat['value'].max()
    palette = "Viridis256" if low_val >= 0 else "RdBu"
    mapper = LinearColorMapper(palette=palette, low=low_val, high=high_val)

    p = figure(
        title="Chemical Similarity Matrix (Click a Square to View Molecules)",
        x_range=names, y_range=list(reversed(names)),
        width=750, height=750,
        tools="", toolbar_location=None, x_axis_location="above"
    )
    p.rect(x="mol2", y="mol1", width=1, height=1, source=source,
           fill_color={"field": "value", "transform": mapper}, line_color=None)

    hover_tool = HoverTool(tooltips=[("Molecules", "@mol1 vs @mol2"), ("Similarity", "@value{0.3f}")])
    p.add_tools(hover_tool)
    p.add_tools(TapTool())

    p.xaxis.major_label_orientation = 1.2
    p.axis.major_label_text_font_size = "9pt"
    p.grid.grid_line_color = None

    color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(), label_standoff=12, location=(0, 0),
                         title="Similarity")
    p.add_layout(color_bar, 'right')

    callback = CustomJS(args=dict(source=source), code="""
        if (source.selected.indices.length > 0) {
            const index = source.selected.indices[0];
            const mol1 = source.data['mol1'][index];
            const mol2 = source.data['mol2'][index];
            alert(`bokeh-molecule-pair://${mol1}::${mol2}`);
        }
    """)
    p.js_on_event('tap', callback)

    return file_html(p, CDN, "Chemical Similarity")


################################################################################
#
# PART 5: MAIN GUI APPLICATION
#
################################################################################

class InterceptingPage(QWebEnginePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent

    def javaScriptAlert(self, securityOrigin, msg):
        if msg.startswith("bokeh-molecule-pair://"):
            parts = msg.replace("bokeh-molecule-pair://", "").split("::")
            if len(parts) == 2:
                mol1_name, mol2_name = parts
                self.main_window.launch_3d_viewers(mol1_name, mol2_name)
        else:
            super().javaScriptAlert(securityOrigin, msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cheminformatics Similarity Explorer")
        self.setGeometry(100, 100, 900, 900)
        self.viewer_windows = []

        self.pack_map = {
            "Freshman - Basic": molecule_packs.FRESHMAN_MAIN_PACK,
            "Freshman - Extended": molecule_packs.FRESHMAN_EXTENDED_PACK,
            "Drugs - Basic": molecule_packs.DRUG_MAIN_PACK, "Drugs - Extended": molecule_packs.DRUG_EXTENDED_PACK,
            "Amino Acids": molecule_packs.AMINO_ACID_PACK,
            "Organic - Basic": molecule_packs.ORGANIC_MAIN_PACK,
            "Organic - Extended": molecule_packs.ORGANIC_EXTENDED_PACK,
        }
        # --- Updated the criterion name ---
        self.criteria_list = [
            "TPSA", "XLogP (Solubility)", "Molecular Weight", "Ellipsoid Volume (3D Size)",
            "Shape (NPR)", "Rotatable Bonds (Flexibility)", "H-Bond Donors", "H-Bond Acceptors",
            "Tanimoto (Overall Structure)"
        ]

        main_layout = QVBoxLayout()
        main_layout.addLayout(self._create_controls_area())
        main_layout.addWidget(self._create_display_area())
        self.setStatusBar(self._create_status_bar())

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def _create_controls_area(self):
        controls_layout = QHBoxLayout()
        self.pack_combo = QComboBox()
        self.pack_combo.addItems(self.pack_map.keys())
        controls_layout.addWidget(QLabel("Select Molecule Pack:"))
        controls_layout.addWidget(self.pack_combo)
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(self.criteria_list)
        controls_layout.addWidget(QLabel("Select Similarity Criterion:"))
        controls_layout.addWidget(self.criterion_combo)
        controls_layout.addStretch()
        self.generate_button = QPushButton("Generate Matrix")
        self.generate_button.setMinimumHeight(30)
        self.generate_button.clicked.connect(self.start_calculation)
        controls_layout.addWidget(self.generate_button)
        return controls_layout

    def _create_display_area(self):
        self.web_view = QWebEngineView()
        self.custom_page = InterceptingPage(self)
        self.web_view.setPage(self.custom_page)
        self.web_view.setHtml(
            "<html><body><h1>Welcome!</h1><p>Select a pack and criterion, then click 'Generate Matrix' to begin.</p></body></html>")
        return self.web_view

    def _create_status_bar(self):
        status_bar = QStatusBar()
        status_bar.showMessage("Ready.")
        help_button = QPushButton("? About Similarity")
        help_button.clicked.connect(self._open_help_url)
        status_bar.addPermanentWidget(help_button)
        return status_bar

    def _open_help_url(self):
        url = QUrl("https://www.darinulness.com/teaching/machine-learning-for-undergraduate-chemistry/cobber-compare-definitions")
        QDesktopServices.openUrl(url)

    def start_calculation(self):
        self.generate_button.setEnabled(False)
        self.statusBar().showMessage("Starting calculation...")
        selected_pack = self.pack_combo.currentText()
        molecules = self.pack_map[selected_pack]
        criterion = self.criterion_combo.currentText()

        self.thread = QThread()
        self.worker = Worker(molecules, criterion)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_calculation_finished)
        self.worker.error.connect(self.on_calculation_error)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_calculation_finished(self, result):
        sim_matrix, names = result
        if sim_matrix is not None and names and len(names) > 1:
            self.statusBar().showMessage("Calculation complete. Generating plot...")
            html = create_bokeh_plot(sim_matrix, names)
            self.web_view.setHtml(html)
            self.statusBar().showMessage("Ready.")
        else:
            QMessageBox.warning(self, "Calculation Failed",
                                "Could not retrieve enough data to generate a matrix. Check the console for details on which molecules may have failed.")
            self.statusBar().showMessage("Failed. Ready.")

        self.generate_button.setEnabled(True)

    def on_calculation_error(self, error_message):
        self.statusBar().showMessage("An error occurred.")
        QMessageBox.critical(self, "Error", f"A critical error occurred in the calculation thread:\n\n{error_message}")
        self.generate_button.setEnabled(True)

    def launch_3d_viewers(self, mol1_name, mol2_name):
        for win in self.viewer_windows:
            win.close()
        self.viewer_windows.clear()

        main_pos = self.pos()
        start_x, start_y = main_pos.x() + 50, main_pos.y() + 50
        win_w, win_h = 500, 400
        offset = 60
        positions = [(start_x, start_y), (start_x + offset, start_y + offset)]

        for i, name in enumerate([mol1_name, mol2_name]):
            self.statusBar().showMessage(f"Fetching 3D data for {name}...")
            sdf_b64 = fetch_sdf_base64(name)
            if sdf_b64:
                x, y = positions[i]
                win = MoleculeWindow(name, sdf_b64, x, y, win_w, win_h)
                self.viewer_windows.append(win)
                win.show()
            else:
                QMessageBox.warning(self, "Data Fetch Error",
                                    f"Could not retrieve 3D structure for '{name}' from PubChem.")
        self.statusBar().showMessage("Ready.")


# --- Application Entry Point ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())