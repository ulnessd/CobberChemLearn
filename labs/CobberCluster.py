# CobberCluster.py
# Cheminformatics Similarity Explorer
# hardcoded molecule packs
# with RDKit DLL loader fix

import sys
import os

# robust path fix
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- RDKit DLL fix ---
try:
    if getattr(sys, 'frozen', False):
        # running in a frozen executable
        base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
        rdkit_libs = os.path.join(base_dir, "rdkit.libs")
        print(f"[DEBUG] (frozen) checking for rdkit.libs at: {rdkit_libs}")
    else:
        # running in a normal environment
        venv_site = os.path.join(sys.prefix, "Lib", "site-packages")
        rdkit_libs = os.path.join(venv_site, "rdkit.libs")
        print(f"[DEBUG] (venv) checking for rdkit.libs at: {rdkit_libs}")

    if os.path.exists(rdkit_libs):
        os.add_dll_directory(rdkit_libs)
        print(f"[INFO] Added RDKit DLL directory: {rdkit_libs}")
    else:
        print(f"[WARNING] rdkit.libs directory does NOT exist at: {rdkit_libs}")

except Exception as e:
    print(f"[ERROR] Could not set RDKit DLL directory: {e}")



import requests
import traceback
import numpy as np
import pandas as pd
from urllib.parse import quote
import base64
from urllib.request import urlopen, HTTPError

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QMainWindow,
    QMessageBox, QComboBox, QPushButton, QLabel, QStatusBar
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices, QFont, QColor

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, TapTool, LinearColorMapper, ColorBar, BasicTicker, CustomJS
from bokeh.resources import CDN
from bokeh.embed import file_html
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, Descriptors3D
from rdkit import __version__ as rdkit_version
from rdkit.Chem import rdFingerprintGenerator

print(f"Using RDKit version: {rdkit_version}")

# ---------------------------------------------------------------------
# Hard-coded molecule packs
# ---------------------------------------------------------------------
AMINO_ACID_PACK = [
    "alanine", "arginine", "asparagine", "aspartic acid", "cysteine",
    "glutamine", "glutamic acid", "glycine", "histidine", "isoleucine",
    "leucine", "lysine", "methionine", "phenylalanine", "proline",
    "serine", "threonine", "tryptophan", "tyrosine", "valine"
]

FRESHMAN_MAIN_PACK = [
    "water", "ammonia", "methane", "ethanol", "acetone", "acetic acid",
    "benzene", "glucose", "caffeine", "serotonin", "sulfuric acid", "histamine"
]

FRESHMAN_EXTENDED_PACK = [
    "water", "carbon dioxide", "oxygen", "ammonia", "methane",
    "acetate", "carbonate", "nitrate", "phosphate", "cyanide",
    "ethanol", "acetone", "acetic acid", "formaldehyde", "butane",
    "benzene", "toluene", "phenol", "glucose", "urea", "lactic acid",
    "caffeine", "nicotine", "serotonin", "dopamine", "epinephrine",
    "hydrochloric acid", "sulfuric acid", "ammonium", "bicarbonate",
    "hydroxide", "nitrate", "histamine"
]

DRUG_MAIN_PACK = [
    "metformin", "acetaminophen", "ibuprofen", "acetylsalicylic acid",
    "morphine", "sertraline", "amoxicillin", "oseltamivir", "sildenafil",
    "loratadine", "caffeine", "nicotine"
]

DRUG_EXTENDED_PACK = [
    "metformin", "empagliflozin", "dapagliflozin", "canagliflozin",
    "acetaminophen", "ibuprofen", "naproxen", "acetylsalicylic acid",
    "celecoxib", "morphine", "fentanyl", "oxycodone", "fluoxetine",
    "sertraline", "escitalopram", "alprazolam", "lorazepam",
    "methylphenidate", "amphetamine", "lisdexamfetamine", "bupropion",
    "amoxicillin", "penicillin G", "ciprofloxacin", "acyclovir",
    "oseltamivir", "nirmatrelvir", "remdesivir", "levonorgestrel",
    "estrogen", "testosterone", "tamoxifen", "sildenafil", "tadalafil",
    "loratadine", "diphenhydramine", "cetirizine", "pseudoephedrine",
    "guaifenesin", "fluticasone", "albuterol", "montelukast", "caffeine",
    "nicotine", "tetrahydrocannabinol", "cannabidiol", "melatonin",
    "acetylcholine", "naloxone", "methadone", "ketamine", "ibogaine"
]

ORGANIC_MAIN_PACK = [
    "hexane", "cyclohexane", "1-hexene", "benzene", "ethanol",
    "diethyl ether", "acetone", "acetic acid", "ethyl acetate",
    "ethylamine", "acetamide", "chloroform"
]

ORGANIC_EXTENDED_PACK = [
    "methane", "ethane", "propane", "butane", "pentane", "hexane", "cyclohexane",
    "ethene", "propene", "1-butene", "1-pentene", "1-hexene", "1-butyne", "2-butyne",
    "benzene", "toluene", "p-xylene", "anisole", "styrene", "phenol", "methanol",
    "ethanol", "isopropanol", "butanol", "glycerol", "dimethyl ether", "diethyl ether",
    "tetrahydrofuran", "1,4-dioxane", "formaldehyde", "acetaldehyde", "acetone",
    "butanone", "benzaldehyde", "formic acid", "acetic acid", "propionic acid",
    "butyric acid", "benzoic acid", "methyl acetate", "ethyl acetate", "methyl benzoate",
    "isoamyl acetate", "methylamine", "ethylamine", "aniline", "trimethylamine",
    "diisopropylamine", "acetamide", "benzamide", "dimethylformamide", "acetonitrile",
    "butanenitrile", "ethanethiol", "thiophenol", "urea", "acetyl chloride",
    "chloroform", "carbon tetrachloride"
]

# ---------------------------------------------------------------------
# Cheminfo Engine
# ---------------------------------------------------------------------

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
                self._emit_progress(f"{name} done ({i+1}/{len(molecules)})")
            except Exception as e:
                print(f"Could not fetch {property_name} for {name}: {e}")
        return results

    # Corrected function
    def _get_mol_from_smiles(self, name):
        if name in self._mol_cache:
            return self._mol_cache[name]
        self._emit_progress(f"Fetching SMILES for {name}...")
        try:
            # Corrected URL to use the more reliable /TXT endpoint
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/property/CanonicalSMILES/TXT"
            r = requests.get(url, timeout=10)
            r.raise_for_status()

            # Simpler, more robust way to get the SMILES string
            smiles = r.text.strip()

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=1)
            AllChem.MMFFOptimizeMolecule(mol)
            self._mol_cache[name] = mol
            return mol
        except Exception as e:
            print(f"Could not get SMILES for {name}: {e}")
            return None

    def _calculate_scalar_similarity(self, values_dict):
        names = list(values_dict.keys())
        values = np.array(list(values_dict.values()))
        n = len(names)
        sim_matrix = np.zeros((n, n))
        max_diff = np.max(np.abs(values[:, None] - values[None, :]))
        if max_diff == 0:
            max_diff = 1.0
        for i in range(n):
            for j in range(n):
                diff = abs(values[i] - values[j])
                sim_matrix[i, j] = 1 - diff / max_diff
        return sim_matrix, names

    def get_similarity_matrix(self, molecules, criterion):
        self._emit_progress(f"Starting: {criterion}")
        scalar_properties = {
            "TPSA": "TPSA",
            "XLogP (Solubility)": "XLogP",
            "Molecular Weight": "MolecularWeight",
            "Rotatable Bonds (Flexibility)": "RotatableBondCount",
            "H-Bond Donors": "HBondDonorCount",
            "H-Bond Acceptors": "HBondAcceptorCount"
        }
        if criterion in scalar_properties:
            props = self._fetch_scalar_properties(molecules, scalar_properties[criterion])
            if not props:
                return None, None
            return self._calculate_scalar_similarity(props)

        mols = {name: self._get_mol_from_smiles(name) for name in molecules}
        mols = {name: m for name, m in mols.items() if m is not None}
        if not mols:
            return None, None
        names = list(mols.keys())

        if criterion == "Ellipsoid Volume (3D Size)":
            self._emit_progress("Calculating ellipsoid volumes...")
            volumes = {}
            for name, mol in mols.items():
                try:
                    mass = Descriptors.MolWt(mol)
                    pmi1 = Descriptors3D.PMI1(mol)
                    pmi2 = Descriptors3D.PMI2(mol)
                    pmi3 = Descriptors3D.PMI3(mol)
                    a2 = (2.5 / mass) * (pmi2 + pmi3 - pmi1)
                    b2 = (2.5 / mass) * (pmi1 + pmi3 - pmi2)
                    c2 = (2.5 / mass) * (pmi1 + pmi2 - pmi3)
                    a2 = max(a2, 0)
                    b2 = max(b2, 0)
                    c2 = max(c2, 0)
                    vol = (4/3) * np.pi * np.sqrt(a2 * b2 * c2)
                    volumes[name] = vol
                except:
                    print(f"Could not calculate ellipsoid volume for {name}")
            return self._calculate_scalar_similarity(volumes)

        if criterion == "Shape (NPR)":
            self._emit_progress("Calculating NPR shape descriptors...")
            npr = {name: (Descriptors3D.NPR1(m), Descriptors3D.NPR2(m)) for name, m in mols.items()}
            points = np.array(list(npr.values()))
            dist = np.sqrt(np.sum((points[:, None, :] - points[None, :, :])**2, axis=-1))
            max_dist = np.max(dist)
            if max_dist == 0:
                max_dist = 1.0
            sim_matrix = 1 - dist / max_dist
            return sim_matrix, names

        if criterion == "Tanimoto (Overall Structure)":
            self._emit_progress("Calculating Tanimoto similarity...")

            # Create a generator for Morgan fingerprints
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

            # The correct method name is GetFingerprint
            fps = [fpgen.GetFingerprint(m) for m in mols.values()]

            n = len(fps)
            sim_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    sim_matrix[i, j] = sim_matrix[j, i] = s
            return sim_matrix, names

        raise ValueError(f"Unknown criterion: {criterion}")



# ---------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------

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
            e = CheminfoEngine(progress_callback=self.progress)
            result = e.get_similarity_matrix(self.molecules, self.criterion)
            self.finished.emit(result)
        except Exception as ex:
            self.error.emit(str(ex))

# ---------------------------------------------------------------------
# 3D Viewer
# ---------------------------------------------------------------------

def fetch_sdf_base64(name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/SDF?record_type=3d"
        data = urlopen(url, timeout=10).read()
        return base64.b64encode(data).decode("utf8")
    except:
        return None

def generate_3dmol_html(name, sdf):
    return f"""
    <html><body>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol.js"></script>
    <div id="viewer" style="width:100%;height:100vh"></div>
    <script>
    var viewer=$3Dmol.createViewer("viewer");
    var sdf=atob("{sdf}");
    viewer.addModel(sdf,"sdf");
    viewer.setStyle({{}},{{stick:{{}}}});
    viewer.zoomTo();viewer.render();
    </script></body></html>
    """

class MoleculeWindow(QMainWindow):
    def __init__(self, name, sdf, x, y, w, h):
        super().__init__()
        self.setGeometry(x, y, w, h)
        self.setWindowTitle(name)
        view = QWebEngineView()
        view.setHtml(generate_3dmol_html(name, sdf))
        self.setCentralWidget(view)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def create_bokeh_plot(sim, names):
    df = pd.DataFrame(sim, index=names, columns=names)
    df_flat = df.stack().reset_index()
    df_flat.columns = ['mol1', 'mol2', 'value']
    source = ColumnDataSource(df_flat)
    mapper = LinearColorMapper(palette="Viridis256", low=df_flat.value.min(), high=df_flat.value.max())
    p = figure(title="Similarity Matrix", x_range=names, y_range=list(reversed(names)), tools="",sizing_mode="stretch_both" )
    p.title.text_font_size = '14pt'
    p.rect(x="mol2", y="mol1", width=1, height=1, source=source,
           fill_color={"field": "value", "transform": mapper}, line_color=None)
    p.add_tools(HoverTool(tooltips=[("Pair", "@mol1 / @mol2"), ("Similarity", "@value{0.2f}")]))
    p.add_tools(TapTool())
    p.xaxis.major_label_orientation = 1.2
    p.xaxis.major_label_text_font_size = "11pt"
    p.yaxis.major_label_text_font_size = "11pt"
    color_bar = ColorBar(color_mapper=mapper)
    p.add_layout(color_bar, "right")
    cb = CustomJS(args=dict(source=source), code="""
        if (source.selected.indices.length > 0) {
            let i = source.selected.indices[0];
            alert("bokeh-molecule-pair://" + source.data.mol1[i] + "::" + source.data.mol2[i]);
            source.selected.indices = [];
        }
    """)
    p.js_on_event('tap', cb)
    return file_html(p, CDN, "Similarities")

# ---------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------

class InterceptingPage(QWebEnginePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent

    def javaScriptAlert(self, origin, msg):
        if msg.startswith("bokeh-molecule-pair://"):
            a, b = msg.replace("bokeh-molecule-pair://", "").split("::")
            self.main_window.launch_3d_viewers(a, b)
        else:
            super().javaScriptAlert(origin, msg)

class CobberClusterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CobberCluster")
        self.setGeometry(100, 100, 900, 800)
        self.viewer_windows = []
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.setFont(QFont("Lato"))

        self.pack_map = {
            "Freshman - Basic": FRESHMAN_MAIN_PACK,
            "Freshman - Extended": FRESHMAN_EXTENDED_PACK,
            "Drugs - Basic": DRUG_MAIN_PACK,
            "Drugs - Extended": DRUG_EXTENDED_PACK,
            "Amino Acids": AMINO_ACID_PACK,
            "Organic - Basic": ORGANIC_MAIN_PACK,
            "Organic - Extended": ORGANIC_EXTENDED_PACK,
        }
        self.criteria_list = [
            "TPSA", "XLogP (Solubility)", "Molecular Weight", "Ellipsoid Volume (3D Size)",
            "Shape (NPR)", "Rotatable Bonds (Flexibility)", "H-Bond Donors", "H-Bond Acceptors",
            "Tanimoto (Overall Structure)"
        ]

        layout = QVBoxLayout()
        layout.addLayout(self._create_controls_area())
        layout.addWidget(self._create_display_area())
        self.setStatusBar(self._create_status_bar())
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

    def _create_controls_area(self):
        l = QHBoxLayout()
        self.pack_combo = QComboBox()
        self.pack_combo.addItems(self.pack_map.keys())
        self.criterion_combo = QComboBox()
        self.criterion_combo.addItems(self.criteria_list)
        self.generate = QPushButton("Generate a Matrix")
        self.generate.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                padding: 6px 12px;
            }
        """)

        self.generate.clicked.connect(self.start_calculation)
        l.addWidget(QLabel("Select Molecule Pack:"))
        l.addWidget(self.pack_combo)
        l.addWidget(QLabel("Select Similarity Criterion:"))
        l.addWidget(self.criterion_combo)
        l.addWidget(self.generate)
        l.addStretch()
        return l

    def _create_display_area(self):
        self.web = QWebEngineView()
        page = InterceptingPage(self)
        self.web.setPage(page)

        # Define the styled welcome message as a multi-line HTML string
        welcome_html = """
        <html>
          <head>
            <style>
              body {
                background-color: white;          /* Explicitly set background to white */
                font-family: 'Lato', sans-serif;  /* Use Lato, fallback to any sans-serif */
                text-align: center;             /* Center the text */
                padding-top: 3em;               /* Add some space from the top */
              }
            </style>
          </head>
          <body>
            <h1>Welcome to CobberCluster</h1>
          </body>
        </html>
        """
        # Set the HTML for the web view
        self.web.setHtml(welcome_html)

        return self.web

    def _create_status_bar(self):
        bar = QStatusBar()
        bar.showMessage("Ready.")
        help = QPushButton("Definitions")
        help.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(
            "https://www.darinulness.com/teaching/machine-learning-for-undergraduate-chemistry/cobber-compare-definitions"
        )))
        bar.addPermanentWidget(help)
        return bar

    def start_calculation(self):
        self.generate.setEnabled(False)
        self.statusBar().showMessage("Running...")
        mols = self.pack_map[self.pack_combo.currentText()]
        crit = self.criterion_combo.currentText()
        self.worker = Worker(mols, crit)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_calc)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.statusBar().showMessage)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def on_calc(self, result):
        sim, names = result
        if sim is not None:
            self.web.setHtml(create_bokeh_plot(sim, names))
            self.statusBar().showMessage("Ready.")
        else:
            QMessageBox.warning(self, "Error", "Calculation failed.")
            self.statusBar().showMessage("Ready.")
        self.generate.setEnabled(True)

    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.statusBar().showMessage("Ready.")
        self.generate.setEnabled(True)

    def launch_3d_viewers(self, a, b):
        for w in self.viewer_windows:
            w.close()
        self.viewer_windows.clear()
        pos = self.pos()
        offset = 60
        coords = [(pos.x() + 50, pos.y() + 50), (pos.x() + 50 + offset, pos.y() + 50 + offset)]
        for i, n in enumerate([a, b]):
            sdf = fetch_sdf_base64(n)
            if sdf:
                win = MoleculeWindow(n, sdf, *coords[i], 500, 400)
                win.show()
                self.viewer_windows.append(win)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CobberClusterApp()
    win.show()
    sys.exit(app.exec())

