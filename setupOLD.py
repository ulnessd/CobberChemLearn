from cx_Freeze import setup, Executable
import os
import sys
import PyQt6

main_script = "CobberLearnChem1.py"

# PyQt6 paths
pyqt6_dir = os.path.dirname(PyQt6.__file__)
qt6_plugins = os.path.join(pyqt6_dir, "Qt6", "plugins")
qt6_translations = os.path.join(pyqt6_dir, "Qt6", "translations")
qt6_qml = os.path.join(pyqt6_dir, "Qt6", "qml")

# RDKit
venv_site = os.path.join(sys.prefix, "Lib", "site-packages")
rdkit_libs = os.path.join(venv_site, "rdkit.libs")

include_files = [
    (rdkit_libs, "rdkit.libs"),
    (os.path.join(qt6_plugins, "platforms"), "platforms"),
    (os.path.join(qt6_plugins, "styles"), "styles"),
    (qt6_translations, "translations"),
    (qt6_qml, "qml")
]

build_exe_options = {
    "packages": [
        "os", "sys", "numpy", "scipy", "sklearn", "Bio", "matplotlib",
        "seaborn", "pandas", "pubchempy", "py3Dmol", "PyQt6", "rdkit",
        # Keras from TensorFlow
        "keras"
    ],
    "include_files": include_files,
    "excludes": ["tkinter"],
    "include_msvcr": True,
}

setup(
    name="CobberLearnChem1",
    version="1.0",
    description="Chemistry learning for AI/ML in chemistry",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            main_script,
            base=None,
            icon="ProgramIcon.ico"
        )
    ]
)

