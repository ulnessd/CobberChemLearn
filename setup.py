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
qt6_bin = os.path.join(pyqt6_dir, "Qt6", "bin")
qt6_resources = os.path.join(pyqt6_dir, "Qt6", "resources")

# RDKit
venv_site = os.path.join(sys.prefix, "Lib", "site-packages")
rdkit_libs = os.path.join(venv_site, "rdkit.libs")

include_files = [
    (rdkit_libs, "rdkit.libs"),
    (os.path.join(qt6_plugins, "platforms"), "platforms"),
    (os.path.join(qt6_plugins, "styles"), "styles"),
    (qt6_translations, "translations"),
    (qt6_qml, "qml"),
    # manually include these:
    (os.path.join(pyqt6_dir, "Qt6", "bin", "Qt6WebEngineCore.dll"), "Qt6/bin/Qt6WebEngineCore.dll"),
    (os.path.join(pyqt6_dir, "Qt6", "bin", "Qt6WebEngineWidgets.dll"), "Qt6/bin/Qt6WebEngineWidgets.dll"),
    (os.path.join(pyqt6_dir, "Qt6", "bin", "QtWebEngineProcess.exe"), "Qt6/bin/QtWebEngineProcess.exe"),
    (os.path.join(pyqt6_dir, "Qt6", "resources"), "Qt6/resources"),
    (os.path.join(pyqt6_dir, "Qt6", "qml", "QtWebEngine"), "Qt6/qml/QtWebEngine"),
    (os.path.join(pyqt6_dir, "Qt6", "translations", "qtwebengine_locales"), "Qt6/translations/qtwebengine_locales"),
    "ChapterTitleIcon.png"
]

build_exe_options = {
    "packages": [
        "os", "sys", "numpy", "scipy", "sklearn", "Bio", "matplotlib",
        "seaborn", "pandas", "pubchempy", "py3Dmol", "PyQt6", "rdkit",
        # Keras from TensorFlow
        "keras"
    ],
    "includes": ["labs.CobberCluster"],
    "include_files":include_files,
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
            base=None,  # leave console visible while debugging
            icon="ProgramIcon.ico"
        )
    ]
)
