# cobber_impute.py
# A PyQt6 application for exploring data imputation strategies.
# This is a complete, self-contained version for the CobberLearnChem launcher.

import sys
import os
from pathlib import Path
from typing import Optional

# --- Robust Import Solution ---
# Add the project's root directory to the Python path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSize
from PyQt6.QtGui import QAction, QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QStackedWidget,
    QSplitter, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QPushButton,
    QTextEdit, QTableView, QTabWidget, QStatusBar, QMessageBox, QGroupBox,
    QRadioButton, QSpinBox, QFormLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

# Set seaborn style for nice plots
sns.set_theme(style="whitegrid")


################################################################################
# PART 1: MODELS LOGIC (from original models.py)
################################################################################

def _run_ensemble_imputer(df, target_col, feature_cols, params):
    is_missing_mask = df[target_col].isna()
    if not is_missing_mask.any(): return df.copy(), {'mae': 0, 'rmse': 0, 'plot_data': pd.DataFrame()}
    df_impute = df[is_missing_mask];
    df_train = df[~is_missing_mask].copy()
    X_train = df_train[feature_cols];
    y_train = df_train[target_col];
    X_impute = df_impute[feature_cols]
    y_train_log = np.log(y_train)
    scaler = StandardScaler();
    X_train_scaled = scaler.fit_transform(X_train);
    X_impute_scaled = scaler.transform(X_impute)
    ll_model = LinearRegression();
    knn_model = KNeighborsRegressor(n_neighbors=params.get('k', 5));
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=params.get('max_depth', 10), random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv_ll_log = cross_val_predict(ll_model, X_train, y_train_log, cv=cv);
    y_pred_cv_ll = np.exp(y_pred_cv_ll_log)
    y_pred_cv_knn = cross_val_predict(knn_model, X_train_scaled, y_train, cv=cv)
    y_pred_cv_rf = cross_val_predict(rf_model, X_train, y_train, cv=cv)
    y_pred_cv_ensemble = (y_pred_cv_ll + y_pred_cv_knn + y_pred_cv_rf) / 3.0
    mae = mean_absolute_error(y_train, y_pred_cv_ensemble);
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv_ensemble))
    ll_model.fit(X_train, y_train_log);
    ll_preds = np.exp(ll_model.predict(X_impute))
    knn_model.fit(X_train_scaled, y_train);
    knn_preds = knn_model.predict(X_impute_scaled)
    rf_model.fit(X_train, y_train);
    rf_preds = rf_model.predict(X_impute)
    imputed_values = (ll_preds + knn_preds + rf_preds) / 3.0
    df_final = df.copy();
    df_final.loc[is_missing_mask, target_col] = imputed_values
    df_final[f"{target_col}_Ensemble_imputed"] = is_missing_mask.astype(int)
    plot_df = pd.DataFrame({'actual': y_train, 'predicted': y_pred_cv_ensemble, 'carbons': X_train['carbons'],
                            'branch number': X_train['branch number']})
    results = {'mae': mae, 'rmse': rmse, 'plot_data': plot_df}
    print(f"Ensemble Imputation for '{target_col}' complete. MAE: {mae:.4f}")
    return df_final, results


def _run_random_forest_imputer(df, target_col, feature_cols, params):
    is_missing_mask = df[target_col].isna();
    if not is_missing_mask.any(): return df.copy(), {'mae': 0, 'rmse': 0, 'plot_data': pd.DataFrame()}
    df_impute = df[is_missing_mask];
    df_train = df[~is_missing_mask]
    X_train = df_train[feature_cols];
    y_train = df_train[target_col];
    X_impute = df_impute[feature_cols]
    model = RandomForestRegressor(n_estimators=100, max_depth=params.get('max_depth', 10), random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42);
    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
    mae = mean_absolute_error(y_train, y_pred_cv);
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    model.fit(X_train, y_train);
    imputed_values = model.predict(X_impute)
    df_final = df.copy();
    df_final.loc[is_missing_mask, target_col] = imputed_values
    df_final[f"{target_col}_Random Forest_imputed"] = is_missing_mask.astype(int)
    plot_df = pd.DataFrame({'actual': y_train, 'predicted': y_pred_cv, 'carbons': X_train['carbons'],
                            'branch number': X_train['branch number']})
    results = {'mae': mae, 'rmse': rmse, 'plot_data': plot_df}
    print(f"Random Forest Imputation for '{target_col}' complete. MAE: {mae:.4f}")
    return df_final, results


def _run_log_linear_imputer(df, target_col, feature_cols, params):
    is_missing_mask = df[target_col].isna()
    if not is_missing_mask.any(): return df.copy(), {'mae': 0, 'rmse': 0, 'plot_data': pd.DataFrame()}
    df_impute = df[is_missing_mask];
    df_train = df[~is_missing_mask].copy()
    if (df_train[target_col] <= 0).any(): return df.copy(), {'mae': float('inf'), 'rmse': float('inf'),
                                                             'plot_data': pd.DataFrame()}
    X_train = df_train[feature_cols];
    y_train_log = np.log(df_train[target_col]);
    X_impute = df_impute[feature_cols]
    model = LinearRegression();
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv_log = cross_val_predict(model, X_train, y_train_log, cv=cv);
    y_pred_cv = np.exp(y_pred_cv_log)
    y_actual = df_train[target_col];
    mae = mean_absolute_error(y_actual, y_pred_cv);
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_cv))
    model.fit(X_train, y_train_log);
    imputed_values_log = model.predict(X_impute);
    imputed_values = np.exp(imputed_values_log)
    df_final = df.copy();
    df_final.loc[is_missing_mask, target_col] = imputed_values
    df_final[f"{target_col}_Log-Linear_imputed"] = is_missing_mask.astype(int)
    plot_df = pd.DataFrame({'actual': y_actual, 'predicted': y_pred_cv, 'carbons': X_train['carbons'],
                            'branch number': X_train['branch number']})
    results = {'mae': mae, 'rmse': rmse, 'plot_data': plot_df}
    print(f"Log-Linear Imputation for '{target_col}' complete. MAE: {mae:.4f}")
    return df_final, results


def _run_knn_imputer(df, target_col, feature_cols, params):
    is_missing_mask = df[target_col].isna()
    if not is_missing_mask.any(): return df.copy(), {'mae': 0, 'rmse': 0, 'plot_data': pd.DataFrame()}
    df_impute = df[is_missing_mask];
    df_train = df[~is_missing_mask]
    X_train = df_train[feature_cols];
    y_train = df_train[target_col];
    X_impute = df_impute[feature_cols]
    scaler = StandardScaler();
    X_train_scaled = scaler.fit_transform(X_train);
    X_impute_scaled = scaler.transform(X_impute)
    model = KNeighborsRegressor(n_neighbors=params.get('k', 5));
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X_train_scaled, y_train, cv=cv)
    mae = mean_absolute_error(y_train, y_pred_cv);
    rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    model.fit(X_train_scaled, y_train);
    imputed_values = model.predict(X_impute_scaled)
    df_final = df.copy();
    df_final.loc[is_missing_mask, target_col] = imputed_values
    df_final[f"{target_col}_KNN_imputed"] = is_missing_mask.astype(int)
    plot_df = pd.DataFrame({'actual': y_train, 'predicted': y_pred_cv, 'carbons': X_train['carbons'],
                            'branch number': X_train['branch number']})
    results = {'mae': mae, 'rmse': rmse, 'plot_data': plot_df}
    print(f"KNN Imputation for '{target_col}' complete. MAE: {mae:.4f}")
    return df_final, results


def run_imputation_model(df, target_col, feature_cols, model_name, params):
    print(f"--- Running Imputation ---\nModel: {model_name} on target '{target_col}' with params {params}")
    model_map = {'Log-Linear': _run_log_linear_imputer, 'KNN': _run_knn_imputer,
                 'Random Forest': _run_random_forest_imputer, 'Ensemble': _run_ensemble_imputer}
    imputer_func = model_map.get(model_name)
    if imputer_func:
        return imputer_func(df, target_col, feature_cols, params)
    return df.copy(), {'mae': -1, 'rmse': -1, 'plot_data': pd.DataFrame()}


################################################################################
# PART 2: GUI WIDGETS AND PAGES (from original gui_core.py)
################################################################################

class DataFrameModel(QAbstractTableModel):
    """A Qt model for displaying pandas DataFrames."""

    def __init__(self, df: pd.DataFrame):
        super().__init__();
        self._df = df

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid(): return None
        val = self._df.iat[index.row(), index.column()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole): return "" if pd.isna(val) else str(val)
        if role == Qt.ItemDataRole.ForegroundRole and pd.isna(val): return QBrush(QColor("red"))
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole: return None
        return self._df.columns[section] if orientation == Qt.Orientation.Horizontal else str(self._df.index[section])

    def setDataFrame(self, df: pd.DataFrame):
        self.beginResetModel(); self._df = df; self.endResetModel()


class InspectCleanPage(QWidget):
    """The first page for loading, inspecting, and cleaning data."""

    def __init__(self, parent: "CobberImputeApp"):
        super().__init__(parent)
        self.main: "CobberImputeApp" = parent;
        self._original_df: Optional[pd.DataFrame] = None
        self.load_btn = QPushButton("Load Alkane Dataset");
        self.drop_btn = QPushButton("Drop incomplete rows");
        self.drop_btn.setEnabled(False)
        self.table = QTableView();
        self.report = QTextEdit(readOnly=True);
        self.fig = Figure(figsize=(5, 3));
        self.canvas = Canvas(self.fig)
        top = QHBoxLayout();
        top.addWidget(self.load_btn);
        top.addStretch();
        top.addWidget(self.drop_btn)
        lay = QVBoxLayout(self);
        lay.addLayout(top);
        lay.addWidget(self.table, 3);
        lay.addWidget(QLabel("Missing Data Report:"));
        lay.addWidget(self.report, 1);
        lay.addWidget(QLabel("Bias After List-wise Deletion:"));
        lay.addWidget(self.canvas, 2)
        self.load_btn.clicked.connect(self._on_load);
        self.drop_btn.clicked.connect(self._on_drop)

    def _missing_report(self, df: pd.DataFrame):
        pct = (df.isna().mean() * 100).round(1).astype(str) + " %";
        self.report.setPlainText("\n".join(f"{c}: {v}" for c, v in pct.items()))

    def _bias_plot(self, before: pd.DataFrame, after: pd.DataFrame):
        self.fig.clf();
        axes = self.fig.subplots(1, 2)
        plot_cols = [col for col in ("carbons", "branch number") if col in before.columns and col in after.columns]
        for ax, col in zip(axes, plot_cols):
            pct_before = before[col].value_counts(normalize=True) * 100;
            pct_after = after[col].value_counts(normalize=True) * 100
            diff = pct_after.subtract(pct_before, fill_value=0).sort_index()
            ax.bar(diff.index.astype(int), diff.values);
            ax.set_title(col);
            ax.set_xlabel(col);
            ax.set_ylabel("Δ% after drop")
        self.fig.tight_layout();
        self.canvas.draw()

    def _on_load(self):
        # MODIFIED: Look for data file in assets subdirectory
        data_dir = os.path.join(project_root, "assets")
        fname, _ = QFileDialog.getOpenFileName(self, "Open CSV", data_dir, "CSV files (*.csv)")
        if not fname: return
        try:
            df = pd.read_csv(fname);
            df.columns = df.columns.str.strip().str.lower()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not read CSV:\n{exc}"); return
        self._original_df = df.copy();
        self.main.reset_app_with_new_df(df);
        self.fig.clf();
        self.canvas.draw();
        self.drop_btn.setEnabled(True)
        self.main.statusBar().showMessage(f"Loaded {Path(fname).name} ({len(df)} rows)")

    def _on_drop(self):
        if self._original_df is None: return
        before = self._original_df;
        after = before.dropna()
        self.main.reset_app_with_new_df(after);
        self._bias_plot(before, after)
        self.main.statusBar().showMessage(f"Dropped {len(before) - len(after)} rows; {len(after)} remain")

    def refresh(self):
        if self.main.df is not None:
            self.table.setModel(DataFrameModel(self.main.df));
            self._missing_report(self.main.df);
            self.drop_btn.setEnabled(True)
            if self._original_df is None: self._original_df = self.main.df.copy()


class SingleImputeTab(QWidget):
    """A tab for imputing a single property."""


    def __init__(self, target_col: str, parent: "CobberImputeApp"):
        super().__init__(parent)
        self.main: "CobberImputeApp" = parent;
        self.target_col = target_col
        model_group = QGroupBox("1. Select Imputation Model");
        model_layout = QVBoxLayout()
        self.model_radios = {'Log-Linear': QRadioButton("Log-Linear Regression"),
                             'KNN': QRadioButton("K-Nearest Neighbors (KNN)"),
                             'Random Forest': QRadioButton("Random Forest"),
                             'Ensemble': QRadioButton("Ensemble (Average)")}
        self.model_radios['Log-Linear'].setChecked(True)
        for radio in self.model_radios.values(): model_layout.addWidget(radio); radio.toggled.connect(
            self._update_param_visibility)
        model_group.setLayout(model_layout)
        param_group = QGroupBox("2. Adjust Model Parameters");
        self.param_stack = QStackedWidget();
        form_layout = QFormLayout(param_group);
        form_layout.addWidget(self.param_stack)
        self.knn_panel = QWidget();
        knn_layout = QFormLayout(self.knn_panel);
        self.knn_k_spinbox = QSpinBox();
        self.knn_k_spinbox.setRange(1, 100);
        self.knn_k_spinbox.setValue(5);
        knn_layout.addRow("k (Neighbors):", self.knn_k_spinbox)
        self.rf_panel = QWidget();
        rf_layout = QFormLayout(self.rf_panel);
        self.rf_depth_spinbox = QSpinBox();
        self.rf_depth_spinbox.setRange(1, 100);
        self.rf_depth_spinbox.setValue(10);
        rf_layout.addRow("Max Tree Depth:", self.rf_depth_spinbox)
        self.param_stack.addWidget(QWidget());
        self.param_stack.addWidget(self.knn_panel);
        self.param_stack.addWidget(self.rf_panel);
        self.param_stack.addWidget(QWidget())
        self.run_btn = QPushButton("Run Imputation");

        # --- BRANDING COLOR DEFINITION ---
        self.cobber_maroon = QColor(108, 29, 69)
        # --- BRANDING ---
        button_style = f"""
            QPushButton {{
                border: 2px solid {self.cobber_maroon.name()};
                font-size: 16px;
                font-weight: 600;
                padding: 10px 14px;
                border-radius: 6px;
                background-color: white;
            }}
            QPushButton:hover {{
                background-color: #FFD700;
            }}
        """
        self.run_btn.setStyleSheet(button_style)

        #self.run_btn.setStyleSheet(
        #    f"QPushButton {{ background-color: {self.main.cobber_gold.name()}; color: {self.main.cobber_maroon.name()}; font-weight: bold; padding: 8px; border-radius: 5px;}}")
        self.run_btn.clicked.connect(self._on_run)
        controls_layout = QVBoxLayout();
        controls_layout.addWidget(model_group);
        controls_layout.addWidget(param_group);
        controls_layout.addWidget(self.run_btn);
        controls_layout.addStretch()
        results_group = QGroupBox("Results");
        results_layout = QVBoxLayout(results_group)
        self.mae_label = QLabel("<b>Mean Absolute Error (MAE):</b> -")
        self.rmse_label = QLabel("<b>Root Mean Squared Error (RMSE):</b> -")
        self.plot_tabs = QTabWidget();
        self.quality_fig = Figure(figsize=(5, 4));
        self.quality_canvas = Canvas(self.quality_fig)
        self.carbon_fig = Figure(figsize=(5, 4));
        self.carbon_canvas = Canvas(self.carbon_fig);
        self.branch_fig = Figure(figsize=(5, 4));
        self.branch_canvas = Canvas(self.branch_fig)
        self.plot_tabs.addTab(self.quality_canvas, "Prediction Quality");
        self.plot_tabs.addTab(self.carbon_canvas, "Bias vs. Carbons");
        self.plot_tabs.addTab(self.branch_canvas, "Bias vs. Branching")
        results_layout.addWidget(self.mae_label);
        results_layout.addWidget(self.rmse_label);
        results_layout.addWidget(self.plot_tabs)
        main_layout = QHBoxLayout(self);
        splitter = QSplitter(Qt.Orientation.Horizontal);
        left_widget = QWidget();
        left_widget.setLayout(controls_layout)
        splitter.addWidget(left_widget);
        splitter.addWidget(results_group);
        splitter.setSizes([350, 650]);
        main_layout.addWidget(splitter)
        self._update_param_visibility()

    def _update_param_visibility(self):
        if self.model_radios['KNN'].isChecked():
            self.param_stack.setCurrentIndex(1)
        elif self.model_radios['Random Forest'].isChecked():
            self.param_stack.setCurrentIndex(2)
        else:
            self.param_stack.setCurrentIndex(0)

    def _on_run(self):
        if self.main._pristine_df is None: QMessageBox.warning(self, "No Data", "Load data first."); return
        model_name = [name for name, radio in self.model_radios.items() if radio.isChecked()][0]
        params = {'k': self.knn_k_spinbox.value(), 'max_depth': self.rf_depth_spinbox.value()}
        feature_cols = ['carbons', 'branch number']
        imputed_df, results = run_imputation_model(self.main._pristine_df, self.target_col, feature_cols, model_name,
                                                   params)
        self.mae_label.setText(f"Mean Absolute Error (MAE): {results['mae']:.4f}");
        self.rmse_label.setText(f"Root Mean Squared Error (RMSE): {results['rmse']:.4f}")
        plot_df = results.get('plot_data');
        self.quality_fig.clf();
        self.carbon_fig.clf();
        self.branch_fig.clf()
        if plot_df is not None and not plot_df.empty:
            with np.errstate(divide='ignore', invalid='ignore'):
                relative_error = 100 * (plot_df['predicted'] - plot_df['actual']) / plot_df[
                    'actual']; relative_error.replace([np.inf, -np.inf], np.nan, inplace=True)
            ax1 = self.quality_fig.add_subplot(111);
            ax1.scatter(plot_df['actual'], plot_df['predicted'], alpha=0.7)
            lims = [min(ax1.get_xlim()[0], ax1.get_ylim()[0]), max(ax1.get_xlim()[1], ax1.get_ylim()[1])];
            ax1.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
            ax1.set_xlabel("Actual Values");
            ax1.set_ylabel("Predicted Values");
            ax1.set_title("Prediction Quality")
            ax2 = self.carbon_fig.add_subplot(111);
            ax2.scatter(plot_df['carbons'], relative_error, alpha=0.7);
            ax2.axhline(0, color='r', linestyle='--');
            ax2.set_xlabel("Number of Carbons");
            ax2.set_ylabel("Relative Error (%)");
            ax2.set_title("Bias vs. Carbon Count")
            ax3 = self.branch_fig.add_subplot(111);
            ax3.scatter(plot_df['branch number'], relative_error, alpha=0.7);
            ax3.axhline(0, color='r', linestyle='--');
            ax3.set_xlabel("Branch Number");
            ax3.set_ylabel("Relative Error (%)");
            ax3.set_title("Bias vs. Branching")
        else:
            for fig in [self.quality_fig, self.carbon_fig, self.branch_fig]: ax = fig.add_subplot(111); ax.text(0.5,
                                                                                                                0.5,
                                                                                                                "Not enough data.",
                                                                                                                ha='center',
                                                                                                                va='center')
        for fig, canvas in [(self.quality_fig, self.quality_canvas), (self.carbon_fig, self.carbon_canvas),
                            (self.branch_fig, self.branch_canvas)]: fig.tight_layout(); canvas.draw()
        self.main.df = imputed_df
        self.main.statusBar().showMessage(f"Imputed '{self.target_col}' using {model_name}. MAE: {results['mae']:.4f}")
        inspect_page = self.main.stack.widget(0)
        if inspect_page and hasattr(inspect_page, 'refresh'): inspect_page.refresh()

    def refresh(self, clear_results=True):
        if clear_results:
            self.mae_label = QLabel("<b>Mean Absolute Error (MAE):</b> -")
            self.rmse_label = QLabel("<b>Root Mean Squared Error (RMSE):</b> -")
            for fig, canvas in [(self.quality_fig, self.quality_canvas), (self.carbon_fig, self.carbon_canvas),
                                (self.branch_fig, self.branch_canvas)]:
                fig.clf();
                ax = fig.add_subplot(111);
                ax.text(0.5, 0.5, "Ready to plot results.", ha='center', va='center');
                canvas.draw()
        df_exists = self.main.df is not None;
        col_exists = False
        if df_exists: col_exists = self.target_col in self.main.df.columns
        self.run_btn.setDisabled(not (df_exists and col_exists))


class ImputePage(QWidget):
    def __init__(self, parent: "CobberImputeApp"):
        super().__init__(parent);
        self.main: "CobberImputeApp" = parent
        layout = QVBoxLayout(self);
        self.tabs = QTabWidget();
        layout.addWidget(self.tabs)
        self.prop_map = {"Viscosity": "viscosity", "Heat Capacity": "heat capacity",
                         "Thermal Conductivity": "thermal conductivity"}

    def refresh(self):
        self.tabs.clear()
        if self.main.df is not None:
            for tab_name, col_name in self.prop_map.items():
                if col_name in self.main.df.columns:
                    self.tabs.addTab(SingleImputeTab(target_col=col_name, parent=self.main), tab_name)
                else:
                    placeholder = QWidget();
                    lbl = QLabel(f"Column '{col_name}' not found in CSV.");
                    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    vbox = QVBoxLayout(placeholder);
                    vbox.addWidget(lbl);
                    self.tabs.addTab(placeholder, tab_name);
                    self.tabs.setTabEnabled(self.tabs.count() - 1, False)
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if hasattr(tab, "refresh"): tab.refresh()


class SaveExportPage(QWidget):
    def __init__(self, parent: "CobberImputeApp"):
        super().__init__(parent);
        self.main: "CobberImputeApp" = parent
        lay = QVBoxLayout(self);
        self.save_btn = QPushButton("Save current CSV…");
        lay.addWidget(self.save_btn);
        lay.addStretch();
        self.save_btn.clicked.connect(self._on_save)

    def _on_save(self):
        if self.main.df is None: QMessageBox.warning(self, "No data", "Load data first."); return
        fname, _ = QFileDialog.getSaveFileName(self, "Save CSV", "imputed_alkane_dataset.csv", "CSV files (*.csv)")
        if fname: self.main.df.to_csv(fname, index=False); self.main.statusBar().showMessage(f"Saved to {fname}")


class CorrelationPage(QWidget):
    def __init__(self, parent: "CobberImputeApp"):
        super().__init__(parent);
        self.main: "CobberImputeApp" = parent
        self.generate_btn = QPushButton("Generate Correlation Matrix");
        self.generate_btn.setEnabled(False)
        self.fig = Figure(figsize=(8, 6), dpi=100);
        self.canvas = Canvas(self.fig)
        layout = QVBoxLayout(self);
        button_layout = QHBoxLayout();
        button_layout.addWidget(self.generate_btn);
        button_layout.addStretch()
        layout.addLayout(button_layout);
        layout.addWidget(self.canvas);
        self.generate_btn.clicked.connect(self._on_generate)

    def _on_generate(self):
        if self.main.df is None: QMessageBox.warning(self, "No Data", "Please load a dataset first."); return
        numeric_df = self.main.df.select_dtypes(include=np.number)
        cols_to_exclude = [col for col in numeric_df.columns if '_imputed' in col or col == 'id']
        final_df = numeric_df.drop(columns=cols_to_exclude)
        if final_df.empty: QMessageBox.warning(self, "No Numeric Data",
                                               "No numeric data available to create a correlation matrix."); return
        corr_matrix = final_df.corr();
        self.fig.clf();
        ax = self.fig.add_subplot(111)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", linewidths=.5, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right');
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        self.fig.tight_layout();
        self.canvas.draw();
        self.main.statusBar().showMessage("Correlation matrix generated.")

    def refresh(self):
        data_loaded = self.main.df is not None;
        self.generate_btn.setEnabled(data_loaded)
        self.fig.clf();
        ax = self.fig.add_subplot(111);
        ax.text(0.5, 0.5, "Load data and click 'Generate' to see the correlation matrix.", ha='center', va='center',
                wrap=True, fontsize=12);
        self.canvas.draw()


################################################################################
# PART 3: MAIN APPLICATION CLASS
################################################################################

class CobberImputeApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.cobber_maroon = QColor(108, 29, 69);
        self.cobber_gold = QColor(234, 170, 0);
        self.lato_font = QFont("Lato")
        self.setWindowTitle("CobberImpute");
        self.resize(1250, 780);
        self.setFont(self.lato_font)
        self.df: Optional[pd.DataFrame] = None;
        self._pristine_df: Optional[pd.DataFrame] = None
        splitter = QSplitter(Qt.Orientation.Horizontal);
        self.nav = QListWidget();
        self.stack = QStackedWidget()
        splitter.addWidget(self.nav);
        splitter.addWidget(self.stack);
        splitter.setStretchFactor(1, 1);
        self.setCentralWidget(splitter)

        # Instantiate and add all pages
        pages = [("Inspect & Clean", InspectCleanPage(self)), ("Impute Properties", ImputePage(self)),
                 ("Correlation", CorrelationPage(self)), ("Save / Export", SaveExportPage(self))]
        for name, widget in pages: self.nav.addItem(name); self.stack.addWidget(widget)

        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex);
        self.nav.setCurrentRow(0)
        self.setStatusBar(QStatusBar());
        self.statusBar().showMessage("Ready – load a CSV to begin")
        file_menu = self.menuBar().addMenu("File");
        quit_act = QAction("Quit", self);
        quit_act.triggered.connect(self.close);
        file_menu.addAction(quit_act)

    def reset_app_with_new_df(self, df: pd.DataFrame) -> None:
        self.df = df;
        self._pristine_df = df.copy()
        for i in range(self.stack.count()):
            page = self.stack.widget(i)
            if hasattr(page, "refresh"): page.refresh()
        self.statusBar().showMessage(f"DataFrame updated ({len(df)} rows)")


# --- Standalone Execution Guard ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = CobberImputeApp()
    win.show()
    sys.exit(app.exec())