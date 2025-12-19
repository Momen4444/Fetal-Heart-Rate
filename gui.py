import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wfdb

# PyQt6 Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                             QFileDialog, QTabWidget, QFrame, QSplitter, QMessageBox, QSpinBox)
from PyQt6.QtCore import Qt
from PyQt6.QtWebEngineWidgets import QWebEngineView

from signal_processing import SignalProcessor
from medical_interpreter import MedicalInterpreter

# ==========================================
# FIX FOR WINDOWS RENDERING ERROR (0x80004002)
# ==========================================
# This flag disables GPU acceleration for the embedded browser, preventing 
# the "IDCompositionDevice4" crash on some Windows systems.
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"

class HeartMonitorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cardiac Health Analysis System (PyQt6 + Plotly)")
        self.resize(1200, 850)
        
        self.processor = SignalProcessor()
        self.interpreter = MedicalInterpreter()
        
        # Load CTU Features for ID Lookup
        try:
            # Try multiple paths
            paths = [
                'ctu_df.csv',
                'CTU-CHB-Intrapartum-Cardiotocography-Caesarean-Section-Prediction/database/ctu_df.csv',
                'database/ctu_df.csv'
            ]
            self.ctu_features_df = None
            for p in paths:
                if os.path.exists(p):
                    self.ctu_features_df = pd.read_csv(p)
                    print(f"Loaded ctu_df.csv from {p}")
                    break
            
            if self.ctu_features_df is not None:
                # Ensure ID is string for matching
                self.ctu_features_df['ID'] = self.ctu_features_df['ID'].astype(str)
            else:
                print("Error: ctu_df.csv not found in common locations.")
                
        except Exception as e:
            print(f"Error loading ctu_df.csv: {e}")
            self.ctu_features_df = None
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        self.init_adult_tab()
        self.init_fetal_tab()
        
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QTextEdit { background-color: white; font-size: 12px; border: 1px solid #ccc; }
            QLabel { font-weight: bold; font-size: 14px; margin-top: 5px; }
            QPushButton { padding: 8px; font-weight: bold; background-color: #0078d7; color: white; border-radius: 4px; }
            QPushButton:hover { background-color: #005a9e; }
        """)

    def init_adult_tab(self):
        self.tab_adult = QWidget()
        self.tabs.addTab(self.tab_adult, "Adult Heart Failure (HRV)")
        
        layout = QHBoxLayout(self.tab_adult)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # -- Left: Plotly Web View --
        self.web_adult = QWebEngineView()
        splitter.addWidget(self.web_adult)
        
        # -- Right: Controls & Report --
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setFixedWidth(350)
        
        # Controls
        right_layout.addWidget(QLabel("Data Simulation"))
        btn_norm = QPushButton("Simulate Healthy")
        btn_norm.clicked.connect(lambda: self.generate_adult_data("healthy"))
        right_layout.addWidget(btn_norm)
        
        btn_fail = QPushButton("Simulate Heart Failure")
        btn_fail.clicked.connect(lambda: self.generate_adult_data("failure"))
        right_layout.addWidget(btn_fail)
        
        right_layout.addSpacing(10)
        right_layout.addWidget(QLabel("File Loader"))
        btn_load = QPushButton("Load Data (.csv / .hea)")
        btn_load.clicked.connect(self.load_adult_data)
        btn_load.setStyleSheet("background-color: #555;")
        right_layout.addWidget(btn_load)
        
        right_layout.addSpacing(20)
        
        # Reports
        right_layout.addWidget(QLabel("Metrics"))
        self.txt_adult_metrics = QTextEdit()
        self.txt_adult_metrics.setReadOnly(True)
        self.txt_adult_metrics.setMaximumHeight(100)
        right_layout.addWidget(self.txt_adult_metrics)
        
        right_layout.addWidget(QLabel("Processing Diagnostic"))
        self.txt_adult_rules = QTextEdit()
        self.txt_adult_rules.setReadOnly(True)
        self.txt_adult_rules.setStyleSheet("background-color: #e6f3ff;")
        right_layout.addWidget(self.txt_adult_rules)

        right_layout.addWidget(QLabel("AI Confidence"))
        self.txt_adult_ai = QTextEdit()
        self.txt_adult_ai.setReadOnly(True)
        self.txt_adult_ai.setMaximumHeight(80)
        self.txt_adult_ai.setStyleSheet("background-color: #e6ffe6;")
        right_layout.addWidget(self.txt_adult_ai)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 350])

    def init_fetal_tab(self):
        self.tab_fetal = QWidget()
        self.tabs.addTab(self.tab_fetal, "Fetal Distress (CTG)")
        
        layout = QHBoxLayout(self.tab_fetal)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        self.web_fetal = QWebEngineView()
        splitter.addWidget(self.web_fetal)
        
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setFixedWidth(350)
        
        right_layout.addWidget(QLabel("Signal Simulation"))
        
        btn_norm = QPushButton("Simulate Normal Fetus")
        btn_norm.clicked.connect(lambda: self.generate_fetal_data("normal"))
        right_layout.addWidget(btn_norm)
        
        btn_distress = QPushButton("Simulate Fetal Distress")
        btn_distress.clicked.connect(lambda: self.generate_fetal_data("distress"))
        right_layout.addWidget(btn_distress)
        
        right_layout.addSpacing(10)
        btn_load = QPushButton("Load Signal File (.hea)")
        btn_load.clicked.connect(self.load_fetal_data)
        btn_load.setStyleSheet("background-color: #555;")
        right_layout.addWidget(btn_load)
        
        right_layout.addSpacing(20)
        
        right_layout.addWidget(QLabel("CTG Metrics"))
        self.txt_fetal_metrics = QTextEdit()
        self.txt_fetal_metrics.setReadOnly(True)
        self.txt_fetal_metrics.setMaximumHeight(100)
        right_layout.addWidget(self.txt_fetal_metrics)
        
        right_layout.addWidget(QLabel("Processing Diagnostic"))
        self.txt_fetal_rules = QTextEdit()
        self.txt_fetal_rules.setReadOnly(True)
        self.txt_fetal_rules.setStyleSheet("background-color: #fff0f0;")
        right_layout.addWidget(self.txt_fetal_rules)

        right_layout.addWidget(QLabel("AI Confidence"))
        self.txt_fetal_ai = QTextEdit()
        self.txt_fetal_ai.setReadOnly(True)
        self.txt_fetal_ai.setMaximumHeight(80)
        self.txt_fetal_ai.setStyleSheet("background-color: #e6ffe6;")
        right_layout.addWidget(self.txt_fetal_ai)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 350])

    # --- Logic ---

    def generate_adult_data(self, condition):
        np.random.seed(42)
        n_beats = 1000
        if condition == "healthy":
            base = 800
            noise = np.random.normal(0, 70, n_beats)
            t = np.linspace(0, 100, n_beats)
            rsa = 50 * np.sin(2 * np.pi * 0.25 * t)
            data = base + noise + rsa
        else:
            base = 600
            noise = np.random.normal(0, 15, n_beats)
            t = np.linspace(0, 100, n_beats)
            rsa = 10 * np.sin(2 * np.pi * 0.25 * t)
            data = base + noise + rsa
        
        self.update_adult_view(data)

    def generate_fetal_data(self, condition):
        np.random.seed(42)
        fs = 4
        points = 10 * 60 * fs
        if condition == "normal":
            baseline = 140
            var = np.random.normal(0, 5, points)
            accel = np.zeros(points)
            accel[1000:1200] = 20
            data = baseline + var + accel
        else:
            baseline = 100
            var = np.random.normal(0, 1, points)
            data = baseline + var
            
        self.update_fetal_view(data)

    def load_adult_data(self):
        # Allow CSV or WFDB Header files
        path, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "Data Files (*.csv *.hea);;CSV (*.csv);;WFDB Header (*.hea)")
        if not path:
            return

        try:
            if path.endswith('.csv'):
                df = pd.read_csv(path)
                col = next((c for c in ['rr', 'RR', 'interval'] if c in df.columns), None)
                if col: 
                    self.update_adult_view(df[col].values)
                else:
                    self.txt_adult_metrics.setText("Error: CSV must contain 'RR' or 'interval' column.")
            
            elif path.endswith('.hea'):
                # Handle WFDB files
                record_name = path[:-4] # remove .hea extension
                
                # 1. Try to read Annotations first (Best for accuracy)
                try:
                    # BIDMC often uses .ecg extension for annotations
                    ann = wfdb.rdann(record_name, 'ecg')
                    # Calculate RR intervals from annotation samples (time difference between beats)
                    # Convert samples to ms: (diff / fs) * 1000
                    rr_samples = np.diff(ann.sample)
                    rr_intervals_ms = (rr_samples / ann.fs) * 1000.0
                    
                    self.txt_adult_metrics.setText(f"Loaded Annotations: {len(rr_intervals_ms)} beats found.")
                    self.update_adult_view(rr_intervals_ms)
                    return
                except Exception:
                    print("No annotation file found, falling back to raw signal processing...")

                # 2. Fallback: Read Raw Signal and detect Peaks
                # Limit to 20 minutes (approx 300k samples at 250Hz) to prevent freezing
                record = wfdb.rdrecord(record_name, sampto=300000)
                signal_data = record.p_signal[:, 0] # Use channel 0
                fs = record.fs
                
                # Preprocess ECG (Notch + Bandpass)
                clean_signal = SignalProcessor.preprocess_ecg(signal_data, fs)
                
                # Detect R-peaks using Pan-Tompkins
                peaks = SignalProcessor.pan_tompkins_detector(clean_signal, fs)
                
                if len(peaks) < 10:
                    self.txt_adult_metrics.setText("Error: Signal too noisy or no beats detected.")
                    return

                # Calculate RR intervals
                rr_samples = np.diff(peaks)
                rr_intervals_ms = (rr_samples / fs) * 1000.0
                
                self.txt_adult_metrics.setText(f"Raw Signal Processed (Pan-Tompkins). Detected {len(peaks)} peaks.")
                self.update_adult_view(rr_intervals_ms)

        except Exception as e:
            self.txt_adult_metrics.setText(f"Error loading file: {str(e)}")
            print(e)

    def load_fetal_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Fetal Data", "", "Data Files (*.csv *.hea);;CSV (*.csv);;WFDB Header (*.hea)")
        if not path:
            return

        try:
            # Extract ID from filename (e.g., "1001.csv" -> "1001")
            file_id = os.path.splitext(os.path.basename(path))[0]
            
            # 1. Load Signal for Plotting
            fhr_signal = None
            if path.endswith('.csv'):
                df = pd.read_csv(path)
                col = next((c for c in ['fhr', 'FHR', 'bpm'] if c in df.columns), None)
                if col: 
                    fhr_signal = df[col].values
                else:
                    self.txt_fetal_metrics.setText("Error: CSV must contain 'FHR' or 'bpm' column.")
                    return
            
            elif path.endswith('.hea'):
                # Handle CTU-CHB PhysioNet files
                record_name = path[:-4]
                try:
                    record = wfdb.rdrecord(record_name)
                    if record.p_signal is None:
                        self.txt_fetal_metrics.setText("Error: No signal data found in record.")
                        return
                    fhr_signal = record.p_signal[:, 0]
                    # Preprocessing: Replace 0s with NaN or interpolate
                    fhr_signal = fhr_signal.astype(float)
                    fhr_signal[fhr_signal == 0] = np.nan
                    nans = np.isnan(fhr_signal)
                    not_nans = ~nans
                    if np.sum(not_nans) > 0:
                        fhr_signal[nans] = np.interp(nans.nonzero()[0], not_nans.nonzero()[0], fhr_signal[not_nans])
                except Exception as e:
                    self.txt_fetal_metrics.setText(f"Error reading WFDB record: {str(e)}")
                    return

            # 2. Lookup Features in ctu_df.csv using ID
            precomputed_features = None
            if self.ctu_features_df is not None:
                row = self.ctu_features_df[self.ctu_features_df['ID'] == file_id]
                if not row.empty:
                    # Found ID, extract features for model
                    if self.interpreter.feature_names:
                        try:
                            # Use features defined during training
                            precomputed_features = row[self.interpreter.feature_names]
                            self.txt_fetal_metrics.setText(f"Loaded Signal & Features for ID: {file_id}\n(Using {len(self.interpreter.feature_names)} features)")
                        except KeyError as e:
                            self.txt_fetal_metrics.setText(f"Error: Database missing features: {e}")
                    else:
                        # Fallback to basic features if no names loaded
                        precomputed_features = pd.DataFrame([{
                            'Mean': row['Mean_FHR'].values[0],
                            'Median': row['Median_FHR'].values[0],
                            'Std': row['Std_FHR'].values[0],
                            'Max': row['Peak_FHR'].values[0]
                        }])
                        self.txt_fetal_metrics.setText(f"Loaded Signal & Features for ID: {file_id}\n(Using basic features)")
                else:
                    self.txt_fetal_metrics.setText(f"Loaded Signal for ID: {file_id}\nWarning: ID not found in ctu_df.csv. Using signal metrics.")
            
            # 3. Update View
            if fhr_signal is not None:
                self.update_fetal_view(fhr_signal, precomputed_features=precomputed_features)

        except Exception as e:
            self.txt_fetal_metrics.setText(f"Error: {str(e)}")

    # --- Plotly Rendering ---

    def update_adult_view(self, rr_data):
        # Filter artifacts (physiologically impossible values: <300ms or >2000ms)
        rr_data = rr_data[(rr_data > 300) & (rr_data < 2000)]
        
        if len(rr_data) < 10:
            self.txt_adult_metrics.setText("Error: Not enough valid heartbeats.")
            return

        metrics = self.processor.calculate_hrv_metrics(rr_data)
        
        # Optimization: Downsample
        max_points = 2000
        step = max(1, len(rr_data) // max_points)
        plot_data = rr_data[::step]
        
        fig = make_subplots(rows=2, cols=1, 
                            subplot_titles=("RR Intervals (Time Domain)", "Power Spectral Density (Freq Domain)"),
                            vertical_spacing=0.15)
        
        fig.add_trace(go.Scatter(y=plot_data, mode='lines', name='RR Interval', 
                                 line=dict(color='blue', width=1), hovertemplate='%{y:.1f} ms'), row=1, col=1)
        
        freqs, psd = metrics['PSD_Freqs'], metrics['PSD_Power']
        fig.add_trace(go.Scatter(x=freqs, y=psd, mode='lines', name='PSD', 
                                 line=dict(color='green'), fill='tozeroy'), row=2, col=1)
        
        fig.add_vrect(x0=0.04, x1=0.15, fillcolor="yellow", opacity=0.2, annotation_text="LF", row=2, col=1)
        fig.add_vrect(x0=0.15, x1=0.40, fillcolor="red", opacity=0.2, annotation_text="HF", row=2, col=1)
        
        fig.update_layout(height=700, showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
        fig.update_xaxes(title_text="Beat Number", row=1, col=1)
        fig.update_yaxes(title_text="Interval (ms)", row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", range=[0, 0.5], row=2, col=1)
        fig.update_yaxes(title_text="Power", row=2, col=1)

        html = fig.to_html(include_plotlyjs='cdn')
        self.web_adult.setHtml(html)
        
        self.txt_adult_metrics.setText(
            f"SDNN:  {metrics['SDNN']:.2f} ms\n"
            f"RMSSD: {metrics['RMSSD']:.2f} ms\n"
            f"Mean HR: {metrics['Mean HR']:.1f} bpm\n"
            f"LF/HF Ratio: {metrics['LF/HF Ratio']:.2f}\n"
            f"VLF Power: {metrics['VLF Power']:.2f}\n"
            f"SD1: {metrics['SD1']:.2f} | SD2: {metrics['SD2']:.2f}\n"
            f"SampEn: {metrics['SampEn']:.3f}"
        )
        rules, ai = self.interpreter.interpret_adult_hrv(metrics)
        self.txt_adult_rules.setText(rules)
        self.txt_adult_ai.setText(ai)

    def update_fetal_view(self, fhr_data, precomputed_features=None):
        # 1. Plot Signal if available
        metrics = {}
        if fhr_data is not None:
            # clean_data = self.processor.preprocess_signal(fhr_data)
            metrics = self.processor.calculate_ctg_metrics(fhr_data)
            
            if metrics:
                clean_data = metrics['Clean_Signal'] # Use the internally cleaned signal for plotting
                max_points = 2000
                step = max(1, len(clean_data) // max_points)
                plot_data = clean_data[::step]

                fig = go.Figure()
                full_duration_min = len(clean_data) / 4.0 / 60.0
                time_axis = np.linspace(0, full_duration_min, len(plot_data))
                
                fig.add_trace(go.Scatter(x=time_axis, y=plot_data, mode='lines', name='FHR',
                                         line=dict(color='purple', width=1.5)))
                
                fig.add_hrect(y0=110, y1=160, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Normal Range")
                
                fig.update_layout(
                    title="Cardiotocography (FHR Trace)",
                    xaxis_title="Time (Minutes)",
                    yaxis_title="Beats Per Minute",
                    yaxis_range=[50, 200],
                    height=600,
                    hovermode="x unified"
                )
                html = fig.to_html(include_plotlyjs='cdn')
                self.web_fetal.setHtml(html)
            else:
                self.web_fetal.setHtml("<h3>Error: Signal too short or empty.</h3>")
        
        # 2. Prepare Metrics for Report & AI
        # If we have precomputed features (from ID lookup), use them for AI
        # Otherwise use calculated metrics
        
        ai_features = None
        report_text = ""
        
        if precomputed_features is not None:
            ai_features = precomputed_features
            row = precomputed_features.iloc[0]
            report_text += "--- Features from Database (ID Lookup) ---\n"
            
            # Display key features if available
            display_cols = ['Mean_FHR', 'Median_FHR', 'Std_FHR', 'Peak_FHR', 'Gest. Weeks', 'Age', 'Weight(g)']
            found_any = False
            for col in display_cols:
                if col in row:
                    val = row[col]
                    if isinstance(val, float):
                        report_text += f"{col}: {val:.2f}\n"
                    else:
                        report_text += f"{col}: {val}\n"
                    found_any = True
            
            if not found_any:
                # Fallback: print first 5
                for k, v in list(row.items())[:5]:
                    report_text += f"{k}: {v}\n"
            
            report_text += f"... ({len(row)} features used for AI)\n"
        
        if metrics:
            report_text += "\n--- Signal Analysis ---\n"
            report_text += f"Baseline: {metrics['Baseline']:.1f} bpm\n"
            report_text += f"Variability (STV): {metrics['STV']:.2f}\n"
            report_text += f"Variability (LTV): {metrics['LTV']:.2f}\n"
            report_text += f"Accelerations: {metrics['Acc_Count']} ({metrics['Acc_Time']:.1f}s)\n"
            report_text += f"Decelerations: {metrics['Dec_Count']} ({metrics['Dec_Time']:.1f}s)\n"
            report_text += f"Sinusoidal Idx: {metrics['Sinusoidal_Idx']:.3f}"
            
            if ai_features is None:
                ai_features = metrics.get('ML_Features')
                report_text += "\n(Using calculated features for AI)"

        self.txt_fetal_metrics.setText(report_text)

        # 3. Run Interpreter
        # We pass the AI features inside the metrics dict for the interpreter
        interp_metrics = metrics.copy() if metrics else {}
        if ai_features is not None:
            interp_metrics['ML_Features'] = ai_features
            
        rules, ai = self.interpreter.interpret_fetal_ctg(interp_metrics)
        self.txt_fetal_rules.setText(rules)
        self.txt_fetal_ai.setText(ai)
