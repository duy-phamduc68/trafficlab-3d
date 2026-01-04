import os
import glob
import json
import yaml
import shutil
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSlot, QSize
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QProgressBar, QTextEdit, QFileDialog, QGroupBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QMessageBox, QSplitter, QCheckBox,
    QDialog, QDialogButtonBox
)
from PyQt5.QtWidgets import QComboBox

# Import the worker engine
from gui.core.x_engine import InferenceSession

class InferenceTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # --- State ---
        self.config_path = "inference_config.yaml"
        self.root_loc_dir = "location"
        self.root_out_dir = "output"
        self.selected_config = None
        self.processing_queue = []  # List of dicts built when Start is clicked
        self.current_worker = None
        self.worker_thread = None
        self.is_processing = False
        
        self._init_ui()
        self._load_defaults()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # --- TOP: Configuration Area ---
        config_group = QGroupBox("Configuration & Plan")
        cg_layout = QVBoxLayout()
        
        # Config Selector Row
        row1 = QHBoxLayout()
        self.lbl_config = QLabel(f"Config: {self.config_path}")
        btn_reload = QPushButton("Reload Config")
        btn_reload.clicked.connect(self._load_defaults)
        # Config picker for multi-config YAML
        self.cfg_picker = QComboBox()
        self.cfg_picker.setVisible(False)
        self.cfg_picker.currentIndexChanged.connect(self._on_config_selected)

        btn_edit = QPushButton("Edit Config")
        btn_edit.clicked.connect(self._toggle_editor)

        btn_edit_meas = QPushButton("Edit Measurements")
        btn_edit_meas.clicked.connect(self._toggle_measurements_editor)

        row1.addWidget(self.lbl_config)
        row1.addWidget(self.cfg_picker)
        row1.addWidget(btn_reload)
        row1.addWidget(btn_edit)
        row1.addWidget(btn_edit_meas)
        cg_layout.addLayout(row1)
        
        # Info Label (Model/Tracker summary)
        self.lbl_info = QLabel("Loaded: None")
        self.lbl_info.setStyleSheet("color: #888; font-style: italic;")
        cg_layout.addWidget(self.lbl_info)

        # Note: Editor is a pop-up dialog (created on demand)
        
        # Actions Row
        row2 = QHBoxLayout()
        self.btn_lock = QPushButton("Scan")
        self.btn_lock.clicked.connect(self.on_lock_clicked)
        self.btn_lock.setStyleSheet("background-color: #2c3e50; color: white; font-weight: bold; padding: 5px;")
        
        self.btn_wipe = QPushButton("Wipe Output")
        self.btn_wipe.clicked.connect(self.on_wipe_clicked)
        self.btn_wipe.setStyleSheet("background-color: #7f2c2c; color: white;")
        
        # Select / Unselect controls for Run column
        self.btn_select_all = QPushButton("Select all")
        self.btn_select_all.clicked.connect(self.on_select_all_clicked)
        self.btn_unselect_all = QPushButton("Unselect all")
        self.btn_unselect_all.clicked.connect(self.on_unselect_all_clicked)

        row2.addWidget(self.btn_lock)
        row2.addWidget(self.btn_select_all)
        row2.addWidget(self.btn_unselect_all)
        row2.addStretch()
        row2.addWidget(self.btn_wipe)
        cg_layout.addLayout(row2)
        
        config_group.setLayout(cg_layout)
        
        # --- MIDDLE: Task Table ---
        self.table = QTableWidget()
        # CHANGED: 4 Columns now (Checkbox, Location, Footage, Status)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Run", "Location", "Footage", "Status"])
        
        # Formatting: Checkbox column small fixed width
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.table.setColumnWidth(0, 50)
        
        # Formatting: Location and Status resize to contents, Footage stretches
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        
        # --- BOTTOM: Execution & Logs ---
        exec_group = QGroupBox("Execution")
        ex_layout = QVBoxLayout()
        
        # Buttons
        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start Inference")
        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_start.setEnabled(False)  # Disabled until locked
        self.btn_start.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold; font-size: 14px; padding: 8px;")
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #c0392b; color: white; font-weight: bold; font-size: 14px; padding: 8px;")
        
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        ex_layout.addLayout(btn_row)
        
        # Progress Bar
        self.pbar = QProgressBar()
        self.pbar.setValue(0)
        self.pbar.setTextVisible(True)
        self.pbar.setFormat("%p% - Waiting...")
        ex_layout.addWidget(self.pbar)
        
        # Log Text Area
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("font-family: Consolas; font-size: 10pt; background-color: #1e1e1e; color: #ddd;")
        ex_layout.addWidget(self.txt_log)
        
        exec_group.setLayout(ex_layout)
        
        # Splitter for resizeable areas
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(config_group)
        splitter.addWidget(self.table)
        splitter.addWidget(exec_group)
        splitter.setStretchFactor(1, 2) # Give table more space
        
        layout.addWidget(splitter)

    # ==========================================================
    # LOGIC: CONFIG & SCANNING
    # ==========================================================

    def _load_defaults(self):
        """Reloads the YAML config to update the summary label."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                # If the YAML uses the new multi-config layout, populate picker
                if isinstance(cfg, dict) and 'configs' in cfg and isinstance(cfg['configs'], dict):
                    keys = list(cfg['configs'].keys())
                    # Prevent signals while updating items and index
                    self.cfg_picker.blockSignals(True)
                    self.cfg_picker.clear()
                    self.cfg_picker.addItems(keys)
                    self.cfg_picker.setVisible(True)
                    # select previously selected or first
                    if self.selected_config in keys:
                        idx = keys.index(self.selected_config)
                        self.cfg_picker.setCurrentIndex(idx)
                        self.selected_config = keys[idx]
                    else:
                        self.selected_config = keys[0]
                        self.cfg_picker.setCurrentIndex(0)
                    self.cfg_picker.blockSignals(False)
                    selected_cfg = cfg['configs'][self.selected_config]
                    cname = self.selected_config
                else:
                    # single-config YAML
                    self.cfg_picker.setVisible(False)
                    selected_cfg = cfg
                    cname = selected_cfg.get('config_name', 'default')

                model = Path(selected_cfg.get('model', {}).get('weights', 'Unknown')).name
                tracker = selected_cfg.get('tracking', {}).get('tracker_type', 'Unknown')
                kine = selected_cfg.get('kinematics', {}).get('heading_smoothing', 'Unknown')
                measurements = selected_cfg.get('prior_dimensions', {}) if 'prior_dimensions' in selected_cfg else 'Unknown'

                info = (f"Config Name: <b>{cname}</b> | Model: {model} | "
                        f"Tracker: {tracker} | Measurements: {measurements} | ")
                self.lbl_info.setText(info)
                self.log(f"Loaded {self.config_path}")

                # Reset table/queue state when reloading
                self.btn_start.setEnabled(False)
                self.processing_queue.clear()
                self.table.setRowCount(0)
                
            except Exception as e:
                self.lbl_info.setText(f"Error loading YAML: {e}")
        else:
            self.lbl_info.setText("inference_config.yaml not found!")

    def _on_config_selected(self, idx):
        if not self.cfg_picker.isVisible():
            return
        sel = self.cfg_picker.currentText()
        if sel:
            # If selection didn't change, do nothing
            if sel == self.selected_config:
                return
            self.selected_config = sel
            # refresh the summary
            self._load_defaults()

    def _toggle_editor(self):
        # Open a modal dialog with YAML content for editing
        if not os.path.exists(self.config_path):
            QMessageBox.critical(self, "Error", "Config file not found.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Config YAML")
        dlg.setModal(True)
        dlg.resize(900, 600)

        layout = QVBoxLayout(dlg)
        txt = QTextEdit(dlg)
        txt.setLineWrapMode(QTextEdit.NoWrap)
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open YAML: {e}")
            return
        txt.setPlainText(content)
        layout.addWidget(txt)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def on_save():
            reply = QMessageBox.question(self, "Save YAML",
                                         "Overwrite the YAML file with edited content?",
                                         QMessageBox.Yes | QMessageBox.Cancel)
            if reply != QMessageBox.Yes:
                return
            try:
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    f.write(txt.toPlainText())
                self.log(f"Saved {self.config_path}")
                dlg.accept()
                # reload and refresh UI
                self._load_defaults()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save YAML: {e}")

        def on_cancel():
            dlg.reject()

        buttons.accepted.connect(on_save)
        buttons.rejected.connect(on_cancel)

        dlg.exec_()

    def _toggle_measurements_editor(self):
        # Open a modal dialog with prior_dimensions.json content for editing
        pd_path = "prior_dimensions.json"
        if not os.path.exists(pd_path):
            QMessageBox.critical(self, "Error", "prior_dimensions.json not found.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Measurements (prior_dimensions.json)")
        dlg.setModal(True)
        dlg.resize(900, 700)

        layout = QVBoxLayout(dlg)
        txt = QTextEdit(dlg)
        txt.setLineWrapMode(QTextEdit.NoWrap)
        try:
            with open(pd_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open prior_dimensions.json: {e}")
            return
        txt.setPlainText(content)
        layout.addWidget(txt)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)

        def on_save():
            # Validate JSON first
            txt_content = txt.toPlainText()
            try:
                parsed = json.loads(txt_content)
            except Exception as e:
                QMessageBox.critical(self, "Invalid JSON", f"The edited content is not valid JSON:\n{e}")
                return

            reply = QMessageBox.question(self, "Save Measurements",
                                         "Overwrite prior_dimensions.json with edited content?",
                                         QMessageBox.Yes | QMessageBox.Cancel)
            if reply != QMessageBox.Yes:
                return
            try:
                with open(pd_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed, f, indent=2)
                self.log(f"Saved {pd_path}")
                dlg.accept()
                # trigger reload of UI/configs
                self._load_defaults()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save prior_dimensions.json: {e}")

        def on_cancel():
            dlg.reject()

        buttons.accepted.connect(on_save)
        buttons.rejected.connect(on_cancel)

        dlg.exec_()

    def on_lock_clicked(self):
        """
        Scans folders and populates the Task Table with checkboxes.
        """
        if not os.path.exists(self.config_path):
            QMessageBox.critical(self, "Error", "Config file not found.")
            return

        with open(self.config_path, 'r') as f:
            current_raw = yaml.safe_load(f)

        # Support multi-config YAML format
        if isinstance(current_raw, dict) and 'configs' in current_raw and isinstance(current_raw['configs'], dict):
            # choose selected_config if set, else pick first
            keys = list(current_raw['configs'].keys())
            if self.selected_config and self.selected_config in keys:
                current_cfg = current_raw['configs'][self.selected_config]
                config_name = self.selected_config
            else:
                config_name = keys[0]
                current_cfg = current_raw['configs'][config_name]
        else:
            current_cfg = current_raw or {}
            config_name = current_cfg.get('config_name', 'default')

        model_stem = Path(current_cfg.get('model', {}).get('weights', 'Unknown')).stem
        tracker_type = current_cfg.get('tracking', {}).get('tracker_type', 'Unknown')
        
        # Define Expected Output Path
        base_out = os.path.join(
            self.root_out_dir, 
            f"model-{model_stem}_tracker-{tracker_type}", 
            config_name
        )
        
        # --- DIRTY CONFIG CHECK ---
        frozen_cfg_path = os.path.join(base_out, f"{config_name}.yaml")
        
        if os.path.exists(frozen_cfg_path):
            with open(frozen_cfg_path, 'r') as f:
                frozen_cfg = yaml.safe_load(f)
            
            if current_cfg != frozen_cfg:
                reply = QMessageBox.warning(
                    self, "Config Mismatch",
                    f"The configuration '{config_name}' already exists but differs from your current YAML.\n\n"
                    "Do you want to overwrite the old config definition?",
                    QMessageBox.Yes | QMessageBox.Cancel
                )
                if reply == QMessageBox.Cancel:
                    return
                else:
                    self.log("[WARN] User chose to overwrite config definition.")

        # --- SCAN FOOTAGE ---
        self.table.setRowCount(0)
        
        locs = [d for d in os.listdir(self.root_loc_dir) if os.path.isdir(os.path.join(self.root_loc_dir, d))]
        locs.sort()
        
        row_idx = 0
        
        for loc in locs:
            footage_dir = os.path.join(self.root_loc_dir, loc, "footage")
            if not os.path.exists(footage_dir): continue
            
            mp4s = glob.glob(os.path.join(footage_dir, "*.mp4"))
            mp4s.sort()
            
            for mp4 in mp4s:
                fname = os.path.basename(mp4)
                # Expected output file
                out_name = os.path.splitext(fname)[0] + ".json.gz"
                out_path = os.path.join(base_out, loc, out_name)
                
                status = "Pending"
                color = "#f39c12" # Orange
                should_check = True
                
                if os.path.exists(out_path):
                    status = "Done"
                    color = "#27ae60" # Green
                    should_check = False # Uncheck if done
                
                # Check for G-Projection
                g_proj_paths = [
                    os.path.join(self.root_loc_dir, loc, f"G_projection_{loc}.json"),
                    os.path.join(self.root_loc_dir, loc, f"G_projection_svg_{loc}.json")
                ]
                valid_g = next((p for p in g_proj_paths if os.path.exists(p)), None)
                
                if not valid_g:
                    status = "No G-Proj"
                    color = "#c0392b" # Red
                    should_check = False
                
                # --- POPULATE TABLE ---
                self.table.insertRow(row_idx)

                # Col 0: Checkbox (Using Cell Widget for Centering)
                chk_widget = QWidget()
                chk_layout = QHBoxLayout(chk_widget)
                chk_layout.setAlignment(Qt.AlignCenter)
                chk_layout.setContentsMargins(0, 0, 0, 0)
                chk = QCheckBox()
                chk.setChecked(should_check)
                chk_layout.addWidget(chk)
                self.table.setCellWidget(row_idx, 0, chk_widget)

                # Col 1: Location (We store the task data hidden here!)
                item_loc = QTableWidgetItem(loc)
                task_data = {
                    "loc": loc,
                    "mp4": mp4,
                    "g_proj": valid_g,
                    "table_row": row_idx
                }
                item_loc.setData(Qt.UserRole, task_data)
                self.table.setItem(row_idx, 1, item_loc)

                # Col 2: Footage
                self.table.setItem(row_idx, 2, QTableWidgetItem(fname))
                
                # Col 3: Status
                item_stat = QTableWidgetItem(status)
                item_stat.setForeground(pyqtColor(color))
                self.table.setItem(row_idx, 3, item_stat)
                
                row_idx += 1
        
        self.log(f"Scan complete. Found {row_idx} files.")
        self.btn_start.setEnabled(row_idx > 0)
        self.btn_start.setText(f"Start Inference")

    def on_wipe_clicked(self):
        """Nuclear Option: Delete the entire output folder structure for this config."""
        with open(self.config_path, 'r') as f:
            raw = yaml.safe_load(f)

        # support multi-config YAML
        if isinstance(raw, dict) and 'configs' in raw and isinstance(raw['configs'], dict):
            if self.selected_config and self.selected_config in raw['configs']:
                cfg = raw['configs'][self.selected_config]
                cname = self.selected_config
            else:
                # fallback
                first = next(iter(raw['configs'].keys()))
                cfg = raw['configs'][first]
                cname = first
        else:
            cfg = raw or {}
            cname = cfg.get('config_name', 'default')

        model = Path(cfg.get('model', {}).get('weights', 'Unknown')).stem
        tracker = cfg.get('tracking', {}).get('tracker_type', 'Unknown')
        
        target_dir = os.path.join(self.root_out_dir, f"model-{model}_tracker-{tracker}", cname)
        
        if not os.path.exists(target_dir):
            QMessageBox.information(self, "Info", "Nothing to wipe.")
            return

        text, ok = QFileDialog.getText(
            self, "CONFIRM WIPE", 
            f"Type 'DELETE' to wipe all data in:\n{target_dir}\n\nThis cannot be undone.",
        )
        
        if ok and text == "DELETE":
            try:
                shutil.rmtree(target_dir)
                self.log(f"[WIPE] Deleted {target_dir}")
                self.on_lock_clicked() # Rescan
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete: {e}")

    # ==========================================================
    # LOGIC: BATCH EXECUTION
    # ==========================================================

    def on_start_clicked(self):
        # 1. Build Queue from Checked Boxes
        self.processing_queue = []
        
        for row in range(self.table.rowCount()):
            # Get widget at column 0
            widget = self.table.cellWidget(row, 0)
            if widget:
                # Find the checkbox inside the layout
                checkbox = widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    # Retrieve the hidden task data from Column 1
                    item_data = self.table.item(row, 1)
                    if item_data:
                        self.processing_queue.append(item_data.data(Qt.UserRole))

        if not self.processing_queue:
            QMessageBox.warning(self, "No Videos Selected", "Please tick at least one video to process.")
            return
            
        self.is_processing = True
        self.btn_start.setEnabled(False)
        self.btn_lock.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_wipe.setEnabled(False)
        
        self.log(f"Queue built: {len(self.processing_queue)} videos selected.")
        self.run_next_task()

    def run_next_task(self):
        """Pops the next task from queue and starts the worker."""
        if not self.processing_queue or not self.is_processing:
            self.finish_batch()
            return
            
        task = self.processing_queue[0] # Peek, remove in finish
        
        loc = task['loc']
        mp4 = task['mp4']
        gp = task['g_proj']
        
        self.log(f"--- Starting: {os.path.basename(mp4)} ---")
        self.pbar.setValue(0)
        self.pbar.setFormat(f"Processing {os.path.basename(mp4)} (%p%)")
        
        # Setup Thread & Worker
        self.worker_thread = QThread()
        self.current_worker = InferenceSession(
            location_code=loc,
            footage_path=mp4,
            config_path=self.config_path,
            output_root=self.root_out_dir,
            g_proj_path=gp,
            config_name=self.selected_config
        )
        self.current_worker.moveToThread(self.worker_thread)
        
        # Connect Signals
        self.current_worker.sig_log.connect(self.log)
        self.current_worker.sig_progress.connect(self.update_progress)
        self.current_worker.sig_error.connect(self.on_worker_error)
        self.current_worker.sig_finished.connect(self.on_session_finished)
        self.worker_thread.started.connect(self.current_worker.run)
        
        # Start
        self.worker_thread.start()

    @pyqtSlot()
    def on_session_finished(self):
        """Called when one video is done."""
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
            
        # Update Table Status
        if self.processing_queue:
            task = self.processing_queue.pop(0)
            row = task['table_row']
            
            # Update Status Column (Index 3)
            self.table.setItem(row, 3, QTableWidgetItem("Done"))
            self.table.item(row, 3).setForeground(pyqtColor("#27ae60"))
            
            # Optional: Uncheck the box visually
            widget = self.table.cellWidget(row, 0)
            checkbox = widget.findChild(QCheckBox)
            if checkbox:
                checkbox.setChecked(False)
            
        # Trigger Next
        self.run_next_task()

    def on_stop_clicked(self):
        self.is_processing = False
        self.log("[STOP] Batch stop requested. Finishing current video...")
        if self.current_worker:
            self.current_worker.request_stop()
        self.btn_stop.setEnabled(False)

    def finish_batch(self):
        self.log("=== Batch Processing Finished ===")
        self.pbar.setValue(100)
        self.pbar.setFormat("Idle")
        self.is_processing = False
        
        self.btn_start.setEnabled(True)
        self.btn_lock.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_wipe.setEnabled(True)
        
        # Refresh Scan to update lists (re-evaluate Pending/Done)
        # self.on_lock_clicked() # Optional: Disable this if you want to keep checkboxes as is

    def on_select_all_clicked(self):
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if not widget:
                continue
            checkbox = widget.findChild(QCheckBox)
            if checkbox and checkbox.isEnabled():
                checkbox.setChecked(True)

    def on_unselect_all_clicked(self):
        for row in range(self.table.rowCount()):
            widget = self.table.cellWidget(row, 0)
            if not widget:
                continue
            checkbox = widget.findChild(QCheckBox)
            if checkbox and checkbox.isEnabled():
                checkbox.setChecked(False)

    # ==========================================================
    # UTILS
    # ==========================================================

    @pyqtSlot(str)
    def log(self, msg):
        self.txt_log.append(msg)
        # Scroll to bottom
        sb = self.txt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    @pyqtSlot(int)
    def update_progress(self, val):
        self.pbar.setValue(val)
    
    @pyqtSlot(str)
    def on_worker_error(self, msg):
        QMessageBox.critical(self, "Pipeline Error", msg)
        self.on_stop_clicked()


# Helper for colors
def pyqtColor(hex_str):
    from PyQt5.QtGui import QColor
    return QColor(hex_str)