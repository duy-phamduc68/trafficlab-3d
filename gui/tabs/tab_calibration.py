import os
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDialog,
    QTextEdit,
    QDialogButtonBox,
    QMessageBox,
)
from ..utils.trafficlab_json import default_config, to_pretty_json, save_config
from .calibration_stage.pick_stage import PickStage
from .calibration_stage.lens_stage import LensStage
from .calibration_stage.undistort_stage import UndistortStage
from .calibration_stage.val1_stage import Val1Stage
from .calibration_stage.homa_stage import HomAStage
from .calibration_stage.homf_stage import HomFStage
from .calibration_stage.val2_stage import Val2Stage
from .calibration_stage.pars_stage import ParsStage
from .calibration_stage.dist_stage import DistStage
from .calibration_stage.val3_stage import Val3Stage
from .calibration_stage.svg_stage import SVGStage
from .calibration_stage.roi_stage import ROIStage
from .calibration_stage.final_stage import FinalStage

class SaveStage(QWidget):
    """
    Final Stage: Review JSON and Save to Disk.
    """
    def __init__(self, project_root=None, parent=None):
        super().__init__(parent)
        # Import inside to avoid circular import
        from ..utils.trafficlab_json import save_config
        self._save_config_func = save_config
        
        self.project_root = project_root or os.getcwd()
        self.host_tab = None 
        self.save_path = ""
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("Save Configuration")
        title.setStyleSheet("font-weight: bold; font-size: 18px;")
        layout.addWidget(title)
        
        self.lbl_path = QLabel("Target Path: -")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setStyleSheet("font-size: 13px; color: #aaa;")
        layout.addWidget(self.lbl_path)
        
        layout.addWidget(QLabel("Final JSON Preview:"))
        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setStyleSheet("font-family: Consolas, monospace; font-size: 12px;")
        layout.addWidget(self.text_preview)
        
        self.btn_save = QPushButton("Save G_projection.json")
        self.btn_save.setFixedHeight(50)
        self.btn_save.setStyleSheet("""
            QPushButton {
                background-color: #2a84ff; 
                color: white; 
                font-weight: bold; 
                font-size: 16px;
                border-radius: 5px;
                border: 1px solid #1a64db;
            }
            QPushButton:hover { background-color: #1a64db; }
            QPushButton:pressed { background-color: #0d47a1; }
            QPushButton:disabled { background-color: #555; color: #888; border: 1px solid #444; }
        """)
        self.btn_save.clicked.connect(self._on_save)
        layout.addWidget(self.btn_save)
        
        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.lbl_status)
        
    def showEvent(self, event):
        super().showEvent(event)
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None):
            self.lbl_path.setText("No configuration found.")
            self.text_preview.clear()
            self.btn_save.setEnabled(False)
            return
            
        obj = host.inspect_obj
        loc_code = obj.get('meta', {}).get('location_code', 'UNKNOWN')
        
        self.save_path = os.path.join(self.project_root, "location", loc_code, f"G_projection_{loc_code}.json")
        
        self.lbl_path.setText(f"Target Path: {self.save_path}")
        self.text_preview.setPlainText(to_pretty_json(obj))
        self.btn_save.setEnabled(True)
        self.lbl_status.setText("")

    def _on_save(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None): return
        
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            self._save_config_func(self.save_path, host.inspect_obj)
            
            self.lbl_status.setText("✅ Saved Successfully!")
            self.lbl_status.setStyleSheet("color: #2ca02c; font-weight: bold; font-size: 14px;")
        except Exception as e:
            self.lbl_status.setText("❌ Save Failed")
            self.lbl_status.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 14px;")

class InspectDialog(QDialog):
    def __init__(self, parent=None, obj=None):
        super().__init__(parent)
        self.setWindowTitle("Inspect Config")
        self.resize(600, 400)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Current JSON State:"))

        sample_obj = obj if obj is not None else default_config()
        text = QTextEdit()
        text.setPlainText(to_pretty_json(sample_obj))
        text.setReadOnly(True)
        layout.addWidget(text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class CalibrationTab(QWidget):
    STEPS = [
        ("Pick", "Pick"),
        ("Intrinsics", "Lens"),
        ("Undistort", "Undis"),
        ("Validation 1", "Val1"),
        ("Homography Anchors", "HomA"),
        ("Homography FOV", "HomF"),
        ("Validation 2", "Val2"),
        ("Parallax Subjects", "ParS"),
        ("Distance Reference", "Dist"),
        ("Validation 3", "Val3"),
        ("SVG", "SVG"),
        ("ROI", "ROI"),
        ("Final Validation", "Final"),
        ("Save", "Save"),
    ]

    def __init__(self):
        super().__init__()

        main_layout = QVBoxLayout(self)

        # Top bar
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        left_area = QWidget()
        left_v = QVBoxLayout(left_area)
        left_v.setContentsMargins(0, 0, 0, 0)

        buttons_row = QWidget()
        buttons_layout = QHBoxLayout(buttons_row)
        buttons_layout.setContentsMargins(6, 6, 6, 6)
        buttons_layout.setSpacing(8)

        self.step_buttons = []
        self.disabled_steps = {"SVG", "ROI", "Save"}
        
        for idx, (full, short) in enumerate(self.STEPS):
            btn = QPushButton(short)
            btn.setToolTip(full)
            btn.setFixedHeight(32)
            if full in self.disabled_steps:
                btn.setEnabled(False)
            else:
                btn.setEnabled(idx == 0)
                
            btn.clicked.connect(lambda _, i=idx: self._on_step_clicked(i))

            buttons_layout.addWidget(btn)
            self.step_buttons.append(btn)

        left_v.addWidget(buttons_row)

        self.progress_container = QWidget()
        self.progress_container.setFixedHeight(6)
        self.progress_container.setStyleSheet("background-color: #000;")

        self.progress_bar = QWidget(self.progress_container)
        self.progress_bar.setStyleSheet("background-color: #2a84ff;")
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setFixedWidth(0)

        left_v.addWidget(self.progress_container)
        top_layout.addWidget(left_area, 1)

        inspect_btn = QPushButton("Inspect")
        inspect_btn.setFixedSize(100, 36)
        inspect_btn.clicked.connect(self._on_inspect)
        top_layout.addWidget(inspect_btn, 0, Qt.AlignRight | Qt.AlignVCenter)

        main_layout.addWidget(top_widget)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        # Initialize Stages
        self.pick_stage = PickStage(project_root=None, parent=None)
        self.pick_stage.host_tab = self
        
        self.lens_stage = LensStage(project_root=None, parent=None)
        self.lens_stage.host_tab = self
        
        self.undistort_stage = UndistortStage(project_root=None, parent=None)
        self.undistort_stage.host_tab = self
        
        self.val1_stage = Val1Stage(project_root=None, parent=None)
        self.val1_stage.host_tab = self
        
        self.homa_stage = HomAStage(project_root=None, parent=None)
        self.homa_stage.host_tab = self
        
        self.homf_stage = HomFStage(project_root=None, parent=None)
        self.homf_stage.host_tab = self
        
        self.val2_stage = Val2Stage(project_root=None, parent=None)
        self.val2_stage.host_tab = self
        
        self.pars_stage = ParsStage(project_root=None, parent=None)
        self.pars_stage.host_tab = self
        
        self.dist_stage = DistStage(project_root=None, parent=None)
        self.dist_stage.host_tab = self
        
        self.val3_stage = Val3Stage(project_root=None, parent=None)
        self.val3_stage.host_tab = self
        
        self.svg_stage = SVGStage(project_root=None, parent=None)
        self.svg_stage.host_tab = self
        
        self.roi_stage = ROIStage(project_root=None, parent=None)
        self.roi_stage.host_tab = self
        
        self.final_stage = FinalStage(project_root=None, parent=None)
        self.final_stage.host_tab = self
        
        self.save_stage = SaveStage(project_root=None, parent=None)
        self.save_stage.host_tab = self

        self.stage_widgets = [
            self.pick_stage,
            self.lens_stage,
            self.undistort_stage,
            self.val1_stage,
            self.homa_stage,
            self.homf_stage,
            self.val2_stage,
            self.pars_stage,
            self.dist_stage,
            self.val3_stage,
            self.svg_stage,
            self.roi_stage,
            self.final_stage,
            self.save_stage
        ]
        
        while len(self.stage_widgets) < len(self.STEPS):
            self.stage_widgets.append(None)

        self.content_layout.addWidget(self.pick_stage)
        main_layout.addWidget(self.content_area, 1)

    def _on_inspect(self):
        obj = getattr(self, 'inspect_obj', None)
        dlg = InspectDialog(self, obj=obj)
        dlg.exec_()

    def _on_step_clicked(self, index: int):
        self.current_step_index = index
        self._show_stage(index)
        QTimer.singleShot(0, lambda: self._update_progress_to_index(index))

    def _show_stage(self, index: int):
        if index < 0 or index >= len(self.stage_widgets):
            return
        widget = self.stage_widgets[index]
        
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        if widget is None:
            placeholder = QWidget()
            ph_layout = QVBoxLayout(placeholder)
            ph_layout.addWidget(QLabel(f"Stage placeholder: {self.STEPS[index][0]}"))
            self.content_layout.addWidget(placeholder)
        else:
            self.content_layout.addWidget(widget)

    def _update_progress_to_index(self, index: int):
        if index < 0 or index >= len(self.step_buttons):
            return
            
        obj = getattr(self, 'inspect_obj', None)
        if obj:
            use_svg = obj.get('use_svg', False)
            use_roi = obj.get('use_roi', False)
            
            if use_svg:
                self.disabled_steps.discard("SVG")
            else:
                self.disabled_steps.add("SVG")
                
            if use_roi:
                self.disabled_steps.discard("ROI")
            else:
                self.disabled_steps.add("ROI")
                
        # FIX: Ensure "Save" is enabled if we are AT that step (or previous steps completed)
        # Simple rule: If we reached step 12 (Final), step 13 (Save) should be enabled/clickable.
        # Or simpler: Always discard "Save" from disabled if we are close to it.
        # Let's say if we are past index 0, Save is technically reachable if we click through? 
        # Actually, let's just ensure it highlights correctly.
        if index == 13:
            self.disabled_steps.discard("Save")

        btn = self.step_buttons[index]
        global_center = btn.mapToGlobal(btn.rect().center())
        local_center = self.progress_container.mapFromGlobal(global_center)
        x = max(0, local_center.x())
        
        if index == len(self.step_buttons) - 1:
            width = self.progress_container.width()
        else:
            width = min(x, self.progress_container.width())
        self.progress_bar.setFixedWidth(width)

        for i, b in enumerate(self.step_buttons):
            full = self.STEPS[i][0]
            if full in self.disabled_steps:
                b.setStyleSheet("")
                continue

            if i <= index:
                b.setStyleSheet("background-color: #cef; color: #000; font-weight: bold;")
            else:
                b.setStyleSheet("")

        for i, b in enumerate(self.step_buttons):
            full = self.STEPS[i][0]
            if full in self.disabled_steps:
                b.setEnabled(False)
                continue
            
            if i <= index:
                b.setEnabled(True)
            else:
                b.setEnabled(False)

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, lambda: self._update_progress_to_index(getattr(self, 'current_step_index', 0)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, lambda: self._update_progress_to_index(getattr(self, 'current_step_index', 0)))