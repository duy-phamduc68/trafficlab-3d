import os
from typing import Optional

from PyQt5.QtCore import Qt, QPointF, QSize
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QCheckBox,
    QDialog,
    QMessageBox,
    QApplication,
    QDialogButtonBox,
    QGraphicsView,
    QGraphicsScene,
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter, QFont
from PyQt5.QtSvg import QSvgRenderer
from ...utils.trafficlab_json import default_config, load_config


class ConstructDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Construct G_projection options")
        self.setModal(True)
        self.resize(320, 140)

        layout = QVBoxLayout(self)
        self.svg_cb = QCheckBox("Use SVG layout")
        self.roi_cb = QCheckBox("Use ROI mask")
        layout.addWidget(self.svg_cb)
        layout.addWidget(self.roi_cb)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def result_options(self):
        return {"use_svg": bool(self.svg_cb.isChecked()), "use_roi": bool(self.roi_cb.isChecked())}


class PickStage(QWidget):
    """UI for the 'Pick' calibration stage."""

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root or os.getcwd()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        row = QHBoxLayout()
        row.addWidget(QLabel("Location:"))
        self.combo = QComboBox()
        row.addWidget(self.combo)
        # Dedicated refresh button (replaces the old "(none chosen / refresh)" hack)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Refresh location list and reset selection to none chosen")
        self.refresh_btn.clicked.connect(self._on_refresh)
        row.addWidget(self.refresh_btn)
        self._populate_location_list()
        self.combo.currentIndexChanged.connect(self._on_location_changed)

        layout.addLayout(row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # Action buttons
        actions = QGroupBox("Actions")
        a_layout = QHBoxLayout(actions)
        self.construct_btn = QPushButton("Construct")
        self.construct_btn.clicked.connect(self._on_construct)
        self.validate_btn = QPushButton("Validate")
        self.validate_btn.clicked.connect(self._on_validate)
        self.reconstruct_btn = QPushButton("Reconstruct")
        self.reconstruct_btn.clicked.connect(self._on_reconstruct)
        
        self.construct_btn.setEnabled(False)
        self.validate_btn.setEnabled(False)
        self.reconstruct_btn.setEnabled(False)

        a_layout.addWidget(self.construct_btn)
        a_layout.addWidget(self.validate_btn)
        a_layout.addWidget(self.reconstruct_btn)
        layout.addWidget(actions)

        # Media Preview
        media_box = QGroupBox("Media Preview")
        m_layout = QHBoxLayout(media_box)
        m_layout.setContentsMargins(6, 6, 6, 6)

        class MediaViewer(QGraphicsView):
            PLACEHOLDER_POINT_SIZE = 18
            PLACEHOLDER_BOLD = True
            PLACEHOLDER_COLOR = QColor(120, 120, 120)
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setScene(QGraphicsScene(self))
                self._pixmap_item = None
                self._overlay_item = None
                self._placeholder_item = None
                self._last_image_path = None
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                self._zoom = 0

            def load_image(self, path: str):
                self.scene().clear()
                # FIX: Reset references immediately after clearing scene
                self._pixmap_item = None
                self._overlay_item = None
                self._placeholder_item = None
                try: self.resetTransform()
                except: pass
                
                pix = QPixmap(path)
                if pix and not pix.isNull():
                    self._pixmap_item = self.scene().addPixmap(pix)
                    self._last_image_path = path
                    self.scene().setSceneRect(self._pixmap_item.boundingRect())
                    self.fit_view()

            def set_placeholder(self, text: str):
                try:
                    # remove any existing items
                    self.scene().clear()
                except: pass
                try: self.resetTransform()
                except: pass
                self._pixmap_item = None
                self._overlay_item = None
                self._last_image_path = None
                try:
                    font = QFont()
                    font.setPointSize(self.PLACEHOLDER_POINT_SIZE)
                    font.setBold(self.PLACEHOLDER_BOLD)
                    self._placeholder_item = self.scene().addText(text, font)
                    try:
                        self._placeholder_item.setDefaultTextColor(self.PLACEHOLDER_COLOR)
                    except: pass
                    # center the text in the view
                    rect = self._placeholder_item.boundingRect()
                    scene_rect = self.scene().sceneRect()
                    if scene_rect.isNull():
                        # set a consistent scene rect so centering is stable
                        vw = max(400, self.viewport().width() or 400)
                        vh = max(300, self.viewport().height() or 300)
                        self.scene().setSceneRect(0, 0, vw, vh)
                        scene_rect = self.scene().sceneRect()
                    x = (scene_rect.width() - rect.width()) / 2
                    y = (scene_rect.height() - rect.height()) / 2
                    self._placeholder_item.setPos(QPointF(x, y))
                except:
                    self._placeholder_item = None

            def clear_placeholder(self):
                try:
                    if self._placeholder_item:
                        if self._placeholder_item.scene() == self.scene():
                            self.scene().removeItem(self._placeholder_item)
                        self._placeholder_item = None
                except: pass

            def load_svg(self, path: str):
                try:
                    renderer = QSvgRenderer(path)
                    if not renderer.isValid(): return
                    target_size = None
                    if self._pixmap_item is not None:
                        rect = self._pixmap_item.boundingRect()
                        target_size = rect.size().toSize()
                    else:
                        default_sz = renderer.defaultSize()
                        if default_sz.isValid(): target_size = default_sz
                    if not target_size or target_size.width() == 0: target_size = QSize(800, 600)
                    
                    img = QImage(target_size, QImage.Format_ARGB32)
                    img.fill(0)
                    painter = QPainter(img)
                    renderer.render(painter)
                    painter.end()
                    pix = QPixmap.fromImage(img)
                    
                    self.scene().clear()
                    try: self.resetTransform()
                    except: pass
                    # FIX: Reset references immediately after clearing scene
                    self._pixmap_item = None
                    self._overlay_item = None
                    self._placeholder_item = None

                    self._pixmap_item = self.scene().addPixmap(pix)
                    self._last_image_path = None
                    self.scene().setSceneRect(self._pixmap_item.boundingRect())
                    self.fit_view()
                except Exception: return

            def clear(self):
                self.scene().clear()
                try: self.resetTransform()
                except: pass
                self._pixmap_item = None
                self._overlay_item = None
                self._placeholder_item = None
                self._last_image_path = None

            def fit_view(self):
                self._zoom = 0
                if self._pixmap_item is None: return
                self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

            def wheelEvent(self, event):
                if self._pixmap_item is None: return
                angle = event.angleDelta().y()
                factor = 1.25 if angle > 0 else 0.8
                self.scale(factor, factor)

            def set_overlay(self, pixmap: QPixmap):
                # FIX: Safer removal logic
                try:
                    if self._overlay_item:
                        # Only try to remove if it is actually in this scene
                        if self._overlay_item.scene() == self.scene():
                            self.scene().removeItem(self._overlay_item)
                        self._overlay_item = None
                except: pass
                
                if pixmap and not pixmap.isNull():
                    self._overlay_item = self.scene().addPixmap(pixmap)
                    try: self._overlay_item.setZValue(1)
                    except: pass

            def clear_overlay(self):
                try:
                    if self._overlay_item:
                        if self._overlay_item.scene() == self.scene():
                            self.scene().removeItem(self._overlay_item)
                        self._overlay_item = None
                except: pass

        self.media1 = MediaViewer()
        self.media1.setMinimumSize(320, 240)
        left_vbox = QVBoxLayout()
        left_vbox.addWidget(self.media1)
        self.media1_fit = QPushButton("Fit CCTV")
        self.media1_fit.clicked.connect(lambda: self.media1.fit_view())
        left_vbox.addWidget(self.media1_fit)
        
        self.media1_roi_cb = QCheckBox("Show ROI overlay")
        self.media1_roi_cb.setEnabled(False)
        self.media1_roi_cb.toggled.connect(self._on_roi_toggled)
        left_vbox.addWidget(self.media1_roi_cb)

        self.media2 = MediaViewer()
        self.media2.setMinimumSize(320, 500)
        right_vbox = QVBoxLayout()
        right_vbox.addWidget(self.media2)
        self.media2_fit = QPushButton("Fit SAT")
        self.media2_fit.clicked.connect(lambda: self.media2.fit_view())
        right_vbox.addWidget(self.media2_fit)
        
        self.media2_svg_cb = QCheckBox("Show SVG layout")
        self.media2_svg_cb.setEnabled(False)
        self.media2_svg_cb.toggled.connect(self._on_svg_toggled)
        right_vbox.addWidget(self.media2_svg_cb)

        left_widget = QWidget(); left_widget.setLayout(left_vbox)
        right_widget = QWidget(); right_widget.setLayout(right_vbox)

        m_layout.addWidget(left_widget)
        m_layout.addWidget(right_widget)
        layout.addWidget(media_box)

        self.summary = QLabel("No location selected.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.proceed_btn = QPushButton("Proceed")
        self.proceed_btn.setEnabled(False)
        self.proceed_btn.clicked.connect(self._on_proceed)
        btn_row.addWidget(self.proceed_btn)
        layout.addLayout(btn_row)

        layout.addStretch(1)

        self.last_location = None
        self.last_options = {}

    def _refresh_host_timeline(self):
        """Forces the parent CalibrationTab to update step status (enable/disable SVG/ROI)."""
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and hasattr(host, '_update_progress_to_index') and hasattr(host, 'current_step_index'):
            host._update_progress_to_index(host.current_step_index)

    def _on_proceed(self):
        if not self.last_location:
            self.status_label.setText("Cannot proceed: no location selected.")
            return

        action = self.last_options.get('action')
        if action not in ("construct", "reconstruct", "validate"):
            self.status_label.setText("Proceed not available. Please choose an action first.")
            return

        host = getattr(self, 'host_tab', None) or self.parent()
        if host is None: return

        # Validate -> Jump to Final (12)
        if action == "validate":
            if hasattr(host, '_show_stage'): host._show_stage(12)
            host.current_step_index = 12
            if hasattr(host, '_update_progress_to_index'): host._update_progress_to_index(12)
            self.status_label.setText(f"Validating {self.last_location}. Going to Final.")
            return

        # Construct/Reconstruct -> Jump to Lens (1)
        if action in ("construct", "reconstruct"):
            if hasattr(host, '_show_stage'): host._show_stage(1)
            host.current_step_index = 1
            if hasattr(host, '_update_progress_to_index'): host._update_progress_to_index(1)
            self.status_label.setText(f"Proceeding to Lens for {self.last_location}.")
            return

    def _populate_location_list(self):
        self.combo.clear()
        loc_root = os.path.join(self.project_root, "location")
        try:
            entries = os.listdir(loc_root)
        except Exception:
            entries = []
        dirs = [n for n in entries if os.path.isdir(os.path.join(loc_root, n))]
        dirs.sort()
        self.combo.addItem("(none chosen)")
        for d in dirs: self.combo.addItem(d)

    def _on_refresh(self):
        """Refresh the location listing and reset selection to '(none chosen)'."""
        try:
            self._populate_location_list()
            # Reset to first entry which is the '(none chosen)'
            self.combo.setCurrentIndex(0)
            try:
                self.status_label.setText("Location list refreshed.")
            except: pass
        except Exception:
            try: self.status_label.setText("Failed to refresh location list.")
            except: pass

    def _on_location_changed(self, idx: int):
        if idx <= 0:
            self.last_location = None
            host = getattr(self, 'host_tab', None) or self.parent()
            if host: host.inspect_obj = None
            self.status_label.setText("No location selected.")
            self.construct_btn.setEnabled(False)
            self.validate_btn.setEnabled(False)
            self.reconstruct_btn.setEnabled(False)
            self.proceed_btn.setEnabled(False)
            self.last_options = {}
            try:
                # show placeholders indicating nothing loaded
                try: self.media1.set_placeholder("No location loaded")
                except: self.media1.clear()
                try: self.media2.set_placeholder("No location loaded")
                except: self.media2.clear()
                self.media1.clear_overlay(); self.media2.clear_overlay()
                self.media1_roi_cb.setChecked(False); self.media1_roi_cb.setEnabled(False)
                self.media2_svg_cb.setChecked(False); self.media2_svg_cb.setEnabled(False)
            except: pass
            self._update_summary()
            return

        code = self.combo.itemText(idx)
        self.code_selected(code)

    def code_selected(self, code: str):
        self.last_location = code
        self.last_options = {}
        gpath = self._gproj_exists_exact(code)
        host = getattr(self, 'host_tab', None) or self.parent()
        
        try:
            if gpath:
                cfg = load_config(gpath)
                if host: host.inspect_obj = cfg
                self.status_label.setText(f"Loaded: {gpath}")
                self.construct_btn.setEnabled(False)
                self.validate_btn.setEnabled(True)
                self.reconstruct_btn.setEnabled(True)
                self.last_options = {"action": "found", "gproj_path": gpath}
            else:
                cfg = default_config(code)
                if host: host.inspect_obj = cfg
                self.status_label.setText(f"No G_projection for {code}.")
                self.construct_btn.setEnabled(True)
                self.validate_btn.setEnabled(False)
                self.reconstruct_btn.setEnabled(False)
                self.last_options = {"action": "missing"}
            
            # REFRESH TIMELINE FLAGS
            self._refresh_host_timeline()

        except Exception as e:
            self.status_label.setText(f"Error: {e}")

        try: self._load_media_previews(code)
        except: pass
        self._update_summary()

    def _find_location_dir_exact(self, code: str) -> Optional[str]:
        loc_root = os.path.join(self.project_root, "location")
        try:
            for name in os.listdir(loc_root):
                if name == code and os.path.isdir(os.path.join(loc_root, name)):
                    return os.path.join(loc_root, name)
        except: pass
        return None

    def _gproj_exists_exact(self, code: str) -> Optional[str]:
        loc_dir = self._find_location_dir_exact(code)
        if not loc_dir: return None
        fname = f"G_projection_{code}.json"
        try:
            for name in os.listdir(loc_dir):
                if name == fname and os.path.isfile(os.path.join(loc_dir, name)):
                    return os.path.join(loc_dir, name)
        except: pass
        return None

    def _check_required_files(self, code: str, use_svg: bool, use_roi: bool):
        missing = []
        loc_dir = self._find_location_dir_exact(code)
        if not loc_dir: return [f"location/{code} missing"]
        
        if not os.path.isfile(os.path.join(loc_dir, f"cctv_{code}.png")): missing.append(f"cctv_{code}.png")
        if not os.path.isfile(os.path.join(loc_dir, f"sat_{code}.png")): missing.append(f"sat_{code}.png")
        if use_svg and not os.path.isfile(os.path.join(loc_dir, f"layout_{code}.svg")): missing.append(f"layout_{code}.svg")
        if use_roi and not os.path.isfile(os.path.join(loc_dir, f"roi_{code}.png")): missing.append(f"roi_{code}.png")
        return missing

    def _confirm_reset_due_to_missing(self, missing):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Missing files")
        msg.setText("Missing:\n" + "\n".join(missing))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        return True

    def _on_construct(self):
        dlg = ConstructDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            opts = dlg.result_options()
            cfg = default_config(self.last_location)
            cfg["use_svg"] = bool(opts.get("use_svg", False))
            cfg["use_roi"] = bool(opts.get("use_roi", False))
            
            missing = self._check_required_files(self.last_location, cfg['use_svg'], cfg['use_roi'])
            if missing:
                self._confirm_reset_due_to_missing(missing)
                return

            host = getattr(self, 'host_tab', None) or self.parent()
            if host: 
                host.inspect_obj = cfg
                self._refresh_host_timeline()
            
            self.last_options = {"action": "construct", **opts}
            self._update_summary()

    def _on_reconstruct(self):
        if not self.last_location: return
        dlg = ConstructDialog(self)
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and getattr(host, 'inspect_obj', None):
            exist = host.inspect_obj
            dlg.svg_cb.setChecked(bool(exist.get('use_svg', False)))
            dlg.roi_cb.setChecked(bool(exist.get('use_roi', False)))
            
        if dlg.exec_() == QDialog.Accepted:
            opts = dlg.result_options()
            if host and host.inspect_obj:
                cfg = host.inspect_obj
            else:
                cfg = default_config(self.last_location)
                
            cfg['use_svg'] = bool(opts.get('use_svg', False))
            cfg['use_roi'] = bool(opts.get('use_roi', False))
            
            missing = self._check_required_files(self.last_location, cfg['use_svg'], cfg['use_roi'])
            if missing:
                self._confirm_reset_due_to_missing(missing)
                return
                
            if host: 
                host.inspect_obj = cfg
                self._refresh_host_timeline()
            
            self.last_options = {"action": "reconstruct", **opts}
            self._update_summary()

    def _on_validate(self):
        gpath = self._gproj_exists_exact(self.last_location)
        if not gpath: return
        
        cfg = load_config(gpath)
        use_svg = bool(cfg.get('use_svg', False))
        use_roi = bool(cfg.get('use_roi', False))
        
        missing = self._check_required_files(self.last_location, use_svg, use_roi)
        if missing:
            self._confirm_reset_due_to_missing(missing)
            return
            
        # IMPORTANT: Sync host object immediately so timeline updates
        host = getattr(self, 'host_tab', None) or self.parent()
        if host:
            host.inspect_obj = cfg
            self._refresh_host_timeline()

        self.last_options = {"action": "validate", "gproj_path": gpath}
        self.status_label.setText("Will validate existing G_projection. Click Proceed to jump to Final.")
        self._update_summary()

    def _update_summary(self):
        if not self.last_location: return
        lines = [f"Location: {self.last_location}"]
        act = self.last_options.get('action')
        
        if act: lines.append(f"Action: {act.upper()}")
        
        self.proceed_btn.setEnabled(bool(act))
        self.summary.setText("\n".join(lines))

    def _load_media_previews(self, code):
        loc_dir = self._find_location_dir_exact(code)
        if not loc_dir: return
        
        cctv = os.path.join(loc_dir, f"cctv_{code}.png")
        cctv_exists = os.path.isfile(cctv)
        if cctv_exists:
            self.media1.load_image(cctv)
        else:
            try: self.media1.set_placeholder("Critical resource missing")
            except: self.media1.clear()
        
        sat = os.path.join(loc_dir, f"sat_{code}.png")
        sat_exists = os.path.isfile(sat)
        if sat_exists:
            self.media2.load_image(sat)
        else:
            try: self.media2.set_placeholder("Critical resource missing")
            except: self.media2.clear()
        
        roi = os.path.join(loc_dir, f"roi_{code}.png")
        if os.path.isfile(roi):
            roi_pix = QPixmap(roi)
            if not roi_pix.isNull():
                # Create a semi-transparent red overlay
                self._roi_overlay_pixmap = QPixmap(roi_pix.size())
                self._roi_overlay_pixmap.fill(QColor(255, 0, 0, 100))
                # Create mask where BLACK (0,0,0) becomes transparent
                mask = roi_pix.createMaskFromColor(QColor(0, 0, 0), Qt.MaskOutColor)
                self._roi_overlay_pixmap.setMask(mask)
                self.media1_roi_cb.setEnabled(True)
                
                # Restore state if checked
                if self.media1_roi_cb.isChecked():
                    self.media1.set_overlay(self._roi_overlay_pixmap)
        
        svg = os.path.join(loc_dir, f"layout_{code}.svg")
        if os.path.isfile(svg):
            self._sat_svg_path = svg
            self.media2_svg_cb.setEnabled(True)
        else:
            # if sat is present but svg missing, keep svg checkbox disabled
            if not sat_exists:
                self.media2_svg_cb.setEnabled(False)

    def _on_svg_toggled(self, checked):
        if checked and getattr(self, '_sat_svg_path', None):
            self.media2.load_svg(self._sat_svg_path)
        elif self.last_location:
            self._load_media_previews(self.last_location)

    def _on_roi_toggled(self, checked):
        if checked and hasattr(self, '_roi_overlay_pixmap') and self._roi_overlay_pixmap:
            self.media1.set_overlay(self._roi_overlay_pixmap)
        else:
            self.media1.clear_overlay()