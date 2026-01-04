import os
import shutil
import cv2

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QGraphicsView, QGraphicsScene,
    QComboBox, QTextEdit
)
from PyQt5.QtCore import Qt, QPointF, QSize
from PyQt5.QtGui import QPixmap, QFont


class LocationTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.cctv_path = None
        self.sat_path = None
        self.layout_path = None
        self.roi_path = None

        layout = QVBoxLayout(self)

        grp = QGroupBox("Create Location")
        g_l = QVBoxLayout()

        # Location code input
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Location Code:"))
        self.le_code = QLineEdit()
        self.le_code.setPlaceholderText("e.g. 119NH")
        h1.addWidget(self.le_code)
        g_l.addLayout(h1)

        # CCTV picker
        h2 = QHBoxLayout()
        self.btn_pick_cctv = QPushButton("Choose CCTV Image")
        self.btn_pick_cctv.clicked.connect(self.pick_cctv)
        h2.addWidget(self.btn_pick_cctv)
        self.lbl_cctv_info = QLabel("No CCTV selected")
        h2.addWidget(self.lbl_cctv_info)
        g_l.addLayout(h2)

        # SAT picker
        h3 = QHBoxLayout()
        self.btn_pick_sat = QPushButton("Choose SAT Image")
        self.btn_pick_sat.clicked.connect(self.pick_sat)
        h3.addWidget(self.btn_pick_sat)
        self.lbl_sat_info = QLabel("No SAT selected")
        h3.addWidget(self.lbl_sat_info)
        g_l.addLayout(h3)

        # Layout SVG picker (optional)
        h4 = QHBoxLayout()
        self.btn_pick_layout = QPushButton("Choose Layout SVG (optional)")
        self.btn_pick_layout.clicked.connect(self.pick_layout)
        h4.addWidget(self.btn_pick_layout)
        self.lbl_layout_info = QLabel("No layout selected")
        h4.addWidget(self.lbl_layout_info)
        g_l.addLayout(h4)

        # ROI picker (optional)
        h5 = QHBoxLayout()
        self.btn_pick_roi = QPushButton("Choose ROI Image (optional)")
        self.btn_pick_roi.clicked.connect(self.pick_roi)
        h5.addWidget(self.btn_pick_roi)
        self.lbl_roi_info = QLabel("No ROI selected")
        h5.addWidget(self.lbl_roi_info)
        g_l.addLayout(h5)

        # Create button
        h4 = QHBoxLayout()
        h4.addStretch()
        self.btn_create = QPushButton("Create location code")
        self.btn_create.clicked.connect(self.create_location)
        h4.addWidget(self.btn_create)
        g_l.addLayout(h4)

        grp.setLayout(g_l)
        layout.addWidget(grp)

        # --- Media Preview: two panels ---
        class MediaViewer(QGraphicsView):
            PLACEHOLDER_POINT_SIZE = 18
            PLACEHOLDER_BOLD = True
            PLACEHOLDER_COLOR = None

            def __init__(self, parent=None):
                super().__init__(parent)
                self.setScene(QGraphicsScene(self))
                self._pixmap_item = None
                self._placeholder_item = None
                self._last_image_path = None
                self.setDragMode(QGraphicsView.ScrollHandDrag)

            def load_image(self, path: str):
                try:
                    self.scene().clear()
                except: pass
                self._pixmap_item = None
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
                    self.scene().clear()
                except: pass
                try: self.resetTransform()
                except: pass
                self._pixmap_item = None
                self._placeholder_item = None
                try:
                    font = QFont()
                    font.setPointSize(self.PLACEHOLDER_POINT_SIZE)
                    font.setBold(self.PLACEHOLDER_BOLD)
                    self._placeholder_item = self.scene().addText(text, font)
                    rect = self._placeholder_item.boundingRect()
                    scene_rect = self.scene().sceneRect()
                    if scene_rect.isNull():
                        vw = max(400, self.viewport().width() or 400)
                        vh = max(700, self.viewport().height() or 700)
                        self.scene().setSceneRect(0, 0, vw, vh)
                        scene_rect = self.scene().sceneRect()
                    x = (scene_rect.width() - rect.width()) / 2
                    y = (scene_rect.height() - rect.height()) / 2
                    self._placeholder_item.setPos(QPointF(x, y))
                except: self._placeholder_item = None

            def clear(self):
                try: self.scene().clear()
                except: pass
                try: self.resetTransform()
                except: pass
                self._pixmap_item = None
                self._placeholder_item = None
                self._last_image_path = None

            def fit_view(self):
                if self._pixmap_item is None: return
                self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

            def wheelEvent(self, event):
                if self._pixmap_item is None: return
                angle = event.angleDelta().y()
                factor = 1.25 if angle > 0 else 0.8
                self.scale(factor, factor)

        media_box = QGroupBox("Media Preview")
        m_layout = QHBoxLayout(media_box)
        m_layout.setContentsMargins(6, 6, 6, 6)

        # Left - CCTV
        self.media_cctv = MediaViewer()
        self.media_cctv.setMinimumSize(320, 500)
        left_vbox = QVBoxLayout()
        left_vbox.addWidget(self.media_cctv)
        self.media_cctv_fit = QPushButton("Fit CCTV")
        self.media_cctv_fit.clicked.connect(lambda: self.media_cctv.fit_view())
        left_vbox.addWidget(self.media_cctv_fit)
        left_widget = QWidget(); left_widget.setLayout(left_vbox)

        # Right - SAT
        self.media_sat = MediaViewer()
        self.media_sat.setMinimumSize(320, 500)
        right_vbox = QVBoxLayout()
        right_vbox.addWidget(self.media_sat)
        self.media_sat_fit = QPushButton("Fit SAT")
        self.media_sat_fit.clicked.connect(lambda: self.media_sat.fit_view())
        right_vbox.addWidget(self.media_sat_fit)
        right_widget = QWidget(); right_widget.setLayout(right_vbox)

        m_layout.addWidget(left_widget)
        m_layout.addWidget(right_widget)
        layout.addWidget(media_box)
        # --- Import Footage section ---
        imp_grp = QGroupBox("Import Footage into existing location")
        imp_l = QVBoxLayout()

        row_loc = QHBoxLayout()
        row_loc.addWidget(QLabel("Location:"))
        self.combo_locations = QComboBox()
        row_loc.addWidget(self.combo_locations)
        self.btn_refresh_locations = QPushButton("Refresh")
        self.btn_refresh_locations.clicked.connect(self._populate_location_combo)
        row_loc.addWidget(self.btn_refresh_locations)
        imp_l.addLayout(row_loc)

        row_imp = QHBoxLayout()
        self.btn_add_footage = QPushButton("Add footage (mp4)")
        self.btn_add_footage.clicked.connect(self.add_footage)
        row_imp.addWidget(self.btn_add_footage)
        row_imp.addStretch()
        imp_l.addLayout(row_imp)

        # Console
        self.footage_console = QTextEdit()
        self.footage_console.setReadOnly(True)
        self.footage_console.setFixedHeight(160)
        imp_l.addWidget(self.footage_console)

        imp_grp.setLayout(imp_l)
        layout.addWidget(imp_grp)
        layout.addStretch()

        # populate locations
        self._populate_location_combo()

    # --- Helpers ---
    def pick_cctv(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select CCTV image", os.getcwd(), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not fn: return
        info = self._ensure_png(fn, "cctv")
        if not info: return
        path, (w, h) = info
        self.cctv_path = path
        self.lbl_cctv_info.setText(f"{os.path.basename(path)} — {w}x{h}")
        try:
            self.media_cctv.load_image(path)
        except Exception:
            pass

    def pick_sat(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select SAT image", os.getcwd(), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not fn: return
        info = self._ensure_png(fn, "sat")
        if not info: return
        path, (w, h) = info
        self.sat_path = path
        self.lbl_sat_info.setText(f"{os.path.basename(path)} — {w}x{h}")
        try:
            self.media_sat.load_image(path)
        except Exception:
            pass

    def pick_layout(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select Layout SVG", os.getcwd(), "SVG Files (*.svg)")
        if not fn: return
        # Only accept .svg
        ext = os.path.splitext(fn)[1].lower()
        if ext != '.svg':
            QMessageBox.warning(self, "Invalid file", "Please select an SVG file for layout.")
            return
        self.layout_path = fn
        self.lbl_layout_info.setText(os.path.basename(fn))

    def pick_roi(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select ROI image", os.getcwd(), "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not fn: return
        info = self._ensure_png(fn, "roi")
        if not info: return
        path, (w, h) = info
        self.roi_path = path
        self.lbl_roi_info.setText(f"{os.path.basename(path)} — {w}x{h}")

    def _ensure_png(self, src_path, kind):
        # Read with cv2 to get resolution
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.warning(self, "Image load failed", f"Failed to read image: {src_path}")
            return None
        h, w = img.shape[:2]

        # If PNG already and writable, just return original path
        ext = os.path.splitext(src_path)[1].lower()
        if ext == '.png':
            return src_path, (w, h)

        # Else convert to PNG in a temp location inside workspace (do not overwrite source)
        try:
            tmp_dir = os.path.join(os.getcwd(), 'location_temp')
            os.makedirs(tmp_dir, exist_ok=True)
            base = os.path.basename(src_path)
            name, _ = os.path.splitext(base)
            dst = os.path.join(tmp_dir, f"{name}.png")
            # Write as PNG
            ok = cv2.imwrite(dst, img)
            if not ok:
                QMessageBox.warning(self, "Conversion failed", f"Failed to convert {src_path} to PNG")
                return None
            QMessageBox.information(self, "Converted to PNG", f"Converted {base} → {os.path.basename(dst)}")
            return dst, (w, h)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error converting image: {e}")
            return None

    # --- Footage import helpers ---
    def _populate_location_combo(self):
        self.combo_locations.clear()
        loc_root = os.path.join(os.getcwd(), 'location')
        try:
            entries = os.listdir(loc_root)
        except Exception:
            entries = []
        dirs = [n for n in entries if os.path.isdir(os.path.join(loc_root, n))]
        dirs.sort()
        self.combo_locations.addItem('(none)')
        for d in dirs:
            self.combo_locations.addItem(d)

    def _log_console(self, text: str):
        try:
            self.footage_console.append(text)
        except Exception:
            pass

    def add_footage(self):
        code = self.combo_locations.currentText()
        if not code or code == '(none)':
            QMessageBox.warning(self, "No location", "Please select an existing location code.")
            return

        src_files, _ = QFileDialog.getOpenFileNames(self, "Select MP4 files", os.getcwd(), "Videos (*.mp4)")
        if not src_files: return

        loc_dir = os.path.join(os.getcwd(), 'location', code)
        footage_dir = os.path.join(loc_dir, 'footage')
        os.makedirs(footage_dir, exist_ok=True)

        for src in src_files:
            try:
                ext = os.path.splitext(src)[1].lower()
                if ext != '.mp4':
                    self._log_console(f"Skipped non-mp4: {os.path.basename(src)}")
                    continue

                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    self._log_console(f"Failed to open: {os.path.basename(src)}")
                    continue

                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                duration = frames / fps if fps > 0 else 0.0
                cap.release()

                dst = os.path.join(footage_dir, os.path.basename(src))
                # avoid overwriting existing footage file
                if os.path.exists(dst):
                    base, ext = os.path.splitext(os.path.basename(src))
                    i = 1
                    while os.path.exists(os.path.join(footage_dir, f"{base}_{i}{ext}")):
                        i += 1
                    dst = os.path.join(footage_dir, f"{base}_{i}{ext}")

                shutil.copy2(src, dst)

                self._log_console(f"[{code}] {os.path.basename(dst)} — {w}x{h}, {fps:.2f} fps, {duration:.2f}s, {frames} frames")
            except Exception as e:
                self._log_console(f"Error importing {os.path.basename(src)}: {e}")

    def create_location(self):
        code = self.le_code.text().strip()
        if not code:
            QMessageBox.warning(self, "Missing code", "Please enter a location code.")
            return
        # sanitize: no path separators
        if any(c in code for c in ('/', '\\')):
            QMessageBox.warning(self, "Invalid code", "Location code may not contain path separators.")
            return

        loc_dir = os.path.join(os.getcwd(), 'location', code)
        if os.path.exists(loc_dir):
            QMessageBox.warning(self, "Exists", f"Location '{code}' already exists. Will not overwrite.")
            return

        if not self.cctv_path or not self.sat_path:
            QMessageBox.warning(self, "Missing images", "Please select both CCTV and SAT images before creating the location.")
            return

        try:
            os.makedirs(loc_dir, exist_ok=False)
            # Enforce filenames
            dst_cctv = os.path.join(loc_dir, f'cctv_{code}.png')
            dst_sat = os.path.join(loc_dir, f'sat_{code}.png')

            # Copy or move from temp/current paths
            shutil.copy2(self.cctv_path, dst_cctv)
            shutil.copy2(self.sat_path, dst_sat)

            # Optional: layout SVG
            try:
                if self.layout_path:
                    dst_layout = os.path.join(loc_dir, f'layout_{code}.svg')
                    shutil.copy2(self.layout_path, dst_layout)
            except Exception:
                pass

            # Optional: roi image (already PNG via _ensure_png)
            try:
                if self.roi_path:
                    dst_roi = os.path.join(loc_dir, f'roi_{code}.png')
                    shutil.copy2(self.roi_path, dst_roi)
            except Exception:
                pass

            QMessageBox.information(self, "Created", f"Location '{code}' created with CCTV and SAT images.")
            # Reset UI
            self.le_code.clear()
            self.cctv_path = None; self.sat_path = None
            self.layout_path = None; self.roi_path = None
            self.lbl_cctv_info.setText("No CCTV selected"); self.lbl_sat_info.setText("No SAT selected")
            try:
                self.lbl_layout_info.setText("No layout selected")
            except Exception:
                pass
            try:
                self.lbl_roi_info.setText("No ROI selected")
            except Exception:
                pass
            try:
                # Clear the media preview panels
                if hasattr(self, 'media_cctv'):
                    self.media_cctv.clear()
                if hasattr(self, 'media_sat'):
                    self.media_sat.clear()
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create location: {e}")
            # Attempt to cleanup if partially created
            try:
                if os.path.isdir(loc_dir):
                    shutil.rmtree(loc_dir)
            except Exception:
                pass
