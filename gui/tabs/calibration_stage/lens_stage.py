import os
import numpy as np
try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QGridLayout,
    QLineEdit,
    QSizePolicy,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)
from PyQt5.QtGui import QPixmap, QImage, QDoubleValidator
from PyQt5.QtCore import Qt, QPointF


class ImageViewer(QGraphicsView):
    """Simple pan/zoom image viewer backed by QGraphicsView."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = None
        self._zoom = 0
        self.setRenderHints(self.renderHints() | Qt.SmoothTransformation)
        self.setInteractive(True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def load_pixmap(self, pixmap: QPixmap):
        self.scene().clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self._pixmap_item)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self._zoom = 0

    def fitToView(self):
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        # Zoom in/out with wheel
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        if self._zoom < -10:
            self._zoom = -10
            return
        if self._zoom > 30:
            self._zoom = 30
            return
        self.scale(factor, factor)


class LensStage(QWidget):
    """Lens (Intrinsics) stage UI with image viewer and controls."""

    def __init__(self, project_root: str = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel("Lens (Intrinsics K)")
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)

        desc = QLabel(
            "Camera intrinsics editor. Load a CCTV still, pan/zoom the image, edit a 3x3 intrinsics matrix, or use defaults."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Image viewer
        self.viewer = ImageViewer()
        self.viewer.setMinimumHeight(600)
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.viewer)

        # Controls area below image
        controls = QWidget()
        c_layout = QVBoxLayout(controls)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(8)

        # top row: resolution text + Fit button
        top_row = QWidget()
        tr_layout = QHBoxLayout(top_row)
        tr_layout.setContentsMargins(0, 0, 0, 0)
        tr_layout.setSpacing(8)

        self.res_label = QLabel("Resolution: -")
        tr_layout.addWidget(self.res_label)

        tr_layout.addStretch(1)

        self.fit_btn = QPushButton("Fit CCTV")
        self.fit_btn.clicked.connect(self.viewer.fitToView)
        tr_layout.addWidget(self.fit_btn)

        c_layout.addWidget(top_row)

        # intrinsics matrix inputs (3x3)
        matrix_widget = QWidget()
        mg = QGridLayout(matrix_widget)
        mg.setContentsMargins(0, 0, 0, 0)
        mg.setSpacing(6)

        self.matrix_inputs = []
        validator = QDoubleValidator()
        for r in range(3):
            row_inputs = []
            for c in range(3):
                le = QLineEdit()
                le.setFixedWidth(100)
                le.setValidator(validator)
                mg.addWidget(le, r, c)
                row_inputs.append(le)
            self.matrix_inputs.append(row_inputs)

        c_layout.addWidget(matrix_widget)

        # bottom row: Use Default button
        bottom_row = QWidget()
        br_layout = QHBoxLayout(bottom_row)
        br_layout.setContentsMargins(0, 0, 0, 0)
        br_layout.setSpacing(8)

        br_layout.addStretch(1)
        self.use_default_btn = QPushButton("Use Default")
        self.use_default_btn.clicked.connect(self._apply_defaults_from_image)
        br_layout.addWidget(self.use_default_btn)

        # Apply Intrinsics row
        # Replace simple Apply with a combined Apply Intrinsics which previews undistort and updates the JSON
        self.apply_intrinsics_btn = QPushButton("Apply Intrinsics")
        self.apply_intrinsics_btn.clicked.connect(self._apply_intrinsics_and_preview)
        br_layout.addWidget(self.apply_intrinsics_btn)

        # Proceed button: move to Undistort stage (index 2) without confirmation
        self.proceed_btn = QPushButton("Proceed")
        self.proceed_btn.clicked.connect(self._on_proceed)
        br_layout.addWidget(self.proceed_btn)

        c_layout.addWidget(bottom_row)

        layout.addWidget(controls)
        layout.addStretch(1)

        # placeholder: no image loaded yet
        self._current_pixmap = None
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

    def showEvent(self, event):
        super().showEvent(event)
        # Refresh UI from host inspect_obj when this stage is shown
        try:
            host = getattr(self, 'host_tab', None) or self.parent()
            if host is None:
                return
            obj = getattr(host, 'inspect_obj', None)
            if obj is None:
                return
            und = obj.get('undistort', {})
            # populate matrix if present
            K = und.get('K')
            if K and isinstance(K, (list, tuple)) and len(K) >= 3:
                for r in range(3):
                    for c in range(3):
                        try:
                            self.matrix_inputs[r][c].setText(str(K[r][c]))
                        except Exception:
                            self.matrix_inputs[r][c].setText("")

            # populate resolution label
            res = und.get('resolution')
            if res and isinstance(res, (list, tuple)) and len(res) >= 2:
                self.res_label.setText(f"Resolution: {res[0]} x {res[1]}")
            else:
                self.res_label.setText("Resolution: -")

            # attempt to load CCTV image from project location
            loc_code = None
            try:
                loc_code = obj.get('meta', {}).get('location_code')
            except Exception:
                loc_code = None

            proj_root = None
            if getattr(self, 'project_root', None):
                proj_root = self.project_root
            else:
                # try to inherit from host's pick_stage
                try:
                    proj_root = getattr(host, 'pick_stage').project_root
                except Exception:
                    proj_root = os.getcwd()

            if loc_code:
                cctv_path = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
                if os.path.isfile(cctv_path):
                    try:
                        img = QImage(cctv_path)
                        if not img.isNull():
                            # load as the original source image so multiple applies don't compound
                            self.load_original_image(img)
                            self.status_label.setText(f"Loaded CCTV image for {loc_code}.")
                    except Exception:
                        pass

        except Exception:
            pass

    def load_image(self, qimage: QImage):
        pix = QPixmap.fromImage(qimage)
        self._current_pixmap = pix
        self._current_qimage = qimage
        self.viewer.load_pixmap(pix)
        self.viewer.fitToView()
        self._update_resolution_label()

    def load_original_image(self, qimage: QImage):
        # keep an immutable original copy for undistort processing
        self._original_qimage = qimage
        # display original initially
        self._current_qimage = qimage
        pix = QPixmap.fromImage(qimage)
        self._current_pixmap = pix
        self.viewer.load_pixmap(pix)
        self.viewer.fitToView()
        self._update_resolution_label()

    def load_pixmap(self, pixmap: QPixmap):
        self._current_pixmap = pixmap
        self.viewer.load_pixmap(pixmap)
        self.viewer.fitToView()
        self._update_resolution_label()

    def _update_resolution_label(self):
        if self._current_pixmap is None:
            self.res_label.setText("Resolution: -")
        else:
            w = self._current_pixmap.width()
            h = self._current_pixmap.height()
            self.res_label.setText(f"Resolution: {w} x {h}")

    def _apply_defaults_from_image(self):
        if self._current_pixmap is None:
            return
        w = self._current_pixmap.width()
        h = self._current_pixmap.height()

        # FIX: Center X should be width/2, Center Y should be height/2
        fx = float(w)
        fy = float(w)
        s = 0.0
        cx = float(w) / 2.0  # Corrected from h/2
        cy = float(h) / 2.0  # Corrected from w/2

        vals = [
            [fx, s, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ]

        for r in range(3):
            for c in range(3):
                self.matrix_inputs[r][c].setText(str(vals[r][c]))

    def _qimage_to_cv(self, qimg: QImage):
        if qimg is None:
            return None
        img = qimg.convertToFormat(QImage.Format_RGB888)
        w = img.width()
        h = img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
        # RGB -> BGR for OpenCV
        return arr[:, :, ::-1].copy()

    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None:
            return None
        # convert BGR to RGB
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        return qimg.copy()

    def _apply_intrinsics_and_preview(self):
        # Read matrix inputs into K
        K = np.zeros((3, 3), dtype=np.float64)
        for r in range(3):
            for c in range(3):
                txt = self.matrix_inputs[r][c].text().strip()
                try:
                    K[r, c] = float(txt) if txt != "" else 0.0
                except Exception:
                    K[r, c] = 0.0

        host = getattr(self, 'host_tab', None) or self.parent()
        if host is None:
            self.status_label.setText("No host available to update inspect object.")
            return
        obj = getattr(host, 'inspect_obj', None)
        if obj is None:
            self.status_label.setText("No inspect object available.")
            return

        und = obj.setdefault('undistort', {})
        D = und.get('D', [0.0, 0.0, 0.0, 0.0, 0.0])
        try:
            D_np = np.array(D, dtype=np.float64)
        except Exception:
            D_np = np.zeros((5,), dtype=np.float64)

        if not HAS_CV2:
            self.status_label.setText('OpenCV (cv2) not available; cannot preview undistort.')
            und['K'] = K.tolist()
            if self._current_pixmap is not None:
                und['resolution'] = [self._current_pixmap.width(), self._current_pixmap.height()]
            host.inspect_obj = obj
            return

        src_cv = None
        try:
            # Prefer the original source image so repeated applies do not compound
            if getattr(self, '_original_qimage', None) is not None:
                src_cv = self._qimage_to_cv(self._original_qimage)
            elif getattr(self, '_current_qimage', None) is not None:
                src_cv = self._qimage_to_cv(self._current_qimage)
        except Exception:
            src_cv = None

        if src_cv is None:
            loc_code = obj.get('meta', {}).get('location_code')
            proj_root = self.project_root or os.getcwd()
            if loc_code:
                p = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
                if os.path.isfile(p):
                    src_cv = cv2.imread(p, cv2.IMREAD_COLOR)

        if src_cv is None:
            self.status_label.setText('No CCTV image available to preview undistort.')
            return

        h, w = src_cv.shape[:2]
        try:
            # FIX: Use K.copy() instead of getOptimalNewCameraMatrix to match Reference Code logic
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D_np, (w, h), 1, (w, h))
            newcameramtx = K.copy()
            undist = cv2.undistort(src_cv, K, D_np, None, newcameramtx)
        except Exception as e:
            self.status_label.setText(f'Undistort failed: {e}')
            return

        qimg = self._cv_to_qimage(undist)
        if qimg is not None:
            self.load_image(qimg)
            und['K'] = K.tolist()
            und['D'] = D_np.tolist()
            und['resolution'] = [w, h]
            host.inspect_obj = obj
            self.status_label.setText('Applied intrinsics and updated inspect object (in-memory).')
        else:
            self.status_label.setText('Failed to convert undistorted image for preview.')


    def _apply_matrix_to_inspect(self):
        """Read matrix inputs and write back to host.inspect_obj['undistort']['K'] and resolution."""
        try:
            host = getattr(self, 'host_tab', None) or self.parent()
            if host is None:
                self.status_label.setText("No host available.")
                return
            obj = getattr(host, 'inspect_obj', None)
            if obj is None:
                self.status_label.setText("No inspect object available.")
                return
            K = [[0.0, 0.0, 0.0] for _ in range(3)]
            for r in range(3):
                for c in range(3):
                    txt = self.matrix_inputs[r][c].text().strip()
                    if txt == "":
                        val = 0.0
                    else:
                        try:
                            val = float(txt)
                        except Exception:
                            val = 0.0
                    K[r][c] = val

            und = obj.setdefault('undistort', {})
            und['K'] = K
            # update resolution from current pixmap if available
            if self._current_pixmap is not None:
                und['resolution'] = [self._current_pixmap.width(), self._current_pixmap.height()]

            host.inspect_obj = obj
            self.status_label.setText("Applied intrinsics to inspect object (in-memory).")
            # refresh calibration tab progress highlight if available
            try:
                if hasattr(host, '_update_progress_to_index'):
                    host._update_progress_to_index(getattr(host, 'current_step_index', 1))
            except Exception:
                pass
        except Exception as e:
            self.status_label.setText(f"Apply failed: {e}")

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host is None:
            self.status_label.setText("No host available to change stage.")
            return
        try:
            # move to Undistort stage (index 2)
            if hasattr(host, '_show_stage'):
                host._show_stage(2)
            host.current_step_index = 2
            if hasattr(host, '_update_progress_to_index'):
                host._update_progress_to_index(2)
            self.status_label.setText("Proceeding to Undistort stage.")
        except Exception as e:
            self.status_label.setText(f"Failed to proceed: {e}")