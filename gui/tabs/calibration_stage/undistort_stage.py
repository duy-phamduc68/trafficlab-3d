import os
from typing import Optional

import numpy as np
try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSizePolicy,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
)


def remap_with_supersample(src, K, D, newcameramtx):
    """Perform undistort using initUndistortRectifyMap+remap with optional supersampling.

    Adaptive upsample factor based on max |D| to reduce pixelation for strong distortion.
    Returns undistorted image (same shape as src).
    """
    if src is None:
        return None
    h, w = src.shape[:2]
    try:
        max_abs = float(np.max(np.abs(D))) if D is not None else 0.0
    except Exception:
        max_abs = 0.0
    # choose scale: aggressive when distortion large
    if max_abs > 1.0:
        scale = 3
    elif max_abs > 0.6:
        scale = 2
    else:
        scale = 1

    try:
        if scale == 1:
            mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (w, h), cv2.CV_32FC1)
            und = cv2.remap(src, mapx, mapy, interpolation=cv2.INTER_LANCZOS4)
            return und
        # supersample path
        w2 = int(w * scale)
        h2 = int(h * scale)
        src_up = cv2.resize(src, (w2, h2), interpolation=cv2.INTER_CUBIC)
        # scale intrinsics
        K_scaled = K.astype(np.float64).copy()
        newmtx_scaled = newcameramtx.astype(np.float64).copy()
        K_scaled[0, 0] *= scale
        K_scaled[1, 1] *= scale
        K_scaled[0, 2] *= scale
        K_scaled[1, 2] *= scale
        newmtx_scaled[0, 0] *= scale
        newmtx_scaled[1, 1] *= scale
        newmtx_scaled[0, 2] *= scale
        newmtx_scaled[1, 2] *= scale
        mapx, mapy = cv2.initUndistortRectifyMap(K_scaled, D, None, newmtx_scaled, (w2, h2), cv2.CV_32FC1)
        und_up = cv2.remap(src_up, mapx, mapy, interpolation=cv2.INTER_LANCZOS4)
        # downsample back to original size using area resampling
        und = cv2.resize(und_up, (w, h), interpolation=cv2.INTER_AREA)
        return und
    except Exception:
        try:
            return cv2.undistort(src, K, D, None, newcameramtx)
        except Exception:
            return None



class ImageViewer(QGraphicsView):
    # emits scene (image) coordinates as floats (x, y)
    clicked = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pixmap_item = None
        self._overlay_item = None
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
        # clear any overlay reference
        self._overlay_item = None
        self.setSceneRect(self._pixmap_item.boundingRect())
        self._zoom = 0

    def set_overlay_rect(self, x: int, y: int, w: int, h: int):
        try:
            # remove existing overlay
            if self._overlay_item is not None:
                try:
                    self.scene().removeItem(self._overlay_item)
                except Exception:
                    pass
                self._overlay_item = None
            # create new rect
            rect = QGraphicsRectItem(x, y, w, h)
            pen = rect.pen()
            pen.setWidth(2)
            from PyQt5.QtGui import QColor
            pen.setColor(QColor(0, 255, 0))
            rect.setPen(pen)
            rect.setBrush(QColor(0, 0, 0, 0))
            rect.setZValue(2)
            self._overlay_item = self.scene().addItem(rect)
        except Exception:
            pass

    def clear_overlay(self):
        try:
            if self._overlay_item is not None:
                self.scene().removeItem(self._overlay_item)
                self._overlay_item = None
        except Exception:
            pass

    def fitToView(self):
        if self._pixmap_item is None:
            return
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._pixmap_item is None:
            return
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        try:
            if self._pixmap_item is not None and event.button() == Qt.LeftButton:
                # map mouse pos to scene coords which correspond to image pixels
                pt = self.mapToScene(event.pos())
                self.clicked.emit(float(pt.x()), float(pt.y()))
                # continue to base handling for panning if enabled
        except Exception:
            pass
        super().mousePressEvent(event)


class UndistortStage(QWidget):
    """Interactive undistort stage: sliders for k1,k2,k3,p1,p2 with per-coefficient ranges.

    OpenCV `D` vector ordering is `[k1, k2, p1, p2, k3]` internally; the UI shows controls
    for k1, k2, k3, p1, p2 but maps k3 -> D[4], p1->D[2], p2->D[3].
    """

    COEFF_ORDER = ["k1", "k2", "k3", "p1", "p2"]
    # mapping from UI index to D array index
    UI_TO_D_IDX = {0: 0, 1: 1, 2: 4, 3: 2, 4: 3}

    DEFAULT_RANGES = {
        "k1": (-1.2, 1.2),
        "k2": (-1.0, 1.0),
        "p1": (-0.01, 0.01),
        "p2": (-0.01, 0.01),
        "k3": (-2.0, 2.0),
    }

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root

        self._original_qimage = None

        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(8)

        title = QLabel("Undistort")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        main.addWidget(title)

        self.viewer = ImageViewer()
        self.viewer.setMinimumHeight(360)
        main.addWidget(self.viewer)

        # Controls container
        self.control_rows = []
        self.sliders = []
        self.min_boxes = []
        self.max_boxes = []
        self.set_buttons = []
        self.reset_buttons = []

        for ui_idx, name in enumerate(self.COEFF_ORDER):
            row = QHBoxLayout()
            lbl = QLabel(f"{name}:")
            lbl.setFixedWidth(36)
            row.addWidget(lbl)

            min_le = QLineEdit()
            min_le.setFixedWidth(80)
            max_le = QLineEdit()
            max_le.setFixedWidth(80)
            # set defaults
            mn, mx = self.DEFAULT_RANGES.get(name, (-1.0, 1.0))
            min_le.setText(str(mn))
            max_le.setText(str(mx))

            row.addWidget(QLabel("min"))
            row.addWidget(min_le)
            row.addWidget(QLabel("max"))
            row.addWidget(max_le)

            set_btn = QPushButton("Set Range")
            set_btn.setProperty("ui_idx", ui_idx)
            set_btn.clicked.connect(self._on_set_range)
            row.addWidget(set_btn)
            self.set_buttons.append(set_btn)

            # enable/disable set button depending on whether 0 is inside min/max
            def _wire_textchange(le_min, le_max, idx):
                le_min.textChanged.connect(lambda _t, i=idx: self._on_range_text_changed(i))
                le_max.textChanged.connect(lambda _t, i=idx: self._on_range_text_changed(i))

            _wire_textchange(min_le, max_le, ui_idx)

            reset_btn = QPushButton("Reset")
            reset_btn.setProperty("ui_idx", ui_idx)
            reset_btn.clicked.connect(self._on_reset)
            row.addWidget(reset_btn)
            self.reset_buttons.append(reset_btn)

            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 1000)
            slider.setProperty("ui_idx", ui_idx)
            slider.valueChanged.connect(self._on_slider_changed)
            slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row.addWidget(slider)

            val_lbl = QLabel("0.0000")
            val_lbl.setFixedWidth(80)
            row.addWidget(val_lbl)

            self.min_boxes.append(min_le)
            self.max_boxes.append(max_le)
            self.sliders.append((slider, val_lbl))
            # default set button state (defaults include 0)
            try:
                mn, mx = self._ranges[ui_idx]
            except Exception:
                mn, mx = (0.0, 0.0)
            enabled = (mn < mx and mn <= 0.0 <= mx)
            if self.set_buttons:
                self.set_buttons[-1].setEnabled(bool(enabled))
            self.control_rows.append(row)
            main.addLayout(row)

        # status / info
        self.status = QLabel("")
        main.addWidget(self.status)

        # pen / arc controls
        pen_row = QHBoxLayout()
        self.pen_mode_button = QPushButton("Pen Mode")
        self.pen_mode_button.setCheckable(True)
        self.pen_mode_button.toggled.connect(self._on_pen_toggled)
        pen_row.addWidget(self.pen_mode_button)

        self.activate_pen_button = QPushButton("Activate/Deactivate Pen")
        self.activate_pen_button.setCheckable(True)
        self.activate_pen_button.setEnabled(False)
        self.activate_pen_button.toggled.connect(self._on_activate_toggled)
        pen_row.addWidget(self.activate_pen_button)

        self.new_arc_button = QPushButton("New Arc")
        self.new_arc_button.setEnabled(False)
        self.new_arc_button.clicked.connect(self._on_new_arc)
        pen_row.addWidget(self.new_arc_button)

        self.clear_arcs_button = QPushButton("Clear All Arcs")
        self.clear_arcs_button.setEnabled(False)
        self.clear_arcs_button.clicked.connect(self._on_clear_arcs)
        pen_row.addWidget(self.clear_arcs_button)

        main.addLayout(pen_row)

        # proceed button -> go to Validation 1 (index 3)
        proc_row = QHBoxLayout()
        proc_row.addStretch(1)
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.clicked.connect(self._on_proceed)
        proc_row.addWidget(self.proceed_button)
        main.addLayout(proc_row)

        # initialize internal state
        self._ranges = [self.DEFAULT_RANGES[name] for name in self.COEFF_ORDER]
        self._current_D = [0.0, 0.0, 0.0, 0.0, 0.0]

        # arc drawing state
        self.pen_mode = False
        # arcs: list of list-of-(x,y) in original image pixel coordinates
        self.arcs = []
        # current in-progress arc (list of points) or None
        self.current_arc = None
        # whether pen is active (will record clicks). Can be deactivated to allow pan/zoom
        self.pen_active = False

        # connect viewer clicks
        try:
            self.viewer.clicked.connect(self._on_viewer_clicked)
        except Exception:
            pass

    def _on_proceed(self):
        """Advance to Validation 1 (index 3) without confirmation."""
        host = getattr(self, 'host_tab', None) or self.parent()
        if host is None:
            return
        try:
            host.current_step_index = 3
            host._show_stage(3)
            QTimer.singleShot(0, lambda: host._update_progress_to_index(3))
        except Exception:
            pass

    def showEvent(self, event):
        super().showEvent(event)
        # load CCTV image and populate sliders from inspect_obj if available
        host = getattr(self, 'host_tab', None) or self.parent()
        if host is None:
            return
        obj = getattr(host, 'inspect_obj', None)
        if obj is None:
            return

        # set initial D from inspect_obj if present
        und = obj.get('undistort', {})
        D = und.get('D', [0.0, 0.0, 0.0, 0.0, 0.0])
        # ensure length 5
        if not isinstance(D, (list, tuple)) or len(D) != 5:
            D = [0.0] * 5
        self._current_D = list(D)

        # populate sliders to match D
        for ui_idx, name in enumerate(self.COEFF_ORDER):
            d_idx = self.UI_TO_D_IDX[ui_idx]
            val = float(self._current_D[d_idx])
            mn, mx = self._ranges[ui_idx]
            # if inspect_obj had values outside current range, expand range
            if val < mn:
                mn = val
            if val > mx:
                mx = val
            self._ranges[ui_idx] = (mn, mx)
            self.min_boxes[ui_idx].setText(str(mn))
            self.max_boxes[ui_idx].setText(str(mx))
            slider, lbl = self.sliders[ui_idx]
            # position slider proportionally
            pos = int(0 if mx == mn else round(1000 * (val - mn) / (mx - mn)))
            slider.blockSignals(True)
            slider.setValue(pos)
            slider.blockSignals(False)
            lbl.setText(f"{val:.6f}")

        # load CCTV original image if possible
        loc_code = obj.get('meta', {}).get('location_code')
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        if loc_code:
            p = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
            if os.path.isfile(p):
                qimg = QImage(p)
                if not qimg.isNull():
                    self._original_qimage = qimg
                    pix = QPixmap.fromImage(qimg)
                    self.viewer.load_pixmap(pix)
                    self.viewer.fitToView()

    def _on_set_range(self):
        btn = self.sender()
        ui_idx = btn.property('ui_idx')
        try:
            mn = float(self.min_boxes[ui_idx].text().strip())
            mx = float(self.max_boxes[ui_idx].text().strip())
        except Exception:
            self.status.setText('Invalid min/max')
            return
        if mn >= mx:
            self.status.setText('Min must be < max')
            return
        self._ranges[ui_idx] = (mn, mx)
        # update slider position to current value proportionally
        d_idx = self.UI_TO_D_IDX[ui_idx]
        cur = float(self._current_D[d_idx])
        pos = int(round(1000 * (cur - mn) / (mx - mn))) if mx != mn else 0
        slider, lbl = self.sliders[ui_idx]
        slider.blockSignals(True)
        slider.setValue(max(0, min(1000, pos)))
        slider.blockSignals(False)
        lbl.setText(f"{cur:.6f}")
        self.status.setText(f'Range set for {self.COEFF_ORDER[ui_idx]}')

    def _on_range_text_changed(self, ui_idx: int):
        """Enable Set Range button only when min < max and 0 is between them."""
        try:
            mn = float(self.min_boxes[ui_idx].text())
            mx = float(self.max_boxes[ui_idx].text())
            enabled = (mn < mx and mn <= 0.0 <= mx)
        except Exception:
            enabled = False
        try:
            self.set_buttons[ui_idx].setEnabled(bool(enabled))
        except Exception:
            pass

    def _on_reset(self):
        btn = self.sender()
        ui_idx = btn.property('ui_idx')
        # set to zero within current range
        mn, mx = self._ranges[ui_idx]
        zero = 0.0
        # clamp zero into range if outside
        if zero < mn:
            zero = mn
        if zero > mx:
            zero = mx
        # compute slider position
        pos = int(round(1000 * (zero - mn) / (mx - mn))) if mx != mn else 0
        slider, lbl = self.sliders[ui_idx]
        slider.blockSignals(True)
        slider.setValue(max(0, min(1000, pos)))
        slider.blockSignals(False)
        lbl.setText(f"{zero:.6f}")
        # update internal D and inspect_obj and preview
        d_idx = self.UI_TO_D_IDX[ui_idx]
        self._current_D[d_idx] = float(zero)
        host = getattr(self, 'host_tab', None) or self.parent()
        if host is not None:
            obj = getattr(host, 'inspect_obj', None)
            if obj is not None:
                und = obj.setdefault('undistort', {})
                und['D'] = list(self._current_D)
                host.inspect_obj = obj

        self._update_preview()

    def _on_slider_changed(self, v: int):
        slider = self.sender()
        ui_idx = slider.property('ui_idx')
        mn, mx = self._ranges[ui_idx]
        val = mn + (mx - mn) * (v / 1000.0)
        # update display label
        lbl = self.sliders[ui_idx][1]
        lbl.setText(f"{val:.6f}")

        # write into current D using mapping
        d_idx = self.UI_TO_D_IDX[ui_idx]
        self._current_D[d_idx] = float(val)

        # update inspect_obj immediately
        host = getattr(self, 'host_tab', None) or self.parent()
        if host is not None:
            obj = getattr(host, 'inspect_obj', None)
            if obj is not None:
                und = obj.setdefault('undistort', {})
                und['D'] = list(self._current_D)
                host.inspect_obj = obj

        # preview undistort live
        self._update_preview()

    def _on_pen_toggled(self, checked: bool):
        if checked:
            self._enter_pen_mode()
        else:
            self._exit_pen_mode()

    def _on_activate_toggled(self, checked: bool):
        # toggle whether clicks are recorded as drawing; allow pan/zoom when deactivated
        self.pen_active = bool(checked)
        try:
            if self.pen_active:
                self.viewer.setDragMode(QGraphicsView.NoDrag)
            else:
                self.viewer.setDragMode(QGraphicsView.ScrollHandDrag)
        except Exception:
            pass

    def _enter_pen_mode(self):
        # clear existing arcs and set coefficients to zero; disable coeff controls
        self.pen_mode = True
        self.arcs = []
        self.current_arc = []
        # zero coefficients and update sliders/labels
        for ui_idx in range(len(self.COEFF_ORDER)):
            mn, mx = self._ranges[ui_idx]
            zero = 0.0
            if zero < mn:
                zero = mn
            if zero > mx:
                zero = mx
            # slider pos
            pos = int(round(1000 * (zero - mn) / (mx - mn))) if mx != mn else 0
            slider, lbl = self.sliders[ui_idx]
            slider.blockSignals(True)
            slider.setValue(max(0, min(1000, pos)))
            slider.blockSignals(False)
            lbl.setText(f"{zero:.6f}")
            d_idx = self.UI_TO_D_IDX[ui_idx]
            self._current_D[d_idx] = float(zero)

        # update inspect_obj
        host = getattr(self, 'host_tab', None) or self.parent()
        if host is not None:
            obj = getattr(host, 'inspect_obj', None)
            if obj is not None:
                und = obj.setdefault('undistort', {})
                und['D'] = list(self._current_D)
                host.inspect_obj = obj

        # disable coefficient controls
        try:
            for le in self.min_boxes:
                le.setEnabled(False)
            for le in self.max_boxes:
                le.setEnabled(False)
            for btn in self.set_buttons:
                btn.setEnabled(False)
            for btn in self.reset_buttons:
                btn.setEnabled(False)
            for slider, _lbl in self.sliders:
                slider.setEnabled(False)
        except Exception:
            pass

        # enable arc UI and activate pen by default
        self.new_arc_button.setEnabled(True)
        self.clear_arcs_button.setEnabled(True)
        self.activate_pen_button.setEnabled(True)
        self.activate_pen_button.blockSignals(True)
        self.activate_pen_button.setChecked(True)
        self.activate_pen_button.blockSignals(False)
        self.pen_active = True

        # set drag mode according to pen_active
        try:
            if self.pen_active:
                self.viewer.setDragMode(QGraphicsView.NoDrag)
            else:
                self.viewer.setDragMode(QGraphicsView.ScrollHandDrag)
        except Exception:
            pass

        self._update_preview()

    def _exit_pen_mode(self):
        # leaving pen mode: re-enable controls; arcs remain as painted pixels
        self.pen_mode = False
        try:
            for le in self.min_boxes:
                le.setEnabled(True)
            for le in self.max_boxes:
                le.setEnabled(True)
            for btn in self.set_buttons:
                btn.setEnabled(True)
            for btn in self.reset_buttons:
                btn.setEnabled(True)
            for slider, _lbl in self.sliders:
                slider.setEnabled(True)
        except Exception:
            pass

        self.new_arc_button.setEnabled(False)
        self.clear_arcs_button.setEnabled(False)
        try:
            self.activate_pen_button.setEnabled(False)
            self.activate_pen_button.blockSignals(True)
            self.activate_pen_button.setChecked(False)
            self.activate_pen_button.blockSignals(False)
        except Exception:
            pass

        try:
            self.viewer.setDragMode(QGraphicsView.ScrollHandDrag)
        except Exception:
            pass

        # update preview so arcs get applied with current coefficients
        self._update_preview()
        try:
            # when exiting pen mode, re-fit the view to reset to image framing
            self.viewer.fitToView()
        except Exception:
            pass

    def _on_new_arc(self):
        # start a new current arc (close previous if empty)
        if self.current_arc is None:
            self.current_arc = []
        else:
            # if current has points, store it
            if len(self.current_arc) >= 1:
                self.arcs.append(self.current_arc)
            self.current_arc = []
        self._update_preview()

    def _on_clear_arcs(self):
        self.arcs = []
        self.current_arc = None
        self._update_preview()

    def _on_viewer_clicked(self, x: float, y: float):
        # only record clicks in pen mode when pen is active and when original image available
        if not self.pen_mode or not self.pen_active or self._original_qimage is None:
            return
        h = self._original_qimage.height()
        w = self._original_qimage.width()
        ix = int(round(x))
        iy = int(round(y))
        # clamp
        ix = max(0, min(w - 1, ix))
        iy = max(0, min(h - 1, iy))
        if self.current_arc is None:
            self.current_arc = []
        self.current_arc.append((ix, iy))
        self._update_preview()

    def _qimage_to_cv(self, qimg: QImage):
        if qimg is None:
            return None
        img = qimg.convertToFormat(QImage.Format_RGB888)
        w = img.width()
        h = img.height()
        ptr = img.bits()
        ptr.setsize(img.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 3))
        return arr[:, :, ::-1].copy()

    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None:
            return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        return qimg.copy()

    def _update_preview(self):
        if not HAS_CV2:
            self.status.setText('OpenCV not available; preview disabled')
            return
        if self._original_qimage is None:
            self.status.setText('No CCTV image to preview')
            return

        src = self._qimage_to_cv(self._original_qimage)
        if src is None:
            self.status.setText('Failed to read image')
            return

        # build K from inspect_obj if available, else identity-like
        host = getattr(self, 'host_tab', None) or self.parent()
        K = None
        if host is not None:
            obj = getattr(host, 'inspect_obj', None)
            if obj is not None:
                und = obj.get('undistort', {})
                K_list = und.get('K')
                if isinstance(K_list, (list, tuple)) and len(K_list) >= 3:
                    K = np.array(K_list, dtype=np.float64)

        h, w = src.shape[:2]
        if K is None:
            # fallback focal lengths to image height
            fx = float(max(w, h))
            fy = fx
            cx = w / 2.0
            cy = h / 2.0
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        # when in pen mode coefficients are forced to zero for drawing
        if self.pen_mode:
            D = np.zeros((5,), dtype=np.float64)
        else:
            D = np.array(self._current_D, dtype=np.float64)

        # draw arcs (if any) onto a copy of the source image so they are
        # transformed by the undistort operation below
        draw_src = src.copy()
        try:
            all_arcs = list(self.arcs) if self.arcs is not None else []
            if self.current_arc:
                all_arcs = all_arcs + [self.current_arc]
            for arc in all_arcs:
                if not arc:
                    continue
                if len(arc) == 1:
                    x, y = arc[0]
                    cv2.circle(draw_src, (int(x), int(y)), 3, (0, 0, 255), -1)
                else:
                    pts = np.array(arc, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(draw_src, [pts], False, (0, 0, 255), thickness=2)
        except Exception:
            pass

        try:
            # FIX: Use K.copy() instead of getOptimalNewCameraMatrix to match reference code
            # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
            newcameramtx = K.copy()
            roi = None

            # use supersampled remap to reduce pixelation on strong distortion
            undist = remap_with_supersample(draw_src, K, D, newcameramtx)
            # after undistortion: draw straight aqua connectors between transformed endpoints
            try:
                for arc in all_arcs:
                    if not arc:
                        continue
                    # build points array for undistortPoints
                    pts = np.array(arc, dtype=np.float32).reshape((-1, 1, 2))
                    # map to undistorted image coordinates using new camera matrix
                    try:
                        tpts = cv2.undistortPoints(pts, K, D, P=newcameramtx)
                    except Exception:
                        tpts = None
                    if tpts is None:
                        continue
                    # tpts shape (N,1,2)
                    if tpts.shape[0] >= 2:
                        x0, y0 = tpts[0, 0]
                        x1, y1 = tpts[-1, 0]
                        cv2.line(undist, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), (255, 255, 0), thickness=2)
                    # redraw the red arc on top so it overlays the aqua connector
                    try:
                        if tpts.shape[0] == 1:
                            x, y = tpts[0, 0]
                            cv2.circle(undist, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1)
                        else:
                            pts2 = (tpts.astype(np.int32)).reshape((-1, 1, 2))
                            cv2.polylines(undist, [pts2], False, (0, 0, 255), thickness=2)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            self.status.setText(f'Undistort failed: {e}')
            return

        qimg = self._cv_to_qimage(undist)
        if qimg is not None:
            pix = QPixmap.fromImage(qimg)
            self.viewer.load_pixmap(pix)
            # draw the ROI rectangle (x,y,w,h) if provided by OpenCV
            try:
                if roi is not None and isinstance(roi, (list, tuple)) and len(roi) == 4:
                    x, y, rw, rh = roi
                    # viewer coordinates match image pixel coords
                    self.viewer.set_overlay_rect(int(x), int(y), int(rw), int(rh))
                else:
                    self.viewer.clear_overlay()
            except Exception:
                try:
                    self.viewer.clear_overlay()
                except Exception:
                    pass
            # don't reset the view while drawing in pen mode (preserve pan/zoom)
            try:
                if not self.pen_mode:
                    self.viewer.fitToView()
            except Exception:
                pass
            self.status.setText('Preview updated')
        else:
            self.status.setText('Preview conversion failed')