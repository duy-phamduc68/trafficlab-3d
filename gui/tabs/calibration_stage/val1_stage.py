import os
from typing import Optional

import numpy as np
try:
    import cv2
    HAS_CV2 = True
except Exception:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QGraphicsView

# reuse the ImageViewer from undistort_stage to provide pan/zoom overlay capabilities
from .undistort_stage import ImageViewer


class Val1Stage(QWidget):
    """Validation 1 placeholder: two-pan/zoom viewers.

    Left: original CCTV. Right: undistorted CCTV computed from `inspect_obj['undistort']`.
    """

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        from PyQt5.QtWidgets import QVBoxLayout
        main = QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(8)

        title = QLabel("Validation 1: Undistort")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        main.addWidget(title)

        self.left_view = ImageViewer()
        self.right_view = ImageViewer()

        # marker mode toggle
        self.marker_button = QPushButton("Marker Mode")
        self.marker_button.setCheckable(True)
        self.marker_button.toggled.connect(self._on_marker_toggled)
        # clear markers button (placed at bottom)
        self.clear_markers_button = QPushButton("Clear All Markers")
        self.clear_markers_button.clicked.connect(self._on_clear_markers)

        # label containers so users understand which pane is which
        left_box = QWidget()
        left_layout = QHBoxLayout(left_box)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(self.left_view)

        right_box = QWidget()
        right_layout = QHBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.right_view)

        # viewers side-by-side
        viewers_h = QHBoxLayout()
        viewers_h.addWidget(left_box, 1)
        viewers_h.addWidget(right_box, 1)
        main.addLayout(viewers_h)

        # place marker toggle and clear button at bottom filling width
        vbox = QHBoxLayout()
        # make buttons expand to take available width evenly
        for btn in (self.marker_button, self.clear_markers_button):
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            vbox.addWidget(btn)
        main.addLayout(vbox)
        # marker item containers
        self._left_markers = []
        self._right_markers = []

        # proceed button to Homography Anchors (index 4)
        proc_row = QHBoxLayout()
        proc_row.addStretch(1)
        self.proceed_button = QPushButton("Proceed")
        self.proceed_button.clicked.connect(self._on_proceed)
        proc_row.addWidget(self.proceed_button)
        main.addLayout(proc_row)

    def showEvent(self, event):
        super().showEvent(event)
        host = getattr(self, 'host_tab', None) or self.parent()
        if host is None:
            return
        obj = getattr(host, 'inspect_obj', None)
        if obj is None:
            self.info.setText('No inspect object')
            return

        loc_code = obj.get('meta', {}).get('location_code')
        proj_root = getattr(self, 'project_root', None) or os.getcwd()

        orig_path = None
        if loc_code:
            p = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
            if os.path.isfile(p):
                orig_path = p

        if orig_path is None:
            # self.info does not exist in the class definition above, suppress error or log
            print('No CCTV image found for this location')
            self.left_view.scene().clear()
            self.right_view.scene().clear()
            return

        qimg = QImage(orig_path)
        if qimg.isNull():
            print('Failed to load CCTV image')
            return

        # show original on left
        self.left_view.load_pixmap(QPixmap.fromImage(qimg))
        try:
            self.left_view.fitToView()
        except Exception:
            pass

        # compute undistorted using inspect_obj undistort K/D
        und = obj.get('undistort', {})
        K_list = und.get('K')
        D_list = und.get('D', [0.0, 0.0, 0.0, 0.0, 0.0])

        src = self._qimage_to_cv(qimg)
        h, w = src.shape[:2]

        K = None
        if isinstance(K_list, (list, tuple)) and len(K_list) >= 3:
            try:
                K = np.array(K_list, dtype=np.float64)
            except Exception:
                K = None

        if K is None:
            fx = float(max(w, h))
            fy = fx
            cx = w / 2.0
            cy = h / 2.0
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        D = np.array(D_list, dtype=np.float64) if D_list is not None else np.zeros((5,), dtype=np.float64)

        if HAS_CV2:
            try:
                # FIX: Use K.copy() to maintain consistency with reference code
                # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
                newcameramtx = K.copy()
                roi = None 

                # prefer supersampled remap to reduce pixelation
                try:
                    from .undistort_stage import remap_with_supersample
                    undist = remap_with_supersample(src, K, D, newcameramtx)
                except Exception:
                    undist = cv2.undistort(src, K, D, None, newcameramtx)
                # store mapping matrices for marker mapping
                self._val_src = src
                self._val_K = K
                self._val_D = D
                self._val_newcameramtx = newcameramtx
                q2 = self._cv_to_qimage(undist)
                if q2 is not None:
                    self.right_view.load_pixmap(QPixmap.fromImage(q2))
                    try:
                        self.right_view.fitToView()
                    except Exception:
                        pass
                    # draw ROI rectangle on right viewer
                    try:
                        if roi is not None and isinstance(roi, (list, tuple)) and len(roi) == 4:
                            x, y, rw, rh = roi
                            self.right_view.set_overlay_rect(int(x), int(y), int(rw), int(rh))
                    except Exception:
                        pass
            except Exception as e:
                print(f'Undistort failed: {e}')
        else:
            print('OpenCV not available; cannot compute undistorted image')

        # connect click handler for markers
        try:
            self.left_view.clicked.connect(self._on_left_clicked)
        except Exception:
            pass

        # containers for marker items so we can manage them if needed
        self._left_markers = []
        self._right_markers = []
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

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host is None:
            return
        try:
            host.current_step_index = 4
            host._show_stage(4)
            # update progress bar / buttons
            try:
                host._update_progress_to_index(4)
            except Exception:
                pass
        except Exception:
            pass

    def _on_marker_toggled(self, checked: bool):
        # when marker mode is on, disable panning so clicks register as markers
        try:
            if checked:
                self.left_view.setDragMode(QGraphicsView.NoDrag)
                self.right_view.setDragMode(QGraphicsView.NoDrag)
            else:
                self.left_view.setDragMode(QGraphicsView.ScrollHandDrag)
                self.right_view.setDragMode(QGraphicsView.ScrollHandDrag)
        except Exception:
            pass

    def _on_clear_markers(self):
        try:
            for it in list(getattr(self, '_left_markers', []) or []):
                try:
                    self.left_view.scene().removeItem(it)
                except Exception:
                    pass
            for it in list(getattr(self, '_right_markers', []) or []):
                try:
                    self.right_view.scene().removeItem(it)
                except Exception:
                    pass
        except Exception:
            pass
        self._left_markers = []
        self._right_markers = []

    def _on_left_clicked(self, x: float, y: float):
        # when marker mode is active, add an aqua marker on left and corresponding point on right
        if not getattr(self, 'marker_button', None) or not self.marker_button.isChecked():
            return
        if not HAS_CV2 or not hasattr(self, '_val_K') or self._val_K is None:
            return
        try:
            h, w = self._val_src.shape[:2]
        except Exception:
            return
        ix = int(round(max(0, min(w - 1, x))))
        iy = int(round(max(0, min(h - 1, y))))
        try:
            pen = QColor(0, 255, 255)
            brush = QColor(0, 255, 255)
            item = self.left_view.scene().addEllipse(ix-4, iy-4, 8, 8, pen, brush)
            self._left_markers.append(item)
        except Exception:
            pass

        try:
            pts = np.array([[[float(ix), float(iy)]]], dtype=np.float32)
            tpts = cv2.undistortPoints(pts, self._val_K, self._val_D, P=self._val_newcameramtx)
            if tpts is not None and tpts.shape[0] >= 1:
                ux, uy = tpts[0,0]
                ux_i = int(round(ux))
                uy_i = int(round(uy))
                item2 = self.right_view.scene().addEllipse(ux_i-4, uy_i-4, 8, 8, QColor(0,255,255), QColor(0,255,255))
                self._right_markers.append(item2)
        except Exception:
            pass