import os
import numpy as np
import json
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QFont
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QSplitter, QMessageBox, QGroupBox, QSizePolicy
)

# Use ImageViewer base, redefine RightClick logic locally to keep file self-contained
from .undistort_stage import ImageViewer, remap_with_supersample

class RightClickImageViewer(ImageViewer):
    """
    Subclass that captures Right-Clicks for point validation.
    Left-Click remains for Panning.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(self.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._pixmap_item is not None and event.button() == Qt.RightButton:
            pt = self.mapToScene(event.pos())
            self.clicked.emit(float(pt.x()), float(pt.y()))
            event.accept()
        else:
            # Bypass ImageViewer's left-click signal, go straight to QGraphicsView pan
            super(ImageViewer, self).mousePressEvent(event)

class Val2Stage(QWidget):
    """
    Validation 2 Stage: Pipeline Verification.
    User clicks on RAW (Distorted) CCTV image -> System undistorts & projects -> Shows on Sat map.
    """

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        
        # Internal state
        self._raw_cctv = None  # Original distorted image
        self._K = None
        self._D = None
        self._new_K = None     # Optimal new camera matrix
        self._H = None
        
        # UI
        main_layout = QHBoxLayout(self)
        
        # 1. Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(15)
        
        lbl_title = QLabel("Validation 2: Projection")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #fff;")
        side_vbox.addWidget(lbl_title)
        
        desc = QLabel(
            "Verify the full pipeline:\n"
            "1. Click a point on the GROUND in the Raw CCTV view.\n"
            "2. System undistorts it.\n"
            "3. System applies Homography.\n"
            "4. Result appears on Satellite map."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; font-size: 12px;")
        side_vbox.addWidget(desc)
        
        self.lbl_status = QLabel("Status: Waiting for data...")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #ccc; font-style: italic;")
        side_vbox.addWidget(self.lbl_status)
        
        side_vbox.addStretch()
        
        # Proceed
        self.btn_proceed = QPushButton("Proceed")
        self.btn_proceed.clicked.connect(self._on_proceed)
        side_vbox.addWidget(self.btn_proceed)
        
        main_layout.addWidget(sidebar)
        
        # 2. Splitter for Viewers
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Raw CCTV
        left_cont = QWidget()
        l_vbox = QVBoxLayout(left_cont)
        l_vbox.setContentsMargins(0,0,0,0)
        l_vbox.addWidget(QLabel("Input: Raw CCTV (Right-Click Ground)"))
        self.view_cctv = RightClickImageViewer()
        self.view_cctv.clicked.connect(self._on_cctv_click)
        l_vbox.addWidget(self.view_cctv)
        
        # Right: Satellite
        right_cont = QWidget()
        r_vbox = QVBoxLayout(right_cont)
        r_vbox.setContentsMargins(0,0,0,0)
        r_vbox.addWidget(QLabel("Output: Projected Location"))
        self.view_sat = ImageViewer() # Standard viewer, we just draw on it
        r_vbox.addWidget(self.view_sat)
        
        splitter.addWidget(left_cont)
        splitter.addWidget(right_cont)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter, 1)

    def showEvent(self, event):
        super().showEvent(event)
        if not HAS_CV2:
            self.lbl_status.setText("Error: OpenCV not available.")
            return

        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None):
            self.lbl_status.setText("Error: No Inspection Object found.")
            return
        
        obj = host.inspect_obj
        
        # 1. Load Matrices
        try:
            und = obj.get('undistort', {})
            K_list = und.get('K')
            D_list = und.get('D', [0]*5)
            
            # Default K if missing (though should be present by now)
            w_res, h_res = 1280, 720 # Fallback
            if 'resolution' in und:
                w_res, h_res = und['resolution']
            
            if K_list:
                self._K = np.array(K_list, dtype=np.float64)
            else:
                self._K = np.array([[w_res, 0, w_res/2], [0, w_res, h_res/2], [0, 0, 1]], dtype=np.float64)
                
            self._D = np.array(D_list, dtype=np.float64)
            
            hom = obj.get('homography', {})
            H_list = hom.get('H')
            if H_list:
                self._H = np.array(H_list, dtype=np.float64)
            else:
                self._H = None
                self.lbl_status.setText("Warning: Homography H not found.")
                
        except Exception as e:
            self.lbl_status.setText(f"Error loading matrices: {e}")

        # 2. Load Images
        self._load_images(obj)

    def _load_images(self, obj):
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        if not loc_code: return

        # Load Raw CCTV
        cctv_path = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
        if os.path.isfile(cctv_path):
            self._raw_cctv = cv2.imread(cctv_path)
            if self._raw_cctv is not None:
                h, w = self._raw_cctv.shape[:2]
                
                # Re-calculate New K based on actual image size
                if self._K is not None:
                    # FIX: Use K.copy() instead of getOptimalNewCameraMatrix to match reference pipeline
                    # self._new_K, roi = cv2.getOptimalNewCameraMatrix(self._K, self._D, (w, h), 1, (w, h))
                    self._new_K = self._K.copy()
                
                qimg = self._cv_to_qimage(self._raw_cctv)
                self.view_cctv.load_pixmap(QPixmap.fromImage(qimg))
                self.view_cctv.fitToView()
        
        # Load Satellite
        sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
        if os.path.isfile(sat_path):
            sat_cv = cv2.imread(sat_path)
            if sat_cv is not None:
                qimg = self._cv_to_qimage(sat_cv)
                self.view_sat.load_pixmap(QPixmap.fromImage(qimg))
                self.view_sat.fitToView()

    def _on_cctv_click(self, u, v):
        """Pipeline execution on click."""
        if not HAS_CV2 or self._H is None or self._K is None:
            return
        
        # 1. Draw Marker on CCTV (Raw)
        self._clear_markers(self.view_cctv)
        self._draw_marker(self.view_cctv, u, v, "Input", QColor(255, 0, 0)) # Red
        
        # 2. Pipeline Calculation
        try:
            # Step A: Undistort Point
            # Input: (N, 1, 2)
            src_pts = np.array([[[u, v]]], dtype=np.float64)
            
            # undistortPoints returns normalized coordinates if P is None.
            # We pass P=self._new_K to get back pixel coordinates in the Undistorted Image space
            # which matches what we used to calculate H.
            dst_undist = cv2.undistortPoints(src_pts, self._K, self._D, P=self._new_K)
            
            # Step B: Perspective Transform (Homography)
            # Input to perspectiveTransform must be (N, 1, 2)
            sat_pt_arr = cv2.perspectiveTransform(dst_undist, self._H)
            
            sx, sy = sat_pt_arr[0, 0]
            
            # 3. Draw Marker on Satellite
            self._clear_markers(self.view_sat)
            self._draw_marker(self.view_sat, sx, sy, "Proj", QColor(0, 255, 255)) # Aqua
            
            self.lbl_status.setText(f"Projected: CCTV({int(u)},{int(v)}) -> Sat({int(sx)},{int(sy)})")
            
        except Exception as e:
            self.lbl_status.setText(f"Projection Error: {e}")

    # --- Marker Utils ---
    def _draw_marker(self, viewer, x, y, label, color):
        from PyQt5.QtWidgets import QGraphicsEllipseItem
        scene = viewer.scene()
        if not scene: return
        
        # Circle
        rad = 5
        el = scene.addEllipse(x - rad, y - rad, rad*2, rad*2)
        el.setPen(QPen(color, 2))
        el.setBrush(QBrush(color))
        # Outer Ring (Yellow) for contrast
        el_out = scene.addEllipse(x - (rad+2), y - (rad+2), (rad+2)*2, (rad+2)*2)
        el_out.setPen(QPen(QColor(255, 255, 0), 2))
        
        # We assume _clear_markers is called before, so we don't need to track IDs here strictly,
        # but storing them in a list attribute on the viewer could be cleaner if we wanted multiple.

    def _clear_markers(self, viewer):
        # Remove all Ellipse items. 
        # (Assuming only markers are ellipses; images are PixmapItems)
        from PyQt5.QtWidgets import QGraphicsEllipseItem
        scene = viewer.scene()
        if not scene: return
        for item in scene.items():
            if isinstance(item, QGraphicsEllipseItem):
                scene.removeItem(item)

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and hasattr(host, 'current_step_index'):
            try:
                # Proceed to Index 7 (Parallax Subjects / ParS)
                host.current_step_index = 7
                host._show_stage(7)
                host._update_progress_to_index(7)
            except: pass

    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None: return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()