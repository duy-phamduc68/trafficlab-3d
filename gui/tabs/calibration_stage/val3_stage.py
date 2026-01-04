import os
import numpy as np
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QSplitter, QDoubleSpinBox, QGroupBox, QRadioButton, QButtonGroup, QGraphicsView
)

from .undistort_stage import ImageViewer, remap_with_supersample

class RightClickImageViewer(ImageViewer):
    """
    Captures Right-Clicks for point selection.
    Left-Click is strictly for Panning (ScrollHandDrag).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._pixmap_item is not None and event.button() == Qt.RightButton:
            # Right Click: Map coords and emit signal
            pt = self.mapToScene(event.pos())
            self.clicked.emit(float(pt.x()), float(pt.y()))
            event.accept()
        else:
            # Left Click: Pan
            QGraphicsView.mousePressEvent(self, event)

class Val3Stage(QWidget):
    """
    Validation 3: Parallax & Projection Verification.
    
    Interactive Modes:
    1. Right-Click CCTV (Head) -> Shows Feet on Map + Head/Feet on CCTV.
    2. Right-Click Map (Feet) -> Shows Head & Feet on CCTV.
    """

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        
        # Internal Data
        self._K = None
        self._D = None
        self._new_K = None
        self._H = None
        self._H_inv = None
        self._cam_sat = None 
        self._z_cam = None   
        self._px_per_m = 1.0 
        
        # --- UI ---
        main_layout = QHBoxLayout(self)
        
        # 1. Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(15)
        
        lbl_title = QLabel("Validation 3: Parallax")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #fff;")
        side_vbox.addWidget(lbl_title)
        
        desc = QLabel(
            "Verify the 3D model accuracy.\n"
            "Adjust 'Object Height' to match the target (e.g., 1.7m person)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; font-size: 12px;")
        side_vbox.addWidget(desc)
        
        # Controls
        grp_h = QGroupBox("Simulation Params")
        l_h = QVBoxLayout(grp_h)
        
        l_h.addWidget(QLabel("Object Height (m):"))
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.1, 5.0)
        self.spin_height.setSingleStep(0.1)
        self.spin_height.setValue(1.7)
        l_h.addWidget(self.spin_height)
        
        side_vbox.addWidget(grp_h)
        
        # Interaction Mode
        grp_mode = QGroupBox("Right-Click Mode")
        l_mode = QVBoxLayout(grp_mode)
        self.rb_cctv = QRadioButton("Click CCTV (Head)")
        self.rb_cctv.setChecked(True)
        self.rb_sat = QRadioButton("Click Map (Feet)")
        l_mode.addWidget(self.rb_cctv)
        l_mode.addWidget(self.rb_sat)
        
        self.btn_grp = QButtonGroup()
        self.btn_grp.addButton(self.rb_cctv)
        self.btn_grp.addButton(self.rb_sat)
        
        side_vbox.addWidget(grp_mode)
        
        self.btn_clear = QPushButton("Clear Markers")
        self.btn_clear.clicked.connect(self._on_clear)
        side_vbox.addWidget(self.btn_clear)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #ffd700;")
        side_vbox.addWidget(self.lbl_status)
        
        side_vbox.addStretch()
        
        self.btn_proceed = QPushButton("Proceed")
        self.btn_proceed.clicked.connect(self._on_proceed)
        side_vbox.addWidget(self.btn_proceed)
        
        main_layout.addWidget(sidebar)
        
        # 2. Viewers (Using RightClickImageViewer)
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Undistorted CCTV
        left_cont = QWidget()
        l_vbox = QVBoxLayout(left_cont)
        l_vbox.setContentsMargins(0,0,0,0)
        l_vbox.addWidget(QLabel("Undistorted CCTV (Right-Click Head)"))
        self.view_cctv = RightClickImageViewer()
        self.view_cctv.clicked.connect(self._on_cctv_click)
        l_vbox.addWidget(self.view_cctv)
        
        # Right: Satellite
        right_cont = QWidget()
        r_vbox = QVBoxLayout(right_cont)
        r_vbox.setContentsMargins(0,0,0,0)
        r_vbox.addWidget(QLabel("Satellite Map (Right-Click Feet)"))
        self.view_sat = RightClickImageViewer()
        self.view_sat.clicked.connect(self._on_sat_click)
        r_vbox.addWidget(self.view_sat)
        
        splitter.addWidget(left_cont)
        splitter.addWidget(right_cont)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter, 1)

    def showEvent(self, event):
        super().showEvent(event)
        if not HAS_CV2: return
        
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None): return
        obj = host.inspect_obj
        
        # 1. Load Parameters
        try:
            # Undistort
            und = obj.get('undistort', {})
            K_list = und.get('K')
            D_list = und.get('D', [0]*5)
            
            if K_list:
                self._K = np.array(K_list, dtype=np.float64)
            self._D = np.array(D_list, dtype=np.float64)
            
            # Homography
            hom = obj.get('homography', {})
            H_list = hom.get('H')
            if H_list:
                self._H = np.array(H_list, dtype=np.float64)
                try:
                    self._H_inv = np.linalg.inv(self._H)
                except:
                    self._H_inv = None
            
            # Parallax
            par = obj.get('parallax', {})
            self._z_cam = par.get('z_cam_meters', 10.0) 
            self._cam_sat = np.array([
                par.get('x_cam_coords_sat', 0.0),
                par.get('y_cam_coords_sat', 0.0)
            ], dtype=np.float32)
            
        except Exception as e:
            self.lbl_status.setText(f"Error loading params: {e}")

        # 2. Load Images
        self._load_images(obj)

    def _load_images(self, obj):
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        if not loc_code: return

        # CCTV
        cctv_path = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
        if os.path.isfile(cctv_path):
            src = cv2.imread(cctv_path)
            if src is not None:
                h, w = src.shape[:2]
                
                # Compute Optimal New K
                if self._K is not None:
                    # FIX: Use K.copy() instead of getOptimalNewCameraMatrix to match reference pipeline
                    # self._new_K, roi = cv2.getOptimalNewCameraMatrix(self._K, self._D, (w, h), 1, (w, h))
                    self._new_K = self._K.copy()
                    
                    undist = remap_with_supersample(src, self._K, self._D, self._new_K)
                else:
                    undist = src 
                
                self.view_cctv.load_pixmap(QPixmap.fromImage(self._cv_to_qimage(undist)))
                self.view_cctv.fitToView()

        # Satellite
        sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
        if os.path.isfile(sat_path):
            sat = cv2.imread(sat_path)
            if sat is not None:
                self.view_sat.load_pixmap(QPixmap.fromImage(self._cv_to_qimage(sat)))
                self.view_sat.fitToView()

    def _on_cctv_click(self, u, v):
        if not self.rb_cctv.isChecked(): return
        if self._H is None or self._cam_sat is None: 
            self.lbl_status.setText("Missing Calibration Data")
            return
            
        self._on_clear()
        
        # 1. Project Head to Map (Apparent Head)
        src_h = np.array([[[u, v]]], dtype=np.float64)
        sat_head_apparent = cv2.perspectiveTransform(src_h, self._H)[0, 0]
        
        # 2. Parallax Correction (Head -> Feet)
        vec = sat_head_apparent - self._cam_sat
        h_obj = self.spin_height.value()
        
        if self._z_cam == 0: factor = 1.0
        else: factor = (self._z_cam - h_obj) / self._z_cam
        
        sat_feet = self._cam_sat + (vec * factor)
        
        # 3. Reproject Feet to CCTV
        cctv_feet = self._sat_to_cctv(sat_feet)
        
        # 4. Draw
        # CCTV: Head (Red) -> Feet (Cyan) with Green Line
        self._draw_marker(self.view_cctv, u, v, "H", Qt.red)
        self._draw_marker(self.view_cctv, cctv_feet[0], cctv_feet[1], "F", Qt.cyan)
        self._draw_line(self.view_cctv, (u,v), cctv_feet, Qt.green)
        
        # Sat: ONLY Feet (Cyan) - No camera rays or apparent head
        self._draw_marker(self.view_sat, sat_feet[0], sat_feet[1], "F", Qt.cyan)

    def _on_sat_click(self, x, y):
        if not self.rb_sat.isChecked(): return
        if self._H is None or self._cam_sat is None: return
        
        self._on_clear()
        
        sat_feet = np.array([x, y], dtype=np.float64)
        
        # 1. Project Feet to CCTV (Direct)
        cctv_feet = self._sat_to_cctv(sat_feet)
        
        # 2. Find Apparent Head on Map
        vec = sat_feet - self._cam_sat
        h_obj = self.spin_height.value()
        
        if abs(self._z_cam - h_obj) < 0.1: factor = 10.0 
        else: factor = self._z_cam / (self._z_cam - h_obj)
        
        sat_head_app = self._cam_sat + (vec * factor)
        
        # 3. Project Head to CCTV
        cctv_head = self._sat_to_cctv(sat_head_app)
        
        # 4. Draw
        # Sat: Only Feet
        self._draw_marker(self.view_sat, x, y, "F", Qt.cyan)
        
        # CCTV: Head + Feet + Line
        self._draw_marker(self.view_cctv, cctv_feet[0], cctv_feet[1], "F", Qt.cyan)
        self._draw_marker(self.view_cctv, cctv_head[0], cctv_head[1], "H", Qt.red)
        self._draw_line(self.view_cctv, cctv_feet, cctv_head, Qt.green)

    def _sat_to_cctv(self, pt_sat):
        # Inverse Homography: Map -> Undistorted Pixel
        src = np.array([[[pt_sat[0], pt_sat[1]]]], dtype=np.float64)
        pt_undist = cv2.perspectiveTransform(src, self._H_inv)[0, 0]
        return pt_undist

    def _on_clear(self):
        self._clear_scene_items(self.view_cctv)
        self._clear_scene_items(self.view_sat)

    def _clear_scene_items(self, viewer):
        from PyQt5.QtWidgets import QGraphicsPixmapItem
        scene = viewer.scene()
        if not scene: return
        for item in scene.items():
            if not isinstance(item, QGraphicsPixmapItem):
                scene.removeItem(item)

    def _draw_marker(self, viewer, x, y, label, color):
        scene = viewer.scene()
        rad = 4
        el = scene.addEllipse(x-rad, y-rad, rad*2, rad*2)
        el.setPen(QPen(color, 2))
        el.setBrush(QBrush(color))
        
        t = scene.addSimpleText(label)
        t.setBrush(QBrush(color))
        t.setPos(x+5, y-10)

    def _draw_line(self, viewer, p1, p2, color, dashed=False):
        scene = viewer.scene()
        pen = QPen(color, 2)
        if dashed: pen.setStyle(Qt.DashLine)
        scene.addLine(p1[0], p1[1], p2[0], p2[1], pen)

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host: return
        
        # --- BRANCHING LOGIC ---
        obj = getattr(host, 'inspect_obj', {})
        use_svg = obj.get('use_svg', False)
        use_roi = obj.get('use_roi', False)
        
        next_idx = 10 # Default to SVG
        
        if use_svg:
            next_idx = 10
        elif use_roi:
            next_idx = 11
        else:
            next_idx = 12 # Jump straight to Final if neither are used

        if hasattr(host, 'current_step_index'):
            try:
                host.current_step_index = next_idx
                host._show_stage(next_idx)
                host._update_progress_to_index(next_idx)
            except: pass

    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None: return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()