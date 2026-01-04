import os
import numpy as np
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QFont, QPolygonF
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QSplitter, QMessageBox, QDoubleSpinBox, QGroupBox, QGraphicsView
)

from .undistort_stage import ImageViewer, remap_with_supersample

class RightClickImageViewer(ImageViewer):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._pixmap_item is not None and event.button() == Qt.RightButton:
            pt = self.mapToScene(event.pos())
            self.clicked.emit(float(pt.x()), float(pt.y()))
            event.accept()
        else:
            QGraphicsView.mousePressEvent(self, event)

class ParsStage(QWidget):
    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        
        self.ref_pairs = []
        self._current_head = None 
        self._H = None
        self._loaded_loc = None
        
        main_layout = QHBoxLayout(self)
        
        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(15)
        
        lbl_title = QLabel("Parallax Subjects")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #fff;")
        side_vbox.addWidget(lbl_title)
        
        desc = QLabel(
            "Locate Camera Position (X,Y,Z).\n\n"
            "1. Enter Reference Height (h_ref).\n"
            "2. Select 2 Subjects on CCTV.\n"
            "   (Order: Head -> Feet)\n"
            "3. Compute."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; font-size: 12px;")
        side_vbox.addWidget(desc)
        
        grp = QGroupBox("Settings")
        g_layout = QVBoxLayout(grp)
        
        g_layout.addWidget(QLabel("Ref Height (m):"))
        self.spin_href = QDoubleSpinBox()
        self.spin_href.setRange(0.1, 10.0)
        self.spin_href.setSingleStep(0.1)
        self.spin_href.setValue(1.6)
        g_layout.addWidget(self.spin_href)
        
        self.btn_reset = QPushButton("Reset Points")
        self.btn_reset.clicked.connect(self._on_reset)
        g_layout.addWidget(self.btn_reset)
        
        side_vbox.addWidget(grp)
        
        self.btn_compute = QPushButton("Compute Camera Pos")
        self.btn_compute.setFixedHeight(40)
        self.btn_compute.setStyleSheet("background-color: #2a84ff; color: white; font-weight: bold;")
        self.btn_compute.clicked.connect(self._on_compute)
        side_vbox.addWidget(self.btn_compute)
        
        self.lbl_status = QLabel("Status: Select Subject 1 Head")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #ffd700; font-weight: bold;")
        side_vbox.addWidget(self.lbl_status)
        
        side_vbox.addStretch()
        
        self.btn_proceed = QPushButton("Proceed")
        self.btn_proceed.clicked.connect(self._on_proceed)
        side_vbox.addWidget(self.btn_proceed)
        
        main_layout.addWidget(sidebar)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_cont = QWidget()
        l_vbox = QVBoxLayout(left_cont)
        l_vbox.setContentsMargins(0,0,0,0)
        l_vbox.addWidget(QLabel("Undistorted CCTV (Right-Click: Head -> Feet)"))
        self.view_cctv = RightClickImageViewer()
        self.view_cctv.clicked.connect(self._on_cctv_click)
        l_vbox.addWidget(self.view_cctv)
        
        right_cont = QWidget()
        r_vbox = QVBoxLayout(right_cont)
        r_vbox.setContentsMargins(0,0,0,0)
        r_vbox.addWidget(QLabel("Satellite (Solution Visualization)"))
        self.view_sat = ImageViewer()
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
        if not host or not getattr(host, 'inspect_obj', None): return
        
        obj = host.inspect_obj
        loc_code = obj.get('meta', {}).get('location_code')

        # FIX: Check loaded state
        if self._loaded_loc == loc_code:
            return 
        self._loaded_loc = loc_code
        
        hom = obj.get('homography', {})
        H_list = hom.get('H')
        if H_list:
            self._H = np.array(H_list, dtype=np.float64)
        else:
            self.lbl_status.setText("Error: Homography Matrix missing.")
            self.btn_compute.setEnabled(False)
            
        self._load_images(obj)
        self._reset_state()

    def _load_images(self, obj):
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        if not loc_code: return

        # Undistort CCTV
        cctv_path = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
        if os.path.isfile(cctv_path):
            src = cv2.imread(cctv_path)
            if src is not None:
                und = obj.get('undistort', {})
                K = np.array(und.get('K'), dtype=np.float64)
                D = np.array(und.get('D', [0]*5), dtype=np.float64)
                h, w = src.shape[:2]
                # FIX: Use K.copy()
                # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
                newcameramtx = K.copy()
                
                undist = remap_with_supersample(src, K, D, newcameramtx)
                
                qimg = self._cv_to_qimage(undist)
                self.view_cctv.load_pixmap(QPixmap.fromImage(qimg))
                self.view_cctv.fitToView()

        # Satellite
        sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
        if os.path.isfile(sat_path):
            sat = cv2.imread(sat_path)
            if sat is not None:
                qimg = self._cv_to_qimage(sat)
                self.view_sat.load_pixmap(QPixmap.fromImage(qimg))
                self.view_sat.fitToView()

    def _reset_state(self):
        self.ref_pairs = []
        self._current_head = None
        self._clear_overlays()
        self.lbl_status.setText("Select Subject 1 Head")

    def _on_reset(self):
        self._reset_state()
        self.view_sat.scene().clear() 
        host = getattr(self, 'host_tab', None) or self.parent()
        if host: self._load_images(host.inspect_obj)

    def _on_cctv_click(self, x, y):
        if len(self.ref_pairs) >= 2:
            return 

        if self._current_head is None:
            self._current_head = (x, y)
            self._draw_point(self.view_cctv, x, y, color=Qt.red, size=8)
            curr_idx = len(self.ref_pairs) + 1
            self.lbl_status.setText(f"Select Subject {curr_idx} Feet")
        else:
            feet = (x, y)
            pair = (self._current_head, feet)
            self.ref_pairs.append(pair)
            
            self._draw_pair_overlay(pair, len(self.ref_pairs))
            
            self._current_head = None
            
            if len(self.ref_pairs) < 2:
                self.lbl_status.setText(f"Select Subject {len(self.ref_pairs)+1} Head")
            else:
                self.lbl_status.setText("Selection Complete. Click Compute.")

    def _draw_point(self, viewer, x, y, color=Qt.red, size=6):
        scene = viewer.scene()
        if not scene: return
        rad = size/2
        el = scene.addEllipse(x-rad, y-rad, size, size)
        el.setPen(QPen(color, 2))
        el.setBrush(QBrush(color))

    def _draw_pair_overlay(self, pair, idx):
        head, feet = pair
        scene = self.view_cctv.scene()
        
        self._draw_point(self.view_cctv, feet[0], feet[1], color=Qt.cyan, size=8)
        line = scene.addLine(head[0], head[1], feet[0], feet[1], QPen(Qt.green, 2))
        
        txt = scene.addSimpleText(f"S{idx}")
        txt.setBrush(QBrush(Qt.yellow))
        txt.setFont(QFont("Arial", 12, QFont.Bold))
        txt.setPos(head[0], head[1] - 20)

    def _clear_overlays(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host: self._load_images(host.inspect_obj)

    def _on_compute(self):
        if len(self.ref_pairs) < 2:
            QMessageBox.warning(self, "Incomplete", "Please select 2 subjects (4 clicks).")
            return
            
        subj1, subj2 = self.ref_pairs
        h_ref = self.spin_href.value()
        
        def to_sat(pt):
            src = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, self._H)
            return dst[0, 0]

        try:
            s1_head = to_sat(subj1[0])
            s1_feet = to_sat(subj1[1])
            s2_head = to_sat(subj2[0])
            s2_feet = to_sat(subj2[1])
            
            cam_sat_xy = self._find_intersection(s1_head, s1_feet, s2_head, s2_feet)
            
            if cam_sat_xy is None:
                QMessageBox.critical(self, "Error", "Lines are parallel. Subjects might be collinear with camera.\nPick subjects further apart laterally.")
                return

            def dist(p1, p2): return np.linalg.norm(np.array(p1) - np.array(p2))

            d_true_1 = dist(cam_sat_xy, s1_feet)
            d_app_1  = dist(cam_sat_xy, s1_head)
            d_true_2 = dist(cam_sat_xy, s2_feet)
            d_app_2  = dist(cam_sat_xy, s2_head)
            
            if d_app_1 == 0 or d_app_2 == 0:
                 QMessageBox.critical(self, "Math Error", "Apparent distance is zero (Head on Camera?).")
                 return
                 
            ratio_1 = d_true_1 / d_app_1
            z_cam_1 = h_ref / (1 - ratio_1)
            ratio_2 = d_true_2 / d_app_2
            z_cam_2 = h_ref / (1 - ratio_2)
            
            z_final = (z_cam_1 + z_cam_2) / 2.0
            
            self.lbl_status.setText(
                f"Computed!\n"
                f"Cam XY: {np.round(cam_sat_xy, 1)}\n"
                f"Cam Z: {z_final:.2f}m\n"
            )
            
            self._visualize_solution(cam_sat_xy, [s1_head, s1_feet], [s2_head, s2_feet])
            
            host = getattr(self, 'host_tab', None) or self.parent()
            if host and getattr(host, 'inspect_obj', None):
                obj = host.inspect_obj
                par = obj.setdefault('parallax', {})
                par['x_cam_coords_sat'] = float(cam_sat_xy[0])
                par['y_cam_coords_sat'] = float(cam_sat_xy[1])
                par['z_cam_meters'] = float(z_final)
                host.inspect_obj = obj
            
        except Exception as e:
            QMessageBox.critical(self, "Computation Error", str(e))

    def _find_intersection(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denom == 0: return None
        
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py])

    def _visualize_solution(self, cam_xy, s1_pts, s2_pts):
        scene = self.view_sat.scene()
        if not scene: return
        
        host = getattr(self, 'host_tab', None) or self.parent()
        if host: 
            proj_root = self.project_root or os.getcwd() 
            loc_code = host.inspect_obj['meta']['location_code']
            sat_path = os.path.join(proj_root, 'location', loc_code, f"sat_{loc_code}.png")
            
            if os.path.isfile(sat_path):
                img = cv2.imread(sat_path)
                self.view_sat.load_pixmap(QPixmap.fromImage(self._cv_to_qimage(img)))
                scene = self.view_sat.scene()

        cam_rad = 6
        el = scene.addEllipse(cam_xy[0]-cam_rad, cam_xy[1]-cam_rad, cam_rad*2, cam_rad*2)
        el.setPen(QPen(Qt.yellow, 2))
        el.setBrush(QBrush(Qt.red))
        
        scene.addLine(cam_xy[0], cam_xy[1], s1_pts[0][0], s1_pts[0][1], QPen(Qt.yellow, 1, Qt.DashLine))
        scene.addLine(cam_xy[0], cam_xy[1], s2_pts[0][0], s2_pts[0][1], QPen(Qt.yellow, 1, Qt.DashLine))
        
        scene.addLine(s1_pts[0][0], s1_pts[0][1], s1_pts[1][0], s1_pts[1][1], QPen(Qt.green, 2))
        scene.addLine(s2_pts[0][0], s2_pts[0][1], s2_pts[1][0], s2_pts[1][1], QPen(Qt.green, 2))
        
        rect = scene.itemsBoundingRect()
        self.view_sat.fitInView(rect, Qt.KeepAspectRatio)

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and hasattr(host, 'current_step_index'):
            try:
                host.current_step_index = 8
                host._show_stage(8)
                host._update_progress_to_index(8)
            except: pass

    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None: return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()