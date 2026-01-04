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
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QFont
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QComboBox, QDoubleSpinBox, QGroupBox, QMessageBox
)

from .undistort_stage import ImageViewer

class DistStage(QWidget):
    """
    Distance Reference Stage.
    Calculates Pixel-to-Meter scale using two known anchors on the Satellite Map.
    """

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        
        # Data
        self.anchors = [] # List of dicts {name, coords_sat}
        
        # --- UI ---
        main_layout = QHBoxLayout(self)
        
        # 1. Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(15)
        
        lbl_title = QLabel("Distance Reference")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #fff;")
        side_vbox.addWidget(lbl_title)
        
        desc = QLabel(
            "Calculate map scale (px/meter).\n\n"
            "1. Select Start Anchor.\n"
            "2. Select End Anchor.\n"
            "3. Enter real-world distance."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; font-size: 12px;")
        side_vbox.addWidget(desc)
        
        # Selection Group
        grp_sel = QGroupBox("Reference Points")
        l_sel = QVBoxLayout(grp_sel)
        
        l_sel.addWidget(QLabel("Start Point:"))
        self.combo_start = QComboBox()
        self.combo_start.currentIndexChanged.connect(self._update_visualization)
        l_sel.addWidget(self.combo_start)
        
        l_sel.addWidget(QLabel("End Point:"))
        self.combo_end = QComboBox()
        self.combo_end.currentIndexChanged.connect(self._update_visualization)
        l_sel.addWidget(self.combo_end)
        
        side_vbox.addWidget(grp_sel)
        
        # Distance Input
        grp_dist = QGroupBox("Real World")
        l_dist = QVBoxLayout(grp_dist)
        
        l_dist.addWidget(QLabel("Distance (meters):"))
        self.spin_dist = QDoubleSpinBox()
        self.spin_dist.setRange(0.1, 1000.0)
        self.spin_dist.setValue(5.0) # Default 5m
        self.spin_dist.setSingleStep(0.5)
        l_dist.addWidget(self.spin_dist)
        
        side_vbox.addWidget(grp_dist)
        
        # Compute
        self.btn_compute = QPushButton("Compute Scale")
        self.btn_compute.setFixedHeight(40)
        self.btn_compute.setStyleSheet("background-color: #2a84ff; color: white; font-weight: bold;")
        self.btn_compute.clicked.connect(self._on_compute)
        side_vbox.addWidget(self.btn_compute)
        
        self.lbl_result = QLabel("Scale: Not Computed")
        self.lbl_result.setStyleSheet("color: #ffd700; font-weight: bold;")
        side_vbox.addWidget(self.lbl_result)
        
        side_vbox.addStretch()
        
        self.btn_proceed = QPushButton("Proceed")
        self.btn_proceed.clicked.connect(self._on_proceed)
        side_vbox.addWidget(self.btn_proceed)
        
        main_layout.addWidget(sidebar)
        
        # 2. Main Viewer (Satellite Map)
        self.viewer = ImageViewer()
        main_layout.addWidget(self.viewer, 1)

    def showEvent(self, event):
        super().showEvent(event)
        if not HAS_CV2: return
        
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None): return
        
        obj = host.inspect_obj
        
        # 1. Load Anchors from Homography
        hom = obj.get('homography', {})
        raw_list = hom.get('anchors_list', [])
        
        self.anchors = []
        self.combo_start.clear()
        self.combo_end.clear()
        
        for item in raw_list:
            # We only need anchors that have Satellite coordinates defined
            if item.get('coords_sat'):
                name = item.get('name', f"ID {item['id']}")
                self.anchors.append({
                    'name': name,
                    'pt': item['coords_sat'] # [x, y]
                })
                self.combo_start.addItem(name)
                self.combo_end.addItem(name)
                
        # Select different defaults if possible
        if len(self.anchors) >= 2:
            self.combo_end.setCurrentIndex(1)

        # 2. Load Satellite Image
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
        
        if os.path.isfile(sat_path):
            img = cv2.imread(sat_path)
            if img is not None:
                self.viewer.load_pixmap(QPixmap.fromImage(self._cv_to_qimage(img)))
                self.viewer.fitToView()
                
        self._update_visualization()

    def _update_visualization(self):
        # Clear dynamic overlays (keep pixmap)
        # We assume the pixmap is the bottom-most item.
        scene = self.viewer.scene()
        if not scene: return
        
        # Remove lines and text, keep pixmap
        from PyQt5.QtWidgets import QGraphicsPixmapItem
        for item in scene.items():
            if not isinstance(item, QGraphicsPixmapItem):
                scene.removeItem(item)
                
        # Draw all anchors as small dots
        for a in self.anchors:
            x, y = a['pt']
            self._draw_dot(x, y, Qt.gray, 4)
            
        # Draw Selected Connection
        idx1 = self.combo_start.currentIndex()
        idx2 = self.combo_end.currentIndex()
        
        if idx1 < 0 or idx2 < 0 or idx1 >= len(self.anchors) or idx2 >= len(self.anchors):
            return

        p1 = self.anchors[idx1]['pt']
        p2 = self.anchors[idx2]['pt']
        
        # Highlight Selected Points
        self._draw_dot(p1[0], p1[1], Qt.cyan, 8)
        self._draw_dot(p2[0], p2[1], Qt.cyan, 8)
        
        # Draw Line
        pen = QPen(Qt.yellow)
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        scene.addLine(p1[0], p1[1], p2[0], p2[1], pen)

    def _draw_dot(self, x, y, color, size):
        scene = self.viewer.scene()
        rad = size/2
        el = scene.addEllipse(x-rad, y-rad, size, size)
        el.setPen(QPen(color))
        el.setBrush(QBrush(color))

    def _on_compute(self):
        idx1 = self.combo_start.currentIndex()
        idx2 = self.combo_end.currentIndex()
        
        if idx1 == idx2:
            QMessageBox.warning(self, "Error", "Please select two different anchors.")
            return
            
        p1 = np.array(self.anchors[idx1]['pt'])
        p2 = np.array(self.anchors[idx2]['pt'])
        
        # 1. Pixel Distance
        px_dist = np.linalg.norm(p1 - p2)
        
        # 2. Real Distance
        real_m = self.spin_dist.value()
        
        if real_m <= 0: return
        
        # 3. Scale
        px_per_m = px_dist / real_m
        
        self.lbl_result.setText(f"Scale: {px_per_m:.2f} px/m")
        
        # 4. Save to JSON
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and getattr(host, 'inspect_obj', None):
            obj = host.inspect_obj
            par = obj.setdefault('parallax', {})
            
            # Update scale section
            scale_data = {
                "measured_px": float(px_dist),
                "real_m": float(real_m),
                "reference_anchors": [
                    self.anchors[idx1]['name'],
                    self.anchors[idx2]['name']
                ]
            }
            par['scale'] = scale_data
            par['px_per_meter'] = float(px_per_m)
            
            host.inspect_obj = obj
            QMessageBox.information(self, "Success", f"Scale calculated and saved:\n{px_per_m:.4f} px/m")

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and hasattr(host, 'current_step_index'):
            try:
                # Proceed to Index 9 (Validation 3 / Val3)
                host.current_step_index = 9
                host._show_stage(9)
                host._update_progress_to_index(9)
            except: pass

    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None: return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()