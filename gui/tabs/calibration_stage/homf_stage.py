import os
import json
import numpy as np
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QPolygonF
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGraphicsPolygonItem, QGraphicsRectItem, QMessageBox, 
    QGraphicsPixmapItem, QGroupBox
)

from .undistort_stage import ImageViewer, remap_with_supersample

# --- FIX: Custom Viewer for Composite Scenes ---
class CanvasViewer(ImageViewer):
    """
    Subclass of ImageViewer that allows zooming/panning on a composite scene
    (multiple items) without requiring a single '_pixmap_item' to be set.
    """
    def wheelEvent(self, event):
        # Override: Zoom regardless of specific item existence
        angle = event.angleDelta().y()
        factor = 1.25 if angle > 0 else 0.8
        self.scale(factor, factor)

    def fitToView(self):
        # Override: Fit the entire scene rectangle
        rect = self.sceneRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.KeepAspectRatio)


class HomFStage(QWidget):
    """
    Homography FOV Stage:
    1. Creates a 'Triple Sat' canvas (3x3 grid relative to satellite image).
    2. Warps the Satellite image to the center.
    3. Warps the CCTV image using the composite homography.
    4. Computes the FOV polygon (clipping to the canvas).
    5. Saves the polygon to JSON in Satellite-relative coordinates.
    """

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root

        self._fov_polygon_sat = None 
        
        # --- UI Layout ---
        main_layout = QHBoxLayout(self)
        
        # 1. Sidebar Controls
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(15)

        # Header
        lbl_title = QLabel("Homography FOV")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #fff;")
        side_vbox.addWidget(lbl_title)
        
        lbl_desc = QLabel(
            "Visualizes the projection on a large canvas (Triple Sat Rule).\n"
            "Calculates the CCTV Field of View boundary."
        )
        lbl_desc.setWordWrap(True)
        lbl_desc.setStyleSheet("color: #aaa;")
        side_vbox.addWidget(lbl_desc)

        # Slider Group
        slide_grp = QGroupBox("Visualization")
        slide_grp.setStyleSheet("QGroupBox { font-weight: bold; color: #ddd; }")
        s_layout = QVBoxLayout(slide_grp)
        
        self.lbl_opacity = QLabel("CCTV Opacity: 60%")
        s_layout.addWidget(self.lbl_opacity)
        
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(60)
        self.slider_alpha.valueChanged.connect(self._on_opacity_changed)
        s_layout.addWidget(self.slider_alpha)
        
        side_vbox.addWidget(slide_grp)

        # Compute Button
        self.btn_compute = QPushButton("Compute FOV & Warp")
        self.btn_compute.setFixedHeight(40)
        self.btn_compute.setStyleSheet("background-color: #2a84ff; color: white; font-weight: bold;")
        self.btn_compute.clicked.connect(self._on_compute)
        side_vbox.addWidget(self.btn_compute)

        # Status
        self.lbl_status = QLabel("Status: Ready to compute")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #ccc; font-style: italic;")
        side_vbox.addWidget(self.lbl_status)

        side_vbox.addStretch()
        
        # Proceed
        self.btn_proceed = QPushButton("Proceed")
        self.btn_proceed.clicked.connect(self._on_proceed)
        side_vbox.addWidget(self.btn_proceed)

        main_layout.addWidget(sidebar)

        # 2. Main Viewer (Using fixed CanvasViewer)
        self.viewer = CanvasViewer()
        main_layout.addWidget(self.viewer, 1)

        # Graphics Items References
        self.item_sat = None
        self.item_cctv = None
        self.item_fov = None
        self.item_boundary = None


    def showEvent(self, event):
        super().showEvent(event)
        # Check if we have H
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and getattr(host, 'inspect_obj', None):
            hom = host.inspect_obj.get('homography', {})
            H = hom.get('H')
            if not H:
                self.lbl_status.setText("⚠️ Warning: Homography matrix (H) missing.\nPlease complete 'Homography Anchors' first.")
                self.btn_compute.setEnabled(False)
            else:
                self.lbl_status.setText("Ready. Click Compute to generate view.")
                self.btn_compute.setEnabled(True)

    def _on_opacity_changed(self, val):
        self.lbl_opacity.setText(f"CCTV Opacity: {val}%")
        if self.item_cctv:
            self.item_cctv.setOpacity(val / 100.0)

    def _on_compute(self):
        if not HAS_CV2:
            self.lbl_status.setText("Error: OpenCV not installed.")
            return

        self.lbl_status.setText("Computing... Please wait.")
        self.viewer.scene().clear()
        
        # 1. Gather Data
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host: return
        obj = host.inspect_obj
        
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')

        # Load Images
        cctv_path = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
        sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')

        if not os.path.exists(cctv_path) or not os.path.exists(sat_path):
            self.lbl_status.setText("Error: Input images missing.")
            return

        img_cctv = cv2.imread(cctv_path)
        img_sat = cv2.imread(sat_path)
        
        # Undistort CCTV
        und = obj.get('undistort', {})
        K = np.array(und.get('K'), dtype=np.float64)
        D = np.array(und.get('D', [0]*5), dtype=np.float64)
        
        # FIX: Use K.copy() instead of getOptimalNewCameraMatrix to match reference code logic
        # h_u, w_u = img_cctv.shape[:2]
        # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w_u, h_u), 1, (w_u, h_u))
        newcameramtx = K.copy()
        
        img_cctv_undist = remap_with_supersample(img_cctv, K, D, newcameramtx)

        # Get H
        H = np.array(obj['homography']['H'], dtype=np.float32)

        # --- 2. Define Canvas (Triple Sat Rule) ---
        h_s, w_s = img_sat.shape[:2]
        padding_x = w_s
        padding_y = h_s
        canvas_w = w_s + (2 * padding_x)
        canvas_h = h_s + (2 * padding_y)

        # Translation Matrix T
        T = np.array([
            [1, 0, padding_x],
            [0, 1, padding_y],
            [0, 0, 1]
        ], dtype=np.float32)

        # Final Composite Homography
        H_final = T @ H

        # --- 3. Warp Images ---
        
        # Warp SAT (Simple translation to center)
        q_sat = self._cv_to_qimage(img_sat)
        pix_sat = QPixmap.fromImage(q_sat)
        self.item_sat = QGraphicsPixmapItem(pix_sat)
        self.item_sat.setPos(padding_x, padding_y) 
        self.viewer.scene().addItem(self.item_sat)

        # Draw Sat Boundary (Blue Dashed)
        rect_sat = QGraphicsRectItem(padding_x, padding_y, w_s, h_s)
        pen_b = QPen(Qt.blue)
        pen_b.setStyle(Qt.DashLine)
        pen_b.setWidth(2)
        rect_sat.setPen(pen_b)
        self.viewer.scene().addItem(rect_sat)

        # Warp CCTV
        img_cctv_bgra = cv2.cvtColor(img_cctv_undist, cv2.COLOR_BGR2BGRA)
        img_cctv_warped = cv2.warpPerspective(img_cctv_bgra, H_final, (canvas_w, canvas_h))

        q_cctv = self._cv_to_qimage_rgba(img_cctv_warped)
        pix_cctv = QPixmap.fromImage(q_cctv)
        self.item_cctv = QGraphicsPixmapItem(pix_cctv)
        self.item_cctv.setOpacity(self.slider_alpha.value() / 100.0)
        self.viewer.scene().addItem(self.item_cctv)

        # --- 4. Calculate FOV Contour ---
        h_u, w_u = img_cctv.shape[:2]
        mask_src = 255 * np.ones((h_u, w_u), dtype=np.uint8)
        mask_warped = cv2.warpPerspective(mask_src, H_final, (canvas_w, canvas_h))
        
        contours, _ = cv2.findContours(mask_warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self._fov_polygon_sat = []
        
        if contours:
            fov_contour = max(contours, key=cv2.contourArea)
            # Simplify
            epsilon = 0.001 * cv2.arcLength(fov_contour, True)
            fov_contour = cv2.approxPolyDP(fov_contour, epsilon, True)
            
            # Draw on Scene (Green Line)
            poly_points = []
            for pt in fov_contour[:, 0, :]:
                poly_points.append(QPointF(float(pt[0]), float(pt[1])))
                
                # Convert to Sat Coords for JSON
                x_sat = float(pt[0]) - padding_x
                y_sat = float(pt[1]) - padding_y
                self._fov_polygon_sat.append([x_sat, y_sat])

            if poly_points:
                poly_points.append(poly_points[0])

            poly_item = QGraphicsPolygonItem(QPolygonF(poly_points))
            pen_g = QPen(Qt.green)
            pen_g.setWidth(3)
            poly_item.setPen(pen_g)
            poly_item.setBrush(QBrush(Qt.NoBrush)) # Corrected brush style
            
            self.viewer.scene().addItem(poly_item)
            
            # Save to JSON
            obj['homography']['fov_polygon'] = self._fov_polygon_sat
            host.inspect_obj = obj
            
            self.lbl_status.setText(f"Success. FOV Polygon ({len(self._fov_polygon_sat)} pts) saved.")
        else:
            self.lbl_status.setText("Warning: No FOV contour found (check H matrix).")

        # Fit view to the Canvas area
        self.viewer.setSceneRect(0, 0, canvas_w, canvas_h)
        self.viewer.fitToView()

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and hasattr(host, 'current_step_index'):
            try:
                # Proceed to Index 6 (Validation 2)
                host.current_step_index = 6
                host._show_stage(6)
                host._update_progress_to_index(6)
            except: pass

    # --- Helpers ---
    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None: return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()

    def _cv_to_qimage_rgba(self, cv_bgra):
        if cv_bgra is None: return None
        rgba = cv2.cvtColor(cv_bgra, cv2.COLOR_BGRA2RGBA)
        h, w, ch = rgba.shape
        bytes_per_line = ch * w
        return QImage(rgba.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGBA8888).copy()