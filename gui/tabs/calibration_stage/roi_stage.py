import os
import numpy as np
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QCheckBox, QRadioButton, QButtonGroup, QGroupBox, QGraphicsView, 
    QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem
)

class ROIDrawViewer(QGraphicsView):
    """
    Viewer that allows drawing a rectangle to test ROI logic.
    No Pan/Zoom allowed.
    """
    boxChanged = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints() | Qt.SmoothTransformation)
        self.setDragMode(QGraphicsView.NoDrag) # Disable panning
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self._pixmap_item = None
        self._overlay_item = None
        self._rect_item = None
        
        self._start_pos = None
        self._is_drawing = False

    def load_base_image(self, pixmap: QPixmap):
        self.scene().clear()
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene().addItem(self._pixmap_item)
        self.setSceneRect(self._pixmap_item.boundingRect())
        self._overlay_item = None
        self._rect_item = None

    def set_overlay(self, pixmap: QPixmap):
        if self._overlay_item:
            self.scene().removeItem(self._overlay_item)
            self._overlay_item = None
        
        if pixmap and not pixmap.isNull():
            self._overlay_item = self.scene().addPixmap(pixmap)
            self._overlay_item.setZValue(1)

    def clear_overlay(self):
        if self._overlay_item:
            self.scene().removeItem(self._overlay_item)
            self._overlay_item = None

    def set_box_color(self, is_valid: bool):
        if self._rect_item:
            color = Qt.green if is_valid else Qt.red
            pen = QPen(color)
            pen.setWidth(3)
            self._rect_item.setPen(pen)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start_pos = self.mapToScene(event.pos())
            self._is_drawing = True
            
            # Remove old box
            if self._rect_item:
                self.scene().removeItem(self._rect_item)
                self._rect_item = None
            
            # Create new box
            self._rect_item = QGraphicsRectItem()
            self._rect_item.setZValue(2) # On top of overlay
            
            # FIX: Wrap Qt.NoBrush in QBrush()
            self._rect_item.setBrush(QBrush(Qt.NoBrush))
            
            self._rect_item.setPen(QPen(Qt.white, 2, Qt.DashLine))
            self.scene().addItem(self._rect_item)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_drawing and self._start_pos:
            curr_pos = self.mapToScene(event.pos())
            rect = QRectF(self._start_pos, curr_pos).normalized()
            
            # Clamp to image bounds
            if self._pixmap_item:
                brect = self._pixmap_item.boundingRect()
                rect = rect.intersected(brect)
            
            self._rect_item.setRect(rect)
            self.boxChanged.emit(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_drawing = False
            # Final check emit
            if self._rect_item:
                self.boxChanged.emit(self._rect_item.rect())
        super().mouseReleaseEvent(event)


class ROIStage(QWidget):
    """
    ROI Stage.
    Configure logic for Region of Interest ("partial" vs "in").
    Visual feedback by drawing boxes on the RAW (Distorted) CCTV image.
    """

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        
        self._mask_cv = None # Grayscale numpy array (0=Blocked, 255=ROI)
        self._roi_pixmap = None
        
        # --- UI ---
        main_layout = QHBoxLayout(self)
        
        # 1. Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(300)
        sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(15)
        
        lbl_title = QLabel("ROI Configuration")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #fff;")
        side_vbox.addWidget(lbl_title)
        
        desc = QLabel(
            "Define how objects interacting with the ROI are handled.\n\n"
            "Draw a box on the image to test."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; font-size: 12px;")
        side_vbox.addWidget(desc)
        
        # Method Selection
        grp_method = QGroupBox("Evaluation Method")
        v_meth = QVBoxLayout(grp_method)
        
        self.rb_partial = QRadioButton("Partial")
        self.rb_partial.setToolTip("Object valid if ANY corner is inside ROI")
        self.rb_in = QRadioButton("In (Strict)")
        self.rb_in.setToolTip("Object valid only if ALL 4 corners are inside ROI")
        
        self.bg_method = QButtonGroup()
        self.bg_method.addButton(self.rb_partial)
        self.bg_method.addButton(self.rb_in)
        self.bg_method.buttonClicked.connect(self._on_method_changed)
        
        v_meth.addWidget(self.rb_partial)
        v_meth.addWidget(self.rb_in)
        side_vbox.addWidget(grp_method)
        
        # Visualization
        self.cb_show_mask = QCheckBox("Show ROI Mask")
        self.cb_show_mask.setChecked(True)
        self.cb_show_mask.toggled.connect(self._on_toggle_mask)
        side_vbox.addWidget(self.cb_show_mask)
        
        self.lbl_status = QLabel("Result: Draw a box")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 14px; color: #ccc;")
        side_vbox.addWidget(self.lbl_status)
        
        side_vbox.addStretch()
        
        self.btn_proceed = QPushButton("Proceed")
        self.btn_proceed.clicked.connect(self._on_proceed)
        side_vbox.addWidget(self.btn_proceed)
        
        main_layout.addWidget(sidebar)
        
        # 2. Main Viewer
        self.viewer = ROIDrawViewer()
        self.viewer.boxChanged.connect(self._validate_box)
        main_layout.addWidget(self.viewer, 1)

    def showEvent(self, event):
        super().showEvent(event)
        if not HAS_CV2: return
        
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None): return
        obj = host.inspect_obj
        
        # 1. Load Config
        method = obj.get('roi_method', 'partial')
        if method == 'in':
            self.rb_in.setChecked(True)
        else:
            self.rb_partial.setChecked(True)
            
        # 2. Load Images
        self._load_data(obj)

    def _load_data(self, obj):
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        if not loc_code: return
        
        # Load CCTV - RAW (Distorted) as requested
        cctv_path = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
        if os.path.isfile(cctv_path):
            img = cv2.imread(cctv_path)
            if img is not None:
                # Direct load, NO UNDISTORTION
                self.viewer.load_base_image(QPixmap.fromImage(self._cv_to_qimage(img)))
                self.viewer.fitInView(self.viewer.sceneRect(), Qt.KeepAspectRatio)

        # Load ROI
        roi_path = os.path.join(proj_root, 'location', loc_code, f'roi_{loc_code}.png')
        if os.path.isfile(roi_path):
            # Load as grayscale mask for logic (0=Black, >0=ROI)
            self._mask_cv = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
            
            # Fix dimensions: Ensure strictly 2D (H, W)
            if self._mask_cv is not None and self._mask_cv.ndim == 3:
                self._mask_cv = self._mask_cv[:, :, 0]
            
            # Generate Overlay
            roi_col = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)
            if roi_col is not None:
                h, w = roi_col.shape[:2]
                
                # Logic: We want to visualize the "Excluded" area (Black pixels).
                # Mask: 0 is Excluded.
                overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
                
                # Check dimensions again to be safe
                if self._mask_cv.shape[:2] != (h, w):
                    self._mask_cv = cv2.resize(self._mask_cv, (w, h))

                is_black = (self._mask_cv < 10)

                # Set RGBA for masked pixels in one assignment.
                # Format_ARGB32 expects memory order B,G,R,A on little-endian.
                overlay_arr[is_black] = (0, 0, 255, 100)
                
                self._roi_pixmap = QPixmap.fromImage(
                    QImage(overlay_arr.data, w, h, QImage.Format_ARGB32)
                )
                
                self._on_toggle_mask(self.cb_show_mask.isChecked())
        else:
            self.lbl_status.setText("Warning: ROI file not found.")
            self._mask_cv = None

    def _on_toggle_mask(self, checked):
        if checked and self._roi_pixmap:
            self.viewer.set_overlay(self._roi_pixmap)
        else:
            self.viewer.clear_overlay()

    def _on_method_changed(self):
        # Trigger validation of current box if it exists
        if self.viewer._rect_item:
            self._validate_box(self.viewer._rect_item.rect())
            
        # Update JSON
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and getattr(host, 'inspect_obj', None):
            method = "in" if self.rb_in.isChecked() else "partial"
            host.inspect_obj['roi_method'] = method

    def _validate_box(self, rect: QRectF):
        if self._mask_cv is None:
            return

        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        
        # Get Corners coordinates (integer)
        h_img, w_img = self._mask_cv.shape
        
        corners = [
            (int(x), int(y)),           # Top-Left
            (int(x + w), int(y)),       # Top-Right
            (int(x), int(y + h)),       # Bottom-Left
            (int(x + w), int(y + h))    # Bottom-Right
        ]
        
        valid_corners = 0
        
        for cx, cy in corners:
            # Clamp
            cx = max(0, min(w_img - 1, cx))
            cy = max(0, min(h_img - 1, cy))
            
            if self._mask_cv[cy, cx] > 0:
                valid_corners += 1
                
        is_valid = False
        method = "in" if self.rb_in.isChecked() else "partial"
        
        if method == "partial":
            is_valid = (valid_corners >= 1)
        elif method == "in":
            is_valid = (valid_corners == 4)
            
        self.viewer.set_box_color(is_valid)
        
        color_hex = "#00FF00" if is_valid else "#FF0000"
        txt = "ACCEPTED" if is_valid else "REJECTED"
        self.lbl_status.setText(f"Result: <span style='color:{color_hex}'>{txt}</span> ({method})")

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and hasattr(host, 'current_step_index'):
            try:
                # Proceed to Index 12 (Final Validation)
                host.current_step_index = 12
                host._show_stage(12)
                host._update_progress_to_index(12)
            except: pass

    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None: return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()