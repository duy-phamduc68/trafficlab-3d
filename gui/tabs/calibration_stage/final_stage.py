import os
import math
import re
import numpy as np
import xml.etree.ElementTree as ET
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QPointF
from PyQt5.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QTransform, QFont, QPolygonF
from PyQt5.QtSvg import QGraphicsSvgItem
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QCheckBox, QGroupBox, QGraphicsView, QGraphicsScene, 
    QGraphicsRectItem, QDoubleSpinBox, QComboBox, QSlider, 
    QScrollArea, QGraphicsPolygonItem, QSplitter
)

from .undistort_stage import ImageViewer

# ==========================================================
# ROBUST SVG PARSER (Matching g_projection.py)
# ==========================================================
class SVGParser:
    def __init__(self, svg_path, affine_matrix=None):
        self.svg_path = svg_path
        self.orientation_segments = []
        self.M_align = np.identity(3)
        if affine_matrix is not None:
            self.M_align[:2, :] = np.array(affine_matrix)
            
        self.valid = False
        if os.path.exists(svg_path):
            try:
                self.tree = ET.parse(svg_path)
                self.root = self.tree.getroot()
                self.orientation_segments = self._extract_segments()
                self.valid = True
                # print(f"[SVG] Loaded {len(self.orientation_segments)} segments.")
            except Exception as e:
                print(f"[SVG ERR] {e}")
        else:
            print(f"[SVG ERR] File not found: {svg_path}")

    def _parse_transform(self, txt):
        M = np.identity(3)
        if not txt: return M
        ops = re.findall(r'(\w+)\s*\(([^)]+)\)', txt)
        for name, args in ops:
            vals = list(map(float, filter(None, re.split(r'[ ,]+', args.strip()))))
            T = np.identity(3)
            if name == 'translate':
                T[0,2], T[1,2] = vals[0], vals[1] if len(vals) > 1 else 0
            elif name == 'rotate':
                rad = math.radians(vals[0])
                c, s = math.cos(rad), math.sin(rad)
                if len(vals) == 3:
                    cx, cy = vals[1], vals[2]
                    T1=np.eye(3); T1[0,2]=cx; T1[1,2]=cy
                    R=np.eye(3); R[:2,:2]=[[c,-s],[s,c]]
                    T2=np.eye(3); T2[0,2]=-cx; T2[1,2]=-cy
                    T = T1 @ R @ T2
                else:
                    T[:2,:2] = [[c,-s],[s,c]]
            elif name == 'matrix':
                T = np.array([[vals[0], vals[2], vals[4]],
                              [vals[1], vals[3], vals[5]],
                              [0, 0, 1]])
            M = M @ T
        return M

    def _extract_segments(self):
        segs = []
        # We look for groups with these IDs
        target_ids = ['Guidelines', 'Physical']
        
        # Helper to strip namespace: "{http://www.w3.org/2000/svg}g" -> "g"
        def get_tag(el):
            return el.tag.split('}')[-1]

        # 1. Find the target groups (Guidelines/Physical)
        target_nodes = []
        # Traverse entire tree to find groups with matching IDs
        for el in self.root.iter():
            if get_tag(el) == 'g' and el.get('id') in target_ids:
                target_nodes.append(el)

        if not target_nodes:
            print(f"[SVG] Warning: No groups found with IDs {target_ids}")

        # 2. Process everything inside those groups
        for g in target_nodes:
            self._process_node(g, np.identity(3), segs)
            
        return segs

    def _process_node(self, element, parent_mat, seg_list):
        # Apply local transform if it exists
        local_mat = self._parse_transform(element.get('transform'))
        curr_mat = parent_mat @ local_mat
        
        tag = element.tag.split('}')[-1]
        pts = []
        
        if tag == 'line':
            pts = np.array([[float(element.get('x1',0)), float(element.get('y1',0))],
                            [float(element.get('x2',0)), float(element.get('y2',0))]])
        elif tag == 'polygon' or tag == 'polyline':
            raw = re.split(r'[ ,]+', element.get('points','').strip())
            raw = [x for x in raw if x]
            if raw: pts = np.array(raw, dtype=float).reshape(-1,2)
        
        # Transform and Store
        if len(pts) > 0:
            homo = np.hstack([pts, np.ones((len(pts), 1))])
            # Apply SVG transforms THEN the Alignment Matrix
            t_pts = (self.M_align @ (curr_mat @ homo.T)).T[:, :2]
            
            for i in range(len(t_pts)-1):
                seg_list.append((t_pts[i], t_pts[i+1]))
            if tag == 'polygon':
                seg_list.append((t_pts[-1], t_pts[0]))

        # Recursively process children (important for nested groups)
        for child in element:
            self._process_node(child, curr_mat, seg_list)

    def get_nearest_heading_info(self, pt):
        if not self.valid or not self.orientation_segments: return None, None, None
        min_d = float('inf')
        best_ang = None
        best_seg = None
        pt = np.array(pt)
        for sp1, sp2 in self.orientation_segments:
            ab = sp2 - sp1
            ab_sq = np.dot(ab, ab)
            if ab_sq < 1e-6: continue
            ap = pt - sp1
            t = np.dot(ap, ab) / ab_sq
            closest = sp1 + np.clip(t, 0, 1) * ab
            d = np.linalg.norm(pt - closest)
            if d < min_d:
                min_d = d
                best_ang = math.degrees(math.atan2(ab[1], ab[0]))
                best_seg = (sp1, sp2)
        if best_ang is not None:
            return (best_ang + 360) % 360, best_seg[0], best_seg[1]
        return None, None, None

class BoxDrawViewer(ImageViewer):
    boxDrawn = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rect_item = None
        self._start_pos = None
        self._is_drawing = False
        self._overlay_item = None
        self._pen = QPen(Qt.yellow, 2, Qt.SolidLine)
        self._brush = QBrush(QColor(255, 255, 0, 50))

    def load_pixmap(self, pixmap: QPixmap):
        super().load_pixmap(pixmap) 
        self._rect_item = None
        self._overlay_item = None

    def set_overlay(self, pixmap: QPixmap):
        if self._overlay_item:
            try:
                if self._overlay_item.scene() == self.scene(): self.scene().removeItem(self._overlay_item)
            except RuntimeError: pass
            self._overlay_item = None
        if pixmap and not pixmap.isNull():
            self._overlay_item = self.scene().addPixmap(pixmap)
            self._overlay_item.setZValue(1)

    def clear_overlay(self):
        if self._overlay_item:
            try:
                if self._overlay_item.scene() == self.scene(): self.scene().removeItem(self._overlay_item)
            except RuntimeError: pass
            self._overlay_item = None

    def mousePressEvent(self, event):
        if self._pixmap_item and event.button() == Qt.RightButton:
            self._start_pos = self.mapToScene(event.pos())
            self._is_drawing = True
            if self._rect_item:
                try:
                    if self._rect_item.scene() == self.scene(): self.scene().removeItem(self._rect_item)
                except RuntimeError: pass
                self._rect_item = None
            self._rect_item = QGraphicsRectItem()
            self._rect_item.setPen(self._pen)
            self._rect_item.setBrush(self._brush)
            self._rect_item.setZValue(10)
            self.scene().addItem(self._rect_item)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_drawing and self._start_pos:
            curr_pos = self.mapToScene(event.pos())
            rect = QRectF(self._start_pos, curr_pos).normalized()
            brect = self._pixmap_item.boundingRect()
            rect = rect.intersected(brect)
            if self._rect_item: self._rect_item.setRect(rect)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._is_drawing:
            self._is_drawing = False
            if self._rect_item: self.boxDrawn.emit(self._rect_item.rect())
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class FinalStage(QWidget):
    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        
        self._K = None; self._D = None; self._new_K = None; self._H = None; self._H_inv = None
        self._z_cam = 10.0; self._cam_sat = np.zeros(2); self._px_per_m = 1.0
        
        # Robust SVG Parser Instance
        self._svg_parser = None
        self._svg_affine = None
        self._mask_cv = None 
        
        self._current_rect = None 
        self._ref_point_cctv = None 
        self._proj_point_sat = None 
        self._gc_point_cctv = None  
        self._heading_deg = 0.0
        self._show_3d_active = False
        
        self._svg_item = None; self._roi_overlay = None
        self._sat_markers = []; self._cctv_markers = []; self._wireframe_items = []
        self._floor_poly = None; self._highlight_line = None 
        self._debug_items = []
        
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        sidebar = QWidget(); sidebar.setFixedWidth(320); sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_layout = QVBoxLayout(sidebar); side_layout.setSpacing(10)
        
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet("border: none;")
        scroll_content = QWidget(); vbox = QVBoxLayout(scroll_content)
        scroll.setWidget(scroll_content)
        
        vbox.addWidget(QLabel("<h2>Final Validation</h2>"))
        
        grp1 = QGroupBox("1. Draw Box"); l1 = QVBoxLayout(grp1)
        l1.addWidget(QLabel("Right-click drag on CCTV."))
        self.btn_reset_box = QPushButton("Reset Box")
        self.btn_reset_box.setStyleSheet("border: 1px solid #555; padding: 4px; background-color: #444; color: #fff;")
        self.btn_reset_box.clicked.connect(self._on_reset_box)
        l1.addWidget(self.btn_reset_box)
        vbox.addWidget(grp1)
        
        grp2 = QGroupBox("2. Dimensions (m)"); l2 = QVBoxLayout(grp2); h2 = QHBoxLayout()
        self.spin_w = self._make_spin(1.8); self.spin_l = self._make_spin(3.5); self.spin_h = self._make_spin(1.55)
        self.spin_w.valueChanged.connect(self._refresh_visuals)
        self.spin_l.valueChanged.connect(self._refresh_visuals)
        self.spin_h.valueChanged.connect(self._refresh_visuals)
        h2.addWidget(QLabel("W:")); h2.addWidget(self.spin_w)
        h2.addWidget(QLabel("L:")); h2.addWidget(self.spin_l)
        h2.addWidget(QLabel("H:")); h2.addWidget(self.spin_h)
        l2.addLayout(h2); vbox.addWidget(grp2)
        
        grp3 = QGroupBox("3. Projection"); l3 = QVBoxLayout(grp3)
        l3.addWidget(QLabel("Ref Method:"))
        self.cb_ref = QComboBox(); self.cb_ref.addItems(["center_box", "center_bottom_side"])
        self.cb_ref.currentIndexChanged.connect(self._on_confirm_points)
        l3.addWidget(self.cb_ref)
        l3.addWidget(QLabel("Proj Method:"))
        self.cb_proj = QComboBox(); self.cb_proj.addItems(["down_h_2", "down_h", "match"])
        self.cb_proj.currentIndexChanged.connect(self._on_confirm_points)
        l3.addWidget(self.cb_proj)
        self.btn_confirm_pts = QPushButton("Calculate Points")
        self.btn_confirm_pts.setStyleSheet("background-color: #2a84ff; color: white; font-weight: bold; border: 1px solid #1a64db; padding: 4px;")
        self.btn_confirm_pts.clicked.connect(self._on_confirm_points)
        l3.addWidget(self.btn_confirm_pts)
        vbox.addWidget(grp3)
        
        grp4 = QGroupBox("4. Heading & Floor"); l4 = QVBoxLayout(grp4)
        self.chk_auto_head = QCheckBox("Auto Heading (SVG)"); self.chk_auto_head.setChecked(True)
        self.chk_auto_head.toggled.connect(self._toggle_heading_mode)
        l4.addWidget(self.chk_auto_head)
        h4 = QHBoxLayout(); h4.addWidget(QLabel("Angle:"))
        self.slider_head = QSlider(Qt.Horizontal); self.slider_head.setRange(0, 360)
        self.slider_head.setEnabled(False)
        self.slider_head.valueChanged.connect(self._on_heading_changed)
        h4.addWidget(self.slider_head)
        l4.addLayout(h4); vbox.addWidget(grp4)
        
        grp5 = QGroupBox("5. 3D Reconstruction"); l5 = QVBoxLayout(grp5)
        self.btn_show_3d = QPushButton("Toggle 3D Box"); self.btn_show_3d.setCheckable(True)
        self.btn_show_3d.setStyleSheet("QPushButton { background-color: #2ca02c; color: white; font-weight: bold; border: 1px solid #1c801c; padding: 4px; } QPushButton:checked { background-color: #1e701e; }")
        self.btn_show_3d.toggled.connect(self._on_toggle_3d)
        l5.addWidget(self.btn_show_3d)
        vbox.addWidget(grp5)
        
        grp_opt = QGroupBox("Options"); l_opt = QVBoxLayout(grp_opt)
        self.chk_roi = QCheckBox("Show ROI Mask"); self.chk_roi.toggled.connect(self._on_toggle_roi)
        l_opt.addWidget(self.chk_roi)
        h_svg = QHBoxLayout(); h_svg.addWidget(QLabel("SVG Alpha:"))
        self.slider_alpha = QSlider(Qt.Horizontal); self.slider_alpha.setRange(0, 100); self.slider_alpha.setValue(50)
        self.slider_alpha.valueChanged.connect(self._on_alpha_changed)
        h_svg.addWidget(self.slider_alpha)
        l_opt.addLayout(h_svg)
        vbox.addWidget(grp_opt)
        
        self.lbl_status = QLabel("Ready"); self.lbl_status.setWordWrap(True); self.lbl_status.setStyleSheet("color: #aaa; font-style: italic;")
        vbox.addWidget(self.lbl_status)
        
        self.btn_proceed = QPushButton("Proceed"); self.btn_proceed.setFixedHeight(40)
        self.btn_proceed.setStyleSheet("background-color: #d35400; color: white; font-weight: bold; font-size: 14px; border: 1px solid #a04000; border-radius: 4px;")
        self.btn_proceed.clicked.connect(self._on_proceed)
        vbox.addWidget(self.btn_proceed)
        
        side_layout.addWidget(scroll)
        main_layout.addWidget(sidebar)
        
        splitter = QSplitter(Qt.Horizontal)
        left_c = QWidget(); lv = QVBoxLayout(left_c); lv.setContentsMargins(0,0,0,0)
        lv.addWidget(QLabel("CCTV (Right-Click Drag)"))
        self.view_cctv = BoxDrawViewer(); self.view_cctv.boxDrawn.connect(self._on_box_drawn)
        lv.addWidget(self.view_cctv)
        right_c = QWidget(); rv = QVBoxLayout(right_c); rv.setContentsMargins(0,0,0,0)
        rv.addWidget(QLabel("Satellite / SVG"))
        self.view_sat = ImageViewer()
        rv.addWidget(self.view_sat)
        splitter.addWidget(left_c); splitter.addWidget(right_c)
        main_layout.addWidget(splitter, 1)

    def _make_spin(self, val):
        s = QDoubleSpinBox(); s.setRange(0.1, 50.0); s.setSingleStep(0.1); s.setValue(val)
        return s

    def showEvent(self, event):
        super().showEvent(event)
        if not HAS_CV2: return
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None): return
        
        self._clear_markers()
        self._sat_markers = []; self._cctv_markers = []; self._wireframe_items = []
        self._floor_poly = None; self._highlight_line = None; self._svg_item = None; self._roi_overlay = None
        
        try:
            self._load_params(host.inspect_obj)
            self._load_images(host.inspect_obj)
            
            # Initialize Robust SVG Parser
            self._init_svg_parser(host.inspect_obj)
            
            saved_ref = host.inspect_obj.get('ref_method')
            if saved_ref:
                idx = self.cb_ref.findText(saved_ref)
                if idx >= 0: self.cb_ref.setCurrentIndex(idx)
            saved_proj = host.inspect_obj.get('proj_method')
            if saved_proj:
                idx = self.cb_proj.findText(saved_proj)
                if idx >= 0: self.cb_proj.setCurrentIndex(idx)
        except Exception as e:
            self.lbl_status.setText(f"Error loading: {e}")

    def _load_params(self, obj):
        und = obj.get('undistort', {})
        K = und.get('K'); D = und.get('D', [0]*5)
        self._K = np.array(K, dtype=np.float64) if K else None
        self._D = np.array(D, dtype=np.float64)
        
        hom = obj.get('homography', {})
        H = hom.get('H')
        if H:
            self._H = np.array(H, dtype=np.float64)
            self._H_inv = np.linalg.inv(self._H)
            
        par = obj.get('parallax', {})
        self._z_cam = par.get('z_cam_meters', 10.0)
        self._cam_sat = np.array([par.get('x_cam_coords_sat',0), par.get('y_cam_coords_sat',0)])
        self._px_per_m = par.get('px_per_meter', 1.0)
        if self._px_per_m <= 0.001: self._px_per_m = 10.0
        
        layout = obj.get('layout_svg', {})
        A = layout.get('A') 
        if A: self._svg_affine = np.array(A)
        
        use_svg = obj.get('use_svg', False)
        self.chk_auto_head.setEnabled(use_svg)
        self.slider_alpha.setEnabled(use_svg)
        if not use_svg: self.chk_auto_head.setChecked(False)
        # Disable ROI option when SVG usage is disabled (no overlay/applicability)
        try:
            self.chk_roi.setEnabled(use_svg)
            if not use_svg:
                try:
                    self.chk_roi.blockSignals(True)
                    self.chk_roi.setChecked(False)
                finally:
                    try: self.chk_roi.blockSignals(False)
                    except: pass
                # Ensure any existing overlay is cleared
                try:
                    if hasattr(self, 'view_cctv'):
                        self.view_cctv.clear_overlay()
                except Exception:
                    pass
        except Exception:
            pass

    def _load_images(self, obj):
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        
        cctv_path = os.path.join(proj_root, 'location', loc_code, f'cctv_{loc_code}.png')
        if os.path.isfile(cctv_path):
            img = cv2.imread(cctv_path)
            if img is not None:
                if self._K is not None: self._new_K = self._K.copy()
                self.view_cctv.load_pixmap(QPixmap.fromImage(self._cv_to_qimage(img)))
                self.view_cctv.fitToView()
                
        sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
        if os.path.isfile(sat_path):
            img = cv2.imread(sat_path)
            if img is not None:
                self.view_sat.load_pixmap(QPixmap.fromImage(self._cv_to_qimage(img)))
                self.view_sat.fitToView()
        
        if obj.get('use_svg'):
            svg_path = os.path.join(proj_root, 'location', loc_code, f'layout_{loc_code}.svg')
            if os.path.isfile(svg_path):
                self._svg_item = QGraphicsSvgItem(svg_path)
                if self._svg_affine is not None:
                    m = self._svg_affine
                    trans = QTransform(m[0,0], m[1,0], m[0,1], m[1,1], m[0,2], m[1,2])
                    self._svg_item.setTransform(trans)
                self._svg_item.setOpacity(self.slider_alpha.value() / 100.0)
                self.view_sat.scene().addItem(self._svg_item)

        # --- ROI FIX: Handle Transparent vs Black ---
        self._mask_cv = None
        self._roi_overlay = None # Initialize to prevent attribute errors
        roi_path = os.path.join(proj_root, 'location', loc_code, f'roi_{loc_code}.png')
        
        if os.path.isfile(roi_path):
            # Load UNCHANGED to preserve Alpha Channel
            raw_img = cv2.imread(roi_path, cv2.IMREAD_UNCHANGED)
            if raw_img is not None:
                h, w = raw_img.shape[:2]
                
                # We need a single channel mask for checking logic
                # Rule: Transparent = Valid (Inside), Opaque Black = Invalid (Outside)
                if raw_img.ndim == 3 and raw_img.shape[2] == 4:
                    # Extract Alpha
                    alpha = raw_img[:, :, 3]
                    # Logic: If Alpha is 0 (Transparent), it is Valid (255)
                    # If Alpha is 255 (Opaque), it is Invalid (0)
                    self._mask_cv = cv2.bitwise_not(alpha)
                elif raw_img.ndim == 3:
                    # Fallback for no alpha: Assume Black is Invalid
                    gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                    # If black (<10), Invalid (0). Else Valid (255)
                    _, self._mask_cv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                else:
                    self._mask_cv = raw_img

                # Create Visual Overlay (Red for Invalid/Black areas)
                # We want to highlight the BLOCKED areas (pure black) so the user knows they are blocked.
                arr = np.zeros((h, w, 4), dtype=np.uint8)
                
                # Where mask is 0 (Invalid/Outside/Black), make it semi-transparent RED
                is_invalid = (self._mask_cv < 10)
                arr[is_invalid, 0] = 0    # B
                arr[is_invalid, 1] = 0    # G
                arr[is_invalid, 2] = 255  # R
                arr[is_invalid, 3] = 60   # Alpha
                
                img_data = QImage(arr.data, w, h, w*4, QImage.Format_ARGB32).copy()
                self._roi_overlay = QPixmap.fromImage(img_data)
                
                if self.chk_roi.isChecked():
                    self.view_cctv.set_overlay(self._roi_overlay)

    def _init_svg_parser(self, obj):
        """Initializes the Robust SVG Parser for heading logic."""
        if not obj.get('use_svg'): 
            self._svg_parser = None
            return
            
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        svg_path = os.path.join(proj_root, 'location', loc_code, f'layout_{loc_code}.svg')
        
        # Use matrix A from layout_svg
        matrix_a = obj.get('layout_svg', {}).get('A')
        
        self._svg_parser = SVGParser(svg_path, matrix_a)
        if not self._svg_parser.valid:
            self.lbl_status.setText("Warning: SVG file parsed but no segments found.")
        elif not self._svg_parser.orientation_segments:
            self.lbl_status.setText("Warning: No Guidelines/Physical lines found in SVG.")

    # --- Interaction ---

    def _on_box_drawn(self, rect):
        self._current_rect = rect
        self._clear_markers() 
        if self._mask_cv is not None:
            roi_method = 'partial'
            if getattr(self.parent(), 'inspect_obj', None):
                roi_method = getattr(self.parent(), 'inspect_obj').get('roi_method', 'partial')
            valid = self._check_roi(rect, roi_method)
            if not valid:
                self.lbl_status.setText(f"Box rejected by ROI ({roi_method})")
                if self.view_cctv._rect_item: self.view_cctv._rect_item.setPen(QPen(Qt.red, 2, Qt.DashLine))
                return
            else:
                if self.view_cctv._rect_item: self.view_cctv._rect_item.setPen(QPen(Qt.green, 2, Qt.DashLine))
        self.lbl_status.setText(f"Box drawn: {int(rect.width())}x{int(rect.height())}")
        self._on_confirm_points()

    def _check_roi(self, rect, method):
        if self._mask_cv is None: return True
        mask = self._mask_cv
        h_img, w_img = mask.shape[:2]

        x = max(0, min(w_img-1, int(rect.x())))
        y = max(0, min(h_img-1, int(rect.y())))
        w = int(rect.width())
        h = int(rect.height())

        corners = [
            (x, y),
            (min(x+w, w_img-1), y),
            (x, min(y+h, h_img-1)),
            (min(x+w, w_img-1), min(y+h, h_img-1))
        ]

        valid_corners = 0
        for cx, cy in corners:
            # We defined self._mask_cv such that 255 is VALID (Transparent/Inside)
            # and 0 is INVALID (Black/Outside)
            if mask[cy, cx] > 128:
                valid_corners += 1

        if method == 'in': return valid_corners == 4
        return valid_corners >= 1

    def _on_reset_box(self):
        self._current_rect = None
        if self.view_cctv._rect_item:
            try:
                if self.view_cctv._rect_item.scene() == self.view_cctv.scene():
                    self.view_cctv.scene().removeItem(self.view_cctv._rect_item)
            except RuntimeError: pass
            self.view_cctv._rect_item = None
        self._clear_markers()

    def _refresh_visuals(self):
        if self._proj_point_sat is not None:
            self._draw_floor_box()
            if self._show_3d_active: self._on_show_3d()

    def _on_confirm_points(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and getattr(host, 'inspect_obj', None):
            host.inspect_obj['ref_method'] = self.cb_ref.currentText()
            host.inspect_obj['proj_method'] = self.cb_proj.currentText()

        if not self._current_rect: 
            self.lbl_status.setText("Draw a box first.")
            return
        
        self._clear_markers()
        rect = self._current_rect
        cx = rect.x() + rect.width()/2
        cy = rect.y() + rect.height()/2
        if self.cb_ref.currentText() == "center_bottom_side": cy = rect.y() + rect.height()
        self._ref_point_cctv = (cx, cy)
        
        if self._K is not None:
            src = np.array([[[cx, cy]]], dtype=np.float64)
            dst = cv2.undistortPoints(src, self._K, self._D, P=self._new_K)
            u_undist, v_undist = dst[0,0]
        else:
            u_undist, v_undist = cx, cy
            
        src_h = np.array([[[u_undist, v_undist]]], dtype=np.float64)
        pt_sat = cv2.perspectiveTransform(src_h, self._H)[0,0]
        
        proj_method = self.cb_proj.currentText()
        h_obj = self.spin_h.value()
        final_sat = pt_sat
        
        if proj_method != "match":
            eff_h = h_obj if proj_method == "down_h" else (h_obj/2.0)
            vec = pt_sat - self._cam_sat
            if self._z_cam != 0:
                factor = (self._z_cam - eff_h) / self._z_cam
                final_sat = self._cam_sat + (vec * factor)
        
        self._proj_point_sat = final_sat
        self._gc_point_cctv = self._sat_to_cctv(final_sat)
        
        self._draw_marker(self.view_cctv, cx, cy, "Ref", Qt.red)
        if proj_method != "match":
            dist = np.linalg.norm(np.array([cx, cy]) - np.array(self._gc_point_cctv))
            if dist > 2: self._draw_marker(self.view_cctv, self._gc_point_cctv[0], self._gc_point_cctv[1], "GC", Qt.green)
            
        self._draw_marker(self.view_sat, final_sat[0], final_sat[1], "PROJ", Qt.cyan)
        
        self._draw_floor_box()
        if self._show_3d_active: self._on_show_3d()

    def _toggle_heading_mode(self, checked):
        self.slider_head.setEnabled(not checked)
        if not checked:
            self._heading_deg = self.slider_head.value()
            self._draw_floor_box()
            self.lbl_status.setText("Manual heading mode")
        else:
            # Auto mode: verify parser availability
            if not getattr(self, '_svg_parser', None) or not getattr(self._svg_parser, 'valid', False):
                self.lbl_status.setText("Auto Heading enabled but no SVG loaded or no segments found.")
            elif not self._svg_parser.orientation_segments:
                self.lbl_status.setText("Auto Heading enabled but SVG has no guideline segments.")
            else:
                self.lbl_status.setText("Auto Heading enabled.")
            self._draw_floor_box()

    def _on_heading_changed(self, val):
        self._heading_deg = float(val)
        self._draw_floor_box()
        if self._show_3d_active: self._on_show_3d()

    def _draw_floor_box(self):
        if self._proj_point_sat is None: return
        
        if self._highlight_line:
            try:
                if self._highlight_line.scene() == self.view_sat.scene(): self.view_sat.scene().removeItem(self._highlight_line)
            except RuntimeError: pass
            self._highlight_line = None

        # --- AUTO HEADING LOGIC ---
        # --- DEBUG: Draw ALL SVG Lines faintly ---
        # This helps you see if the SVG is loaded but misaligned
        if self.chk_auto_head.isChecked() and self._svg_parser:
            # Clean up old debug lines
            if hasattr(self, '_debug_items') and self._debug_items:
                for it in self._debug_items:
                    try:
                        if it.scene() == self.view_sat.scene(): self.view_sat.scene().removeItem(it)
                    except Exception:
                        pass
            self._debug_items = []
            pen_debug = QPen(QColor(200, 200, 200, 50), 1)
            for p1, p2 in self._svg_parser.orientation_segments:
                try:
                    l = self.view_sat.scene().addLine(p1[0], p1[1], p2[0], p2[1], pen_debug)
                    l.setZValue(5)
                    self._debug_items.append(l)
                except Exception:
                    pass

        # --- AUTO HEADING LOGIC ---
        if self.chk_auto_head.isChecked() and self._svg_parser:
            heading, p1, p2 = self._svg_parser.get_nearest_heading_info(self._proj_point_sat)
            if heading is not None:
                self._heading_deg = heading
                # Draw Highlight Line (Thick Magenta)
                pen = QPen(Qt.magenta, 4)
                self._highlight_line = self.view_sat.scene().addLine(p1[0], p1[1], p2[0], p2[1], pen)
                self._highlight_line.setZValue(100) # Ensure on top
            else:
                self.lbl_status.setText("No SVG guidelines found nearby.")
            # If 3D view is active, update the 3D box to reflect new heading
            if getattr(self, '_show_3d_active', False):
                try:
                    self._on_show_3d()
                except Exception:
                    pass
        # --------------------------
        
        w_m = self.spin_w.value(); l_m = self.spin_l.value()
        w_px = w_m * self._px_per_m; l_px = l_m * self._px_per_m
        cx, cy = self._proj_point_sat
        dx, dy = l_px/2.0, w_px/2.0
        corners = [[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]]
        
        rad = math.radians(self._heading_deg)
        c, s = math.cos(rad), math.sin(rad)
        rot_corners = []
        for x, y in corners:
            rx = x*c - y*s + cx; ry = x*s + y*c + cy
            rot_corners.append((rx, ry))
            
        if self._floor_poly:
            try:
                if self._floor_poly.scene() == self.view_sat.scene(): self.view_sat.scene().removeItem(self._floor_poly)
            except RuntimeError: pass
            self._floor_poly = None
            
        poly = QGraphicsPolygonItem()
        qp = QPolygonF([QPointF(x, y) for x, y in rot_corners])
        poly.setPolygon(qp)
        poly.setPen(QPen(Qt.green, 2)); poly.setBrush(QBrush(QColor(0, 255, 0, 80)))
        poly.setZValue(15) 
        self.view_sat.scene().addItem(poly)
        self._floor_poly = poly
        self._floor_corners_sat = rot_corners

    def _sat_to_cctv(self, pt_sat):
        src = np.array([[[pt_sat[0], pt_sat[1]]]], dtype=np.float64)
        pt_undist_px = cv2.perspectiveTransform(src, self._H_inv)[0,0]
        fx, fy = self._new_K[0,0], self._new_K[1,1]
        cx, cy = self._new_K[0,2], self._new_K[1,2]
        x_n = (pt_undist_px[0] - cx) / fx; y_n = (pt_undist_px[1] - cy) / fy
        obj_pts = np.array([[[x_n, y_n, 1.0]]], dtype=np.float32)
        img_pts, _ = cv2.projectPoints(obj_pts, (0,0,0), (0,0,0), self._K, self._D)
        return img_pts[0,0]

    def _on_show_3d(self):
        if not hasattr(self, '_floor_corners_sat'): return
        scene = self.view_cctv.scene()
        for item in self._wireframe_items:
            try:
                if item.scene() == scene: scene.removeItem(item)
            except RuntimeError: pass
        self._wireframe_items = []
        cctv_floor = [self._sat_to_cctv(pt) for pt in self._floor_corners_sat]
        h_obj = self.spin_h.value()
        cctv_ceil = []
        for x, y in self._floor_corners_sat:
            pt_true = np.array([x, y]); vec = pt_true - self._cam_sat
            if self._z_cam != h_obj and self._z_cam != 0: factor = self._z_cam / (self._z_cam - h_obj) 
            else: factor = 100.0
            pt_app = self._cam_sat + (vec * factor)
            cctv_ceil.append(self._sat_to_cctv(pt_app))
        self._draw_3d_box(cctv_floor, cctv_ceil)

    def _draw_3d_box(self, floor_pts, ceil_pts):
        scene = self.view_cctv.scene()
        pen_f = QPen(Qt.green, 2); pen_c = QPen(Qt.red, 2); pen_v = QPen(Qt.yellow, 1)
        def add_line(p1, p2, pen):
            l = scene.addLine(p1[0], p1[1], p2[0], p2[1], pen); l.setZValue(20)
            self._wireframe_items.append(l)
        for i in range(4):
            add_line(floor_pts[i], floor_pts[(i+1)%4], pen_f)
            add_line(ceil_pts[i], ceil_pts[(i+1)%4], pen_c)
            add_line(floor_pts[i], ceil_pts[i], pen_v)

    def _clear_markers(self):
        s = self.view_sat.scene()
        if s:
            for i in self._sat_markers:
                try:
                    if i.scene() == s: s.removeItem(i)
                except RuntimeError: pass
        self._sat_markers = []
        c = self.view_cctv.scene()
        if c:
            for i in self._cctv_markers:
                try:
                    if i.scene() == c: c.removeItem(i)
                except RuntimeError: pass
        self._cctv_markers = []
        if self._floor_poly:
            try:
                if self._floor_poly.scene(): self._floor_poly.scene().removeItem(self._floor_poly)
            except RuntimeError: pass
            self._floor_poly = None
        if not self._show_3d_active:
            for w in self._wireframe_items:
                try:
                    if w.scene(): w.scene().removeItem(w)
                except RuntimeError: pass
            self._wireframe_items = []
        if self._highlight_line:
            try:
                if self._highlight_line.scene(): self._highlight_line.scene().removeItem(self._highlight_line)
            except RuntimeError: pass
            self._highlight_line = None

    def _on_toggle_3d(self, checked):
        # Toggle 3D wireframe only (do not change ROI overlay here)
        self._show_3d_active = checked
        if checked:
            self.lbl_status.setText("3D box shown")
            try:
                self._on_show_3d()
            except Exception:
                pass
        else:
            self.lbl_status.setText("3D box hidden")
            # Clear 3D lines
            for item in list(self._wireframe_items):
                try:
                    if item.scene(): item.scene().removeItem(item)
                except Exception:
                    pass
            self._wireframe_items = []

    def _on_toggle_roi(self, checked):
        if checked and self._roi_overlay: self.view_cctv.set_overlay(self._roi_overlay)
        else: self.view_cctv.clear_overlay()

    def _on_alpha_changed(self, val):
        if self._svg_item: self._svg_item.setOpacity(val / 100.0)

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and hasattr(host, 'current_step_index'):
            try:
                host.current_step_index = 13
                host._show_stage(13)
                host._update_progress_to_index(13)
            except: pass

    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None: return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).copy()

    def _draw_marker(self, viewer, x, y, label, color):
        scene = viewer.scene()
        el = scene.addEllipse(x-3, y-3, 6, 6, QPen(color, 2), QBrush(color))
        el.setZValue(25) 
        t = scene.addSimpleText(label)
        t.setBrush(QBrush(color))
        t.setPos(x+5, y-10)
        t.setZValue(25)
        if viewer == self.view_sat: self._sat_markers.extend([el, t])
        else: self._cctv_markers.extend([el, t])

    def _draw_line(self, viewer, p1, p2, color, width=2):
        scene = viewer.scene()
        l = scene.addLine(p1[0], p1[1], p2[0], p2[1], QPen(color, width))
        l.setZValue(15)
        if viewer == self.view_sat: self._sat_markers.append(l)