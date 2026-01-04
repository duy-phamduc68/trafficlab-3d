import os
import xml.etree.ElementTree as ET
import numpy as np
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QColor, QTransform, QPainter, QPen, QBrush, QFont
from PyQt5.QtSvg import QGraphicsSvgItem, QSvgRenderer
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QComboBox, QScrollArea, QSlider, QGroupBox, QMessageBox, QFrame,
    QStackedWidget, QSplitter
)

from .undistort_stage import ImageViewer

class SVGStage(QWidget):
    """
    SVG Alignment Stage.
    
    Phase 1: Setup
    - Left View: Satellite Map (shows available Sat Anchors).
    - Right View: Raw SVG (shows available SVG Anchors).
    - Sidebar: Link SVG points -> Sat points.
    
    Phase 2: Result
    - Computes Affine Transform.
    - Overlay View: Satellite Map + Transformed SVG (with Opacity slider).
    """

    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        
        # Data
        self.svg_anchors = []   
        self.sat_anchors = []   
        self.association_widgets = {} 
        
        # Graphics Item for Result View
        self._svg_item = None
        
        # --- UI ---
        main_layout = QHBoxLayout(self)
        
        # 1. Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(350)
        sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_vbox = QVBoxLayout(sidebar)
        side_vbox.setSpacing(10)
        
        lbl_title = QLabel("SVG Alignment")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 16px; color: #fff;")
        side_vbox.addWidget(lbl_title)
        
        desc = QLabel(
            "1. Identify matching points in the Left (Sat) and Right (SVG) panels.\n"
            "2. Link them in the list below.\n"
            "3. Click Compute to see the overlay."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #aaa; font-size: 11px;")
        side_vbox.addWidget(desc)
        
        # Association List
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: #333; border: none;")
        self.scroll_content = QWidget()
        self.form_layout = QVBoxLayout(self.scroll_content)
        scroll.setWidget(self.scroll_content)
        side_vbox.addWidget(scroll, 1)
        
        # Preview Controls (Only active in Result mode)
        self.vis_grp = QGroupBox("Result Preview")
        self.vis_grp.setEnabled(False) 
        v_layout = QVBoxLayout(self.vis_grp)
        
        h_slide = QHBoxLayout()
        h_slide.addWidget(QLabel("SVG Opacity:"))
        self.slider_alpha = QSlider(Qt.Horizontal)
        self.slider_alpha.setRange(0, 100)
        self.slider_alpha.setValue(50)
        self.slider_alpha.valueChanged.connect(self._update_opacity)
        h_slide.addWidget(self.slider_alpha)
        v_layout.addLayout(h_slide)
        
        self.btn_back_edit = QPushButton("Back to Editing")
        self.btn_back_edit.clicked.connect(self._on_back_edit)
        v_layout.addWidget(self.btn_back_edit)
        
        side_vbox.addWidget(self.vis_grp)
        
        # Compute Button
        self.btn_compute = QPushButton("Compute Alignment")
        self.btn_compute.setFixedHeight(40)
        self.btn_compute.setStyleSheet("background-color: #2a84ff; color: white; font-weight: bold;")
        self.btn_compute.clicked.connect(self._on_compute)
        side_vbox.addWidget(self.btn_compute)
        
        self.lbl_status = QLabel("Status: Waiting for input")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #ccc; font-style: italic;")
        side_vbox.addWidget(self.lbl_status)
        
        side_vbox.addStretch()
        
        self.btn_proceed = QPushButton("Proceed")
        self.btn_proceed.clicked.connect(self._on_proceed)
        side_vbox.addWidget(self.btn_proceed)
        
        main_layout.addWidget(sidebar)
        
        # 2. Main Area (Stacked Widget)
        self.stack = QStackedWidget()
        
        # Page 0: Setup (Split View)
        self.page_setup = QWidget()
        setup_layout = QVBoxLayout(self.page_setup)
        setup_layout.setContentsMargins(0,0,0,0)
        
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Sat View
        left_cont = QWidget()
        l_vbox = QVBoxLayout(left_cont)
        l_vbox.setContentsMargins(0,0,0,0)
        l_vbox.addWidget(QLabel("Satellite Map (Reference)"))
        self.view_sat_setup = ImageViewer()
        l_vbox.addWidget(self.view_sat_setup)
        
        # Right: SVG View
        right_cont = QWidget()
        r_vbox = QVBoxLayout(right_cont)
        r_vbox.setContentsMargins(0,0,0,0)
        r_vbox.addWidget(QLabel("SVG Layout (Source)"))
        self.view_svg_setup = ImageViewer()
        r_vbox.addWidget(self.view_svg_setup)
        
        splitter.addWidget(left_cont)
        splitter.addWidget(right_cont)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        
        setup_layout.addWidget(splitter)
        
        # Page 1: Result (Overlay View)
        self.page_result = QWidget()
        res_layout = QVBoxLayout(self.page_result)
        res_layout.setContentsMargins(0,0,0,0)
        res_layout.addWidget(QLabel("Aligned Overlay (Adjust Opacity in Sidebar)"))
        self.view_result = ImageViewer()
        res_layout.addWidget(self.view_result)
        
        self.stack.addWidget(self.page_setup)
        self.stack.addWidget(self.page_result)
        
        main_layout.addWidget(self.stack, 1)

    def showEvent(self, event):
        super().showEvent(event)
        if not HAS_CV2:
            self.lbl_status.setText("Error: OpenCV missing")
            return
            
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host or not getattr(host, 'inspect_obj', None): return
        
        obj = host.inspect_obj
        
        # Always start in Setup mode
        self._on_back_edit()
        
        # 1. Load Data
        self._load_sat_anchors(obj)
        if not self._load_svg_anchors(obj):
            return
            
        # 2. Setup UI
        self._build_association_ui(obj)
        
        # 3. Render Setup Views
        self._render_setup_views(obj)

    def _on_back_edit(self):
        """Switch back to split view."""
        self.stack.setCurrentIndex(0)
        self.vis_grp.setEnabled(False)
        self.btn_compute.setEnabled(True)
        self.lbl_status.setText("Editing Mode")

    def _render_setup_views(self, obj):
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        if not loc_code: return
        
        # --- A. Setup Satellite View ---
        sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
        if os.path.isfile(sat_path):
            img = cv2.imread(sat_path)
            if img is not None:
                self.view_sat_setup.load_pixmap(QPixmap.fromImage(self._cv_to_qimage(img)))
                self.view_sat_setup.fitToView()
                
                # Draw Sat Anchors
                for item in self.sat_anchors:
                    x, y = item['pt']
                    self._draw_label(self.view_sat_setup, x, y, item['name'], color=Qt.cyan, text_color=Qt.yellow)

        # --- B. Setup SVG View ---
        svg_path = os.path.join(proj_root, 'location', loc_code, f'layout_{loc_code}.svg')
        if os.path.exists(svg_path):
            # Render SVG to Pixmap for display
            renderer = QSvgRenderer(svg_path)
            if renderer.isValid():
                # Get size from viewBox or default
                sz = renderer.defaultSize()
                # Scale up if too small for visibility
                scale = 1.0
                if sz.width() < 1000: scale = 2.0
                
                img = QImage(int(sz.width()*scale), int(sz.height()*scale), QImage.Format_ARGB32)
                img.fill(Qt.white) # White background for visibility
                painter = QPainter(img)
                renderer.render(painter)
                painter.end()
                
                self.view_svg_setup.load_pixmap(QPixmap.fromImage(img))
                self.view_svg_setup.fitToView()
                
                # Draw SVG Anchors (Scaled)
                for item in self.svg_anchors:
                    x, y = item['pt']
                    # We rendered with 'scale', so we must scale points too?
                    # Actually, standard QSvgRenderer renders to the QImage size.
                    # If we used render(painter), it fits the viewBox into the QImage.
                    # To be precise, we should map coordinates. 
                    # Simpler approach: Just render at default size.
                    
                # Re-render at exact default size to match coordinates
                img_def = QImage(sz, QImage.Format_ARGB32)
                img_def.fill(Qt.white)
                painter = QPainter(img_def)
                renderer.render(painter)
                painter.end()
                self.view_svg_setup.load_pixmap(QPixmap.fromImage(img_def))
                self.view_svg_setup.fitToView()
                
                for item in self.svg_anchors:
                    x, y = item['pt']
                    self._draw_label(self.view_svg_setup, x, y, item['name'], color=Qt.red, text_color=Qt.blue)

    def _draw_label(self, viewer, x, y, text, color, text_color):
        scene = viewer.scene()
        rad = 5
        el = scene.addEllipse(x-rad, y-rad, rad*2, rad*2)
        el.setPen(QPen(color, 2))
        el.setBrush(QBrush(color))
        
        t = scene.addSimpleText(text)
        t.setBrush(QBrush(text_color))
        t.setFont(QFont("Arial", 10, QFont.Bold))
        t.setPos(x+8, y-8)
        
        # Optional: Add background rect for text readability
        # (Simplified here)

    def _load_sat_anchors(self, obj):
        self.sat_anchors = []
        hom = obj.get('homography', {})
        for item in hom.get('anchors_list', []):
            if item.get('coords_sat'):
                self.sat_anchors.append({
                    'name': item.get('name', f"ID {item['id']}"),
                    'pt': item['coords_sat']
                })

    def _load_svg_anchors(self, obj):
        self.svg_anchors = []
        proj_root = getattr(self, 'project_root', None) or os.getcwd()
        loc_code = obj.get('meta', {}).get('location_code')
        svg_path = os.path.join(proj_root, 'location', loc_code, f'layout_{loc_code}.svg')
        
        if not os.path.exists(svg_path):
            self.lbl_status.setText(f"SVG file not found: {svg_path}")
            return False
            
        try:
            tree = ET.parse(svg_path)
            root = tree.getroot()
            ns = {'svg': 'http://www.w3.org/2000/svg'}
            
            # Find 'Anchors' group
            anchors_group = root.find(".//svg:g[@id='Anchors']", ns)
            circles = []
            if anchors_group is not None:
                circles = anchors_group.findall("svg:circle", ns)
            else:
                circles = root.findall(".//svg:circle", ns) # Fallback
                
            for elem in circles:
                cx = float(elem.get('cx'))
                cy = float(elem.get('cy'))
                name = elem.get('data-name') or elem.get('id') or f"SVG_{len(self.svg_anchors)}"
                svg_id = elem.get('id') or name
                
                self.svg_anchors.append({
                    'id': svg_id,
                    'name': name,
                    'pt': (cx, cy)
                })
                
            if not self.svg_anchors:
                self.lbl_status.setText("No anchors (circles) found in SVG.")
                return False
                
            return True
            
        except Exception as e:
            self.lbl_status.setText(f"Error parsing SVG: {e}")
            return False

    def _build_association_ui(self, obj):
        while self.form_layout.count():
            child = self.form_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        self.association_widgets = {}
        
        saved_pairs = obj.get('layout_svg', {}).get('association_pairs', [])
        saved_map = {p['svg_id']: p['sat_id'] for p in saved_pairs}
        
        for svg_a in self.svg_anchors:
            row_widget = QFrame()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            lbl = QLabel(svg_a['name'])
            lbl.setFixedWidth(120)
            lbl.setStyleSheet("color: #ccc; font-weight: bold;")
            row_layout.addWidget(lbl)
            
            combo = QComboBox()
            combo.addItem("--- Ignore ---", None)
            
            current_sat_id = saved_map.get(svg_a['id'])
            select_idx = 0
            
            for i, sat_a in enumerate(self.sat_anchors):
                combo.addItem(sat_a['name'], sat_a['name'])
                if current_sat_id == sat_a['name']:
                    select_idx = i + 1
                elif select_idx == 0 and sat_a['name'] in svg_a['name']: 
                    select_idx = i + 1
            
            combo.setCurrentIndex(select_idx)
            row_layout.addWidget(combo)
            
            self.form_layout.addWidget(row_widget)
            self.association_widgets[svg_a['id']] = combo

    def _on_compute(self):
        src_pts = []
        dst_pts = []
        pairs_record = []
        
        for svg_a in self.svg_anchors:
            combo = self.association_widgets.get(svg_a['id'])
            if not combo: continue
            sat_name = combo.currentData()
            if sat_name is None: continue
            
            sat_a = next((s for s in self.sat_anchors if s['name'] == sat_name), None)
            if sat_a:
                src_pts.append(list(svg_a['pt']))
                dst_pts.append(list(sat_a['pt']))
                pairs_record.append({
                    "svg_id": svg_a['id'],
                    "sat_id": sat_name
                })
        
        if len(src_pts) < 3:
            QMessageBox.warning(self, "Error", "Need at least 3 linked pairs.")
            return
            
        try:
            src_arr = np.array(src_pts, dtype=np.float32)
            dst_arr = np.array(dst_pts, dtype=np.float32)
            
            # Affine
            M, inliers = cv2.estimateAffine2D(src_arr, dst_arr)
            if M is None: raise Exception("Affine calculation failed.")
            
            # --- Switch to Result View ---
            self.stack.setCurrentIndex(1)
            self.vis_grp.setEnabled(True)
            self.btn_compute.setEnabled(False)
            
            # Load Sat Background
            host = getattr(self, 'host_tab', None) or self.parent()
            obj = host.inspect_obj
            proj_root = getattr(self, 'project_root', None) or os.getcwd()
            loc_code = obj.get('meta', {}).get('location_code')
            sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
            
            if os.path.isfile(sat_path):
                img = cv2.imread(sat_path)
                self.view_result.load_pixmap(QPixmap.fromImage(self._cv_to_qimage(img)))
                self.view_result.fitToView()
            
            # Add SVG Item
            svg_path = os.path.join(proj_root, 'location', loc_code, f'layout_{loc_code}.svg')
            self._svg_item = QGraphicsSvgItem(svg_path)
            
            # Apply Transform
            a, b, tx = M[0]
            c, d, ty = M[1]
            trans = QTransform(a, c, b, d, tx, ty) # Note constructor order
            self._svg_item.setTransform(trans)
            self._svg_item.setOpacity(self.slider_alpha.value() / 100.0)
            
            self.view_result.scene().addItem(self._svg_item)
            
            # Save
            layout_data = obj.setdefault('layout_svg', {})
            layout_data['A'] = M.tolist()
            layout_data['association_pairs'] = pairs_record
            host.inspect_obj = obj
            
            self.lbl_status.setText("Computed. Use opacity slider to check.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _update_opacity(self, val):
        if self._svg_item:
            self._svg_item.setOpacity(val / 100.0)

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if not host: return
        use_roi = False
        if getattr(host, 'inspect_obj', None):
            use_roi = host.inspect_obj.get('use_roi', False)
        next_idx = 11 if use_roi else 12
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