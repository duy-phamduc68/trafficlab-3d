import sys
import os
import json
import gzip
import cv2
import time
import hashlib
import math
import re
import xml.etree.ElementTree as ET
import numpy as np

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, 
                             QCheckBox, QSlider, QGroupBox, QProgressBar, 
                             QSplitter, QGraphicsView, QGraphicsScene, 
                             QGraphicsPixmapItem, QGraphicsPolygonItem, 
                             QGraphicsLineItem, QGraphicsSimpleTextItem, 
                             QGraphicsItemGroup, QSpinBox, QShortcut, QToolButton, 
                             QMessageBox, QGraphicsEllipseItem, QDialog, QTextEdit, QPushButton)
from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, QLineF
from PyQt5.QtGui import (QImage, QPixmap, QColor, QPen, QBrush, 
                         QPolygonF, QTransform, QPainter, QKeySequence)

# --- Constants ---
OUTPUT_DIR = "output"

def get_color_from_string(s):
    hash_object = hashlib.md5(s.encode())
    hex_hash = hash_object.hexdigest()
    r = int(hex_hash[0:2], 16)
    g = int(hex_hash[2:4], 16)
    b = int(hex_hash[4:6], 16)
    return QColor(r, g, b)

# ==========================================
# ROBUST MANUAL SVG PARSER
# ==========================================
class SvgGeometryParser:
    def __init__(self, svg_path, affine_matrix_array):
        self.tree = ET.parse(svg_path)
        self.root = self.tree.getroot()
        self.ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        self.M_align = np.identity(3)
        if affine_matrix_array:
            self.M_align[:2, :] = np.array(affine_matrix_array)

        # Pre-load CSS classes
        self.css_classes = {
            'cls-1': {'fill': '#afafaf'},
            'cls-2': {'fill': '#939393'},
            'cls-3': {'fill': '#fff', 'stroke': 'none'},
            'cls-4': {'fill': 'none', 'stroke': '#ff0', 'stroke-width': '1px'},
            'cls-5': {'fill': 'none', 'stroke': 'lime', 'stroke-width': '1px'},
            'cls-6': {'fill': '#fff', 'stroke': '#000', 'stroke-width': '2px'},
            'cls-7': {'fill': 'red'}
        }
        self._parse_css_from_file()
        
        self.layer_items = {
            'Background': [], 'Aesthetic': [], 'Guidelines': [], 
            'Physical': [], 'Anchors': []
        }
        self._parse_layers()

    def _parse_css_from_file(self):
        style_elem = self.root.find('.//svg:style', self.ns)
        if style_elem is None: style_elem = self.root.find('.//style')

        if style_elem is not None and style_elem.text:
            clean_css = re.sub(r'/\*.*?\*/', '', style_elem.text, flags=re.DOTALL)
            for match in re.finditer(r'([^{]+)\{(.*?)\}', clean_css, re.DOTALL):
                selectors_str = match.group(1)
                content = match.group(2)
                props = {}
                for prop_match in re.finditer(r'([\w-]+)\s*:\s*([^;]+)', content):
                    key, val = prop_match.group(1).strip(), prop_match.group(2).strip()
                    props[key] = val
                for sel in selectors_str.split(','):
                    cls_name = sel.strip().lstrip('.')
                    if cls_name in self.css_classes: self.css_classes[cls_name].update(props)
                    else: self.css_classes[cls_name] = props

    def _parse_transform_str(self, txt):
        M = np.identity(3)
        if not txt: return M
        ops = re.findall(r'(\w+)\s*\(([^)]+)\)', txt)
        for name, args in ops:
            vals = list(map(float, filter(None, re.split(r'[ ,]+', args.strip()))))
            T = np.identity(3)
            if name == 'translate': T[0,2], T[1,2] = vals[0], vals[1] if len(vals)>1 else 0
            elif name == 'rotate':
                rad = math.radians(vals[0]); c,s = math.cos(rad), math.sin(rad)
                if len(vals)==3:
                    cx,cy=vals[1],vals[2]; T1=np.eye(3); T1[0,2]=cx; T1[1,2]=cy
                    R=np.eye(3); R[:2,:2]=[[c,-s],[s,c]]; T2=np.eye(3); T2[0,2]=-cx; T2[1,2]=-cy
                    T = T1 @ R @ T2
                else: T[:2,:2] = [[c,-s],[s,c]]
            elif name == 'matrix' and len(vals)==6:
                T = np.array([[vals[0],vals[2],vals[4]],[vals[1],vals[3],vals[5]],[0,0,1]])
            M = M @ T
        return M

    def _apply_transform(self, pts, elem_matrix):
        M_total = self.M_align @ elem_matrix
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        return (M_total @ homo.T).T[:, :2]

    def _get_qt_style(self, elem, layer_name):
        styles = {'stroke': 'none', 'stroke-width': '1px', 'fill': 'none'}
        if layer_name == 'Physical': styles.update({'fill': '#fff', 'stroke': '#000'})
        elif layer_name == 'Guidelines': styles.update({'stroke': '#ff0', 'fill': 'none'})

        cls_str = elem.get('class')
        if cls_str:
            for c in cls_str.split():
                if c in self.css_classes: styles.update(self.css_classes[c])
        
        style_attr = elem.get('style')
        if style_attr:
            for prop in style_attr.split(';'):
                if ':' in prop:
                    k, v = prop.split(':', 1)
                    styles[k.strip()] = v.strip()

        for key in ['stroke', 'stroke-width', 'fill']:
            val = elem.get(key)
            if val: styles[key] = val
            
        pen = QPen(Qt.NoPen); brush = QBrush(Qt.NoBrush)
        s_col = styles.get('stroke')
        if s_col and s_col != 'none':
            try:
                c = QColor(s_col)
                if c.isValid(): pen = QPen(c, float(styles.get('stroke-width','1').replace('px','')))
            except: pass
        f_col = styles.get('fill')
        if f_col and f_col != 'none':
            try:
                c = QColor(f_col)
                if c.isValid(): brush = QBrush(c)
            except: pass
        return pen, brush

    def _parse_layers(self):
        for layer_name in self.layer_items.keys():
            group = self.root.find(f".//svg:g[@id='{layer_name}']", self.ns)
            if group is None: group = self.root.find(f".//*[@id='{layer_name}']")
            if group is None: continue
            
            for elem in group:
                tag = elem.tag.split('}')[-1]
                mat = self._parse_transform_str(elem.get('transform'))
                pen, brush = self._get_qt_style(elem, layer_name)
                
                item = None
                if tag == 'line':
                    try:
                        p1 = [float(elem.get('x1',0)), float(elem.get('y1',0))]
                        p2 = [float(elem.get('x2',0)), float(elem.get('y2',0))]
                        t = self._apply_transform(np.array([p1, p2]), mat)
                        item = QGraphicsLineItem(QLineF(QPointF(*t[0]), QPointF(*t[1])))
                        item.setPen(pen)
                    except: pass
                elif tag in ['rect', 'polygon', 'polyline']:
                    pts = []
                    if tag == 'rect':
                        try:
                            x,y,w,h = float(elem.get('x',0)), float(elem.get('y',0)), float(elem.get('width',0)), float(elem.get('height',0))
                            pts = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
                        except: pass
                    else:
                        raw = re.split(r'[ ,]+', elem.get('points','').strip())
                        raw = [x for x in raw if x]
                        if raw: 
                            try: pts = np.array(raw, dtype=float).reshape(-1,2)
                            except: pass
                    if len(pts) > 0:
                        t = self._apply_transform(pts, mat)
                        qpoly = QPolygonF([QPointF(*p) for p in t])
                        item = QGraphicsPolygonItem(qpoly)
                        item.setPen(pen); item.setBrush(brush)
                
                if item: self.layer_items[layer_name].append(item)

# ==========================================
# GRAPHICS VIEWS
# ==========================================
class SatGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.parent_inspector = None

    def wheelEvent(self, event):
        zoom_in = 1.15
        zoom_out = 1 / zoom_in
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        if event.angleDelta().y() > 0:
            self.scale(zoom_in, zoom_in)
        else:
            self.scale(zoom_out, zoom_out)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

class CCTVGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        zoom_in = 1.15
        zoom_out = 1 / zoom_in
        if event.angleDelta().y() > 0:
            self.scale(zoom_in, zoom_in)
        else:
            self.scale(zoom_out, zoom_out)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)

# ==========================================
# VISUALIZATION TAB
# ==========================================
class VisualizationTab(QWidget):
    def __init__(self):
        super().__init__()
        
        # State
        self.current_json_data = None
        self.cap = None
        self.g_data = None 
        self.is_paused = True
        self.current_frame_idx = 0
        self.json_frame_map = {} 
        self.speed_display_cache = {} 
        
        # ROI
        self.roi_overlay_item = None
        
        self.last_real_time = 0
        self.actual_fps = 0.0
        self.target_fps = 30
        
        # UI Defaults
        self.show_tracking = True
        self.show_box = True
        self.box_thickness = 2
        self.show_roi = False

        self.sat_opacity = 0
        self.show_sat_box = True
        self.sat_box_thick = 2
        self.show_sat_arrow = False
        self.show_sat_label = False
        self.sat_label_size = 12

        # --- NEW: Coordinate Dot State ---
        self.show_sat_coords_dot = False 

        self.text_color_mode = "White"
        self.speed_update_delay_frames = 30
        
        self.show_fov = False
        self.fov_fill_opacity = 25
        
        self.has_3d_data = False
        self.show_3d = True
        self.show_label = True
        self.face_opacity = 50

        self.svg_layer_groups = {} 
        self.file_paths = []

        self.init_ui()
        self.load_file_list()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- SIDEBAR ---
        sidebar = QWidget()
        sidebar.setFixedWidth(340)
        sidebar_layout = QVBoxLayout(sidebar)
        
        # File Selection
        file_group = QGroupBox("Select File")
        file_layout = QVBoxLayout()
        self.file_combo = QComboBox()
        # Ensure the slot is invoked without receiving the index argument
        self.file_combo.currentIndexChanged.connect(lambda idx: self.load_selected_file())
        file_layout.addWidget(self.file_combo)
        
        # Buttons row: Load (left) and Refresh (right) with outline
        btn_row = QHBoxLayout()
        self.btn_load_selected = QToolButton()
        self.btn_load_selected.setText("Load")
        self.btn_load_selected.clicked.connect(self.load_selected_file)
        self.btn_load_selected.setStyleSheet("QToolButton { border: 1px solid #888; border-radius:4px; padding:4px; }")
        btn_row.addWidget(self.btn_load_selected)
        # Add List Files button inline so all three are on the same row
        self.btn_list_files = QToolButton()
        self.btn_list_files.setText("List Files")
        self.btn_list_files.clicked.connect(lambda: self.show_file_list_dialog())
        self.btn_list_files.setStyleSheet("QToolButton { border: 1px solid #888; border-radius:4px; padding:4px; }")
        btn_row.addWidget(self.btn_list_files)

        btn_row.addStretch()

        btn_refresh = QToolButton()
        btn_refresh.setText("↻")
        btn_refresh.clicked.connect(self.load_file_list)
        btn_refresh.setStyleSheet("QToolButton { border: 1px solid #888; border-radius:4px; padding:4px; }")
        btn_row.addWidget(btn_refresh)

        file_layout.addLayout(btn_row)
        
        file_group.setLayout(file_layout)
        sidebar_layout.addWidget(file_group)

        # SVG Layers
        layer_group = QGroupBox("Map & SVG Layers")
        layer_layout = QVBoxLayout()
        layer_layout.addWidget(QLabel("Map/SVG Blend:"))
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(self.sat_opacity)
        self.slider_opacity.valueChanged.connect(self.update_sat_layers)
        layer_layout.addWidget(self.slider_opacity)
        
        self.layer_checks = {}
        layer_defaults = {
            'Physical': True,
            'Guidelines': False,
            'Aesthetic': True,
            'Background': True
        }
        for name in ['Physical', 'Guidelines', 'Aesthetic', 'Background']:
            chk = QCheckBox(name)
            chk.setChecked(layer_defaults.get(name, True))
            chk.toggled.connect(lambda c, n=name: self.toggle_svg_layer(n, c))
            layer_layout.addWidget(chk)
            self.layer_checks[name] = chk
        layer_group.setLayout(layer_layout)
        sidebar_layout.addWidget(layer_group)

        # SAT Visuals
        sat_vis_group = QGroupBox("SAT Visualization")
        sat_vis_layout = QVBoxLayout()
        self.chk_sat_box = QCheckBox("Floor Box"); self.chk_sat_box.setChecked(True)
        self.chk_sat_box.toggled.connect(self.update_ui_state)
        # --- NEW: Coords Dot Checkbox ---
        self.chk_sat_coords = QCheckBox("Show Coords Dot")
        self.chk_sat_coords.setChecked(False)
        self.chk_sat_coords.toggled.connect(self.update_ui_state)
        self.chk_sat_arrow = QCheckBox("Heading Arrow"); self.chk_sat_arrow.setChecked(False)
        self.chk_sat_arrow.toggled.connect(self.update_ui_state)
        self.chk_sat_label = QCheckBox("Text Label"); self.chk_sat_label.setChecked(self.show_sat_label)
        self.chk_sat_label.toggled.connect(self.update_ui_state)
        # Add widgets to layout
        sat_vis_layout.addWidget(self.chk_sat_box)
        sat_vis_layout.addWidget(self.chk_sat_coords)
        sat_vis_layout.addWidget(self.chk_sat_arrow)
        sat_vis_layout.addWidget(self.chk_sat_label)
        
        th_lay = QHBoxLayout()
        th_lay.addWidget(QLabel("Box Thick:"))
        self.slider_sat_thick = QSlider(Qt.Horizontal); self.slider_sat_thick.setRange(1, 10); self.slider_sat_thick.setValue(2)
        self.slider_sat_thick.valueChanged.connect(self.update_ui_state)
        th_lay.addWidget(self.slider_sat_thick)
        sat_vis_layout.addLayout(th_lay)

        tx_lay = QHBoxLayout()
        tx_lay.addWidget(QLabel("Size:"))
        self.slider_sat_text = QSlider(Qt.Horizontal); self.slider_sat_text.setRange(6, 48); self.slider_sat_text.setValue(12)
        self.slider_sat_text.valueChanged.connect(self.update_ui_state)
        tx_lay.addWidget(self.slider_sat_text)
        sat_vis_layout.addLayout(tx_lay)
        
        col_lay = QHBoxLayout()
        col_lay.addWidget(QLabel("Color:"))
        self.combo_text_color = QComboBox()
        self.combo_text_color.addItems(["White", "Black", "Yellow"])
        self.combo_text_color.currentIndexChanged.connect(self.update_text_controls)
        col_lay.addWidget(self.combo_text_color)
        sat_vis_layout.addLayout(col_lay)

        fov_row = QHBoxLayout()
        self.chk_fov = QCheckBox("Show FOV Polygon")
        self.chk_fov.setChecked(self.show_fov)
        fov_row.addWidget(self.chk_fov)
        self.chk_fov.toggled.connect(lambda v: self.set_fov_visible(bool(v)))
        sat_vis_layout.addLayout(fov_row)

        fov_alpha_row = QHBoxLayout()
        fov_alpha_row.addWidget(QLabel("FOV Fill %:"))
        self.slider_fov_opacity = QSlider(Qt.Horizontal)
        self.slider_fov_opacity.setRange(0, 100)
        self.slider_fov_opacity.setValue(self.fov_fill_opacity)
        self.slider_fov_opacity.valueChanged.connect(lambda v: self.update_fov_opacity())
        fov_alpha_row.addWidget(self.slider_fov_opacity)
        sat_vis_layout.addLayout(fov_alpha_row)

        spd_lay = QHBoxLayout()
        spd_lay.addWidget(QLabel("Speed Delay:"))
        self.slider_speed_delay = QSlider(Qt.Horizontal); self.slider_speed_delay.setRange(0, 60); self.slider_speed_delay.setValue(10)
        self.slider_speed_delay.valueChanged.connect(self.update_ui_state)
        spd_lay.addWidget(self.slider_speed_delay)
        sat_vis_layout.addLayout(spd_lay)

        sat_vis_group.setLayout(sat_vis_layout)
        sidebar_layout.addWidget(sat_vis_group)
        
        # CCTV Controls
        cctv_group = QGroupBox("CCTV Controls")
        cctv_layout = QVBoxLayout()
        
        self.chk_tracking = QCheckBox("Color by ID"); self.chk_tracking.setChecked(True)
        self.chk_tracking.toggled.connect(self.update_ui_state)
        cctv_layout.addWidget(self.chk_tracking)
        
        # --- 3D Controls ---
        self.chk_3d_box = QCheckBox("Show 3D Box (Hide Text)")
        self.chk_3d_box.setChecked(False); self.chk_3d_box.setEnabled(False) 
        self.chk_3d_box.toggled.connect(self.update_ui_state)
        cctv_layout.addWidget(self.chk_3d_box)
        
        al_lay = QHBoxLayout(); al_lay.addWidget(QLabel("3D Face Opacity:"))
        self.slider_3d_alpha = QSlider(Qt.Horizontal); self.slider_3d_alpha.setRange(0, 255); self.slider_3d_alpha.setValue(50)
        self.slider_3d_alpha.valueChanged.connect(self.update_ui_state)
        al_lay.addWidget(self.slider_3d_alpha); cctv_layout.addLayout(al_lay)

        self.chk_cctv_label = QCheckBox("Show Text (2D Mode)")
        self.chk_cctv_label.setChecked(True)
        self.chk_cctv_label.toggled.connect(self.update_ui_state)
        cctv_layout.addWidget(self.chk_cctv_label)

        self.chk_roi = QCheckBox("Show ROI (Red = Outside)")
        self.chk_roi.setChecked(self.show_roi)
        self.chk_roi.toggled.connect(self.update_ui_state)
        cctv_layout.addWidget(self.chk_roi)

        self.slider_fps = QSlider(Qt.Horizontal); self.slider_fps.setRange(1, 300); self.slider_fps.setValue(40)
        self.slider_fps.valueChanged.connect(self.update_fps_target)
        cctv_layout.addWidget(QLabel("Target FPS:"))
        cctv_layout.addWidget(self.slider_fps)
        
        jump_lay = QHBoxLayout()
        jump_lay.addWidget(QLabel("Jump Frames:"))
        self.spin_jump_frames = QSpinBox()
        self.spin_jump_frames.setRange(1, 10000)
        self.spin_jump_frames.setValue(60)
        jump_lay.addWidget(self.spin_jump_frames)
        cctv_layout.addLayout(jump_lay)
        
        cctv_group.setLayout(cctv_layout)
        sidebar_layout.addWidget(cctv_group)

        self.lbl_info = QLabel("Idle"); self.lbl_info.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_info)
        sidebar_layout.addStretch()
        
        # Shortcuts tooltip
        help_html = (
            "<b>Shortcuts:</b><br>"
            "Space — Pause/Play<br>"
            "R — Reset<br>"
            "[ ] — Previous / Next File<br>"
            "; ' — Back / Forward 1 Frame<br>"
            "&lt; &gt; — Back / Forward N Frames<br>"
            "A — Collapse / Expand Controls<br>"
            "T — Toggle SAT Text Labels<br>"
            "H — Toggle Heading Arrows<br>"
            "G — Toggle Guidelines Layer<br>"
            "1 — Toggle Color by ID<br>"
            "2 — Toggle ROI<br>"
            "3 — Toggle 3D Box<br>"
            "4 — Fit CCTV to Viewport<br>"
            "5 — Toggle View Layout<br>"
            "6 — Toggle Vehicle SAT Coordinates<br>"
            "7 — Toggle Vehicle SAT Floor Box<br>"
            "F — Toggle FOV overlay on SAT map<br>"
        )
        self.btn_help = QToolButton()
        self.btn_help.setText('?')
        self.btn_help.setFixedSize(26, 26)
        self.btn_help.setToolTip(help_html)
        self.btn_help.setStyleSheet('font-weight:bold; border:1px solid #888; border-radius:4px;')
        try:
            self.btn_help.clicked.connect(lambda: QMessageBox.information(self, "Shortcuts", help_html))
        except Exception: pass
        sidebar_layout.addWidget(self.btn_help, alignment=Qt.AlignRight)

        # Collapsible sidebar toggle button
        self.btn_toggle_sidebar = QToolButton()
        self.btn_toggle_sidebar.setCheckable(True)
        self.btn_toggle_sidebar.setChecked(False)
        self.btn_toggle_sidebar.setText('◀')
        self.btn_toggle_sidebar.setFixedWidth(26)
        def _toggle_sidebar(checked):
            if checked:
                sidebar.setVisible(False)
                self.btn_toggle_sidebar.setText('▶')
            else:
                sidebar.setVisible(True)
                self.btn_toggle_sidebar.setText('◀')
        self.btn_toggle_sidebar.toggled.connect(_toggle_sidebar)

        # Shortcuts registration
        try:
            self._shortcuts = []
            def mk(key, handler):
                sc = QShortcut(QKeySequence(key), self)
                sc.setContext(Qt.ApplicationShortcut)
                sc.activated.connect(handler)
                self._shortcuts.append(sc)
                return sc

            mk(Qt.Key_A, lambda: self.btn_toggle_sidebar.toggle())
            mk(Qt.Key_T, lambda: self.chk_sat_label.setChecked(not self.chk_sat_label.isChecked()))
            mk(Qt.Key_H, lambda: self.chk_sat_arrow.setChecked(not self.chk_sat_arrow.isChecked()))
            mk(Qt.Key_G, lambda: self.layer_checks.get('Guidelines').setChecked(not self.layer_checks.get('Guidelines').isChecked()) if self.layer_checks.get('Guidelines') else None)
            mk(Qt.Key_1, lambda: self.chk_tracking.setChecked(not self.chk_tracking.isChecked()))
            mk(Qt.Key_2, lambda: self.chk_roi.setChecked(not self.chk_roi.isChecked()))
            mk(Qt.Key_3, lambda: self.chk_3d_box.setChecked(not self.chk_3d_box.isChecked()) if self.chk_3d_box.isEnabled() else None)
            mk(Qt.Key_4, lambda: self.fit_cctv_to_viewport())
            mk(Qt.Key_5, lambda: self.toggle_view_layout())
            # --- NEW: Shortcut 6 ---
            mk(Qt.Key_6, lambda: self.chk_sat_coords.setChecked(not self.chk_sat_coords.isChecked()))
            # --- NEW: Shortcut 7 (Floor Box Toggle) ---
            mk(Qt.Key_7, lambda: self.chk_sat_box.setChecked(not self.chk_sat_box.isChecked()))
            mk(Qt.Key_F, lambda: self.toggle_fov())
            
            # Frame jumps
            mk(Qt.Key_Less, lambda: self.seek_frames(-int(self.spin_jump_frames.value())))
            mk(Qt.Key_Greater, lambda: self.seek_frames(int(self.spin_jump_frames.value())))
            mk(Qt.Key_Comma, lambda: self.seek_frames(-int(self.spin_jump_frames.value())))
            mk(Qt.Key_Period, lambda: self.seek_frames(int(self.spin_jump_frames.value())))
        except Exception: pass

        main_layout.addWidget(self.btn_toggle_sidebar)
        main_layout.addWidget(sidebar)

        # --- SPLIT VIEW ---
        self.splitter = QSplitter(Qt.Horizontal)
        
        cctv_cont = QWidget()
        cctv_l = QVBoxLayout(cctv_cont); cctv_l.setContentsMargins(0,0,0,0)
        self.cctv_scene = QGraphicsScene()
        self.cctv_view = CCTVGraphicsView(self.cctv_scene)
        self.cctv_view.setBackgroundBrush(QBrush(QColor(0,0,0)))
        self.cctv_view.setMinimumSize(400, 300)
        self.cctv_pixmap_item = QGraphicsPixmapItem()
        self.cctv_pixmap_item.setZValue(0)
        self.cctv_scene.addItem(self.cctv_pixmap_item)
        cctv_l.addWidget(self.cctv_view)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #0078d7; }")
        cctv_l.addWidget(self.progress_bar)
        self.splitter.addWidget(cctv_cont)

        self.sat_scene = QGraphicsScene()
        self.sat_view = SatGraphicsView(self.sat_scene)
        self.sat_view.parent_inspector = self
        self.sat_view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.sat_view.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.splitter.addWidget(self.sat_view)
        
        self.sat_count_label = QLabel("", self.sat_view.viewport())
        self.sat_count_label.setStyleSheet("color: white; font-weight: bold; font-size: 12px; background: rgba(0,0,0,80%);")
        self.sat_count_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.sat_count_label.hide()
        
        self.splitter.setStretchFactor(0, 1); self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([
            self.width() // 2,
            self.width() // 2
        ])

        main_layout.addWidget(self.splitter)

        self.sat_pixmap_item = None
        self.dynamic_group = QGraphicsItemGroup()
        self.dynamic_group.setZValue(100)
        self.sat_scene.addItem(self.dynamic_group)

    # --- LOGIC ---
    def load_file_list(self):
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        self.file_paths = []
        
        if not os.path.exists(OUTPUT_DIR):
            try:
                os.makedirs(OUTPUT_DIR)
            except: pass
        
        # Recursive walk to handle nested structure: output/model/config/location/file.json
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                # Accept both plain JSON and gzipped JSON (.json.gz)
                if not (file.endswith(".json") or file.endswith(".json.gz")): continue

                # We filter loosely to show all json results found in output
                full_path = os.path.join(root, file)
                self.file_paths.append(full_path)

        # Populate combo with 1-based numeric indices + readable relative path
        for i, full_path in enumerate(self.file_paths):
            rel_path = os.path.relpath(full_path, OUTPUT_DIR)
            display = f"{i+1}: {rel_path}"
            self.file_combo.addItem(display)

        if not self.file_paths:
            self.file_combo.addItem("(no files found in output/)")
        
        self.file_combo.blockSignals(False)

    def load_selected_file(self):
        # Debug: confirm this method is being invoked
        try:
            # print("[ROI] load_selected_file called; combo_index=", self.file_combo.currentIndex(), "file_paths_len=", len(self.file_paths), flush=True)
            try: self.lbl_info.setText("Loading selected file...")
            except: pass
        except Exception:
            pass
        idx = self.file_combo.currentIndex()
        if idx < 0 or not self.file_paths: return
        try:
            path = self.file_paths[idx]
            if path.endswith('.gz'):
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            self.current_json_data = data
            self.json_frame_map = {f["frame_index"]: f["objects"] for f in data.get("frames", [])}
            
            # Check 3D availability - look for 'have_measurements' or presence of 'bbox_3d'
            self.has_3d_data = False
            for f in data['frames'][:50]:
                for o in f['objects']:
                    if o.get('have_measurements', False) or (o.get('bbox_3d') and len(o['bbox_3d']) == 8):
                        self.has_3d_data = True
                        break
                if self.has_3d_data: break
            
            self.chk_3d_box.setEnabled(self.has_3d_data)
            self.chk_3d_box.setChecked(self.has_3d_data)
            
            self.setup_video(data)
            self.setup_sat_view(data)
            self.load_roi_mask(data)
            
            # Use mp4_frame_count if animation_frame_count absent
            max_frames = data.get("animation_frame_count", data.get("mp4_frame_count", 1))
            self.progress_bar.setRange(0, max_frames - 1)
            
            self.speed_display_cache = {}
            self.is_paused = True
            self.update_frame()
            # Ensure CCTV view is fitted to the video after the first frame is drawn
            try:
                self.fit_cctv_to_viewport()
            except Exception:
                pass
        except Exception as e:
            self.lbl_info.setText(f"Err: {e}")
            import traceback; traceback.print_exc()

    def setup_sat_view(self, data):
        self.sat_scene.clear()
        self.sat_pixmap_item = None
        self.fov_item = None
        self.dynamic_group = QGraphicsItemGroup()
        self.dynamic_group.setZValue(100)
        self.sat_scene.addItem(self.dynamic_group)
        self.svg_layer_groups = {}

        loc = data["location_code"]
        # Path: location/{loc}/G_projection_{loc}.json
        base = os.path.join("location", loc)
        g_path = os.path.join(base, f"G_projection_{loc}.json")
        
        if not os.path.exists(g_path):
            self.lbl_info.setText(f"G_proj missing: {g_path}")
            return

        with open(g_path, 'r') as f: self.g_data = json.load(f)
        
        # Parse new schema keys
        inputs = self.g_data.get('inputs', {})
        layout_svg = self.g_data.get('layout_svg', {})
        homography = self.g_data.get('homography', {})
        use_roi = self.g_data.get('use_roi', False)
        use_svg = self.g_data.get('use_svg', False)
        # persist flag so drawing logic can adapt when SVG is disabled
        self.sat_use_svg = bool(use_svg)

        # Configure ROI checkbox based on G_projection flag
        try:
            if use_roi:
                # Enable ROI control if the projection supports it, but do NOT
                # auto-check the box. Keep ROI off by default so users opt-in.
                try:
                    self.chk_roi.setEnabled(True)
                except:
                    pass
            else:
                # Disable and uncheck ROI if not supported
                try:
                    self.chk_roi.blockSignals(True)
                    self.chk_roi.setChecked(False)
                    self.chk_roi.setEnabled(False)
                    self.show_roi = False
                    # Hide any existing overlay immediately
                    if getattr(self, 'roi_overlay_item', None):
                        try: self.roi_overlay_item.setVisible(False)
                        except: pass
                finally:
                    try: self.chk_roi.blockSignals(False)
                    except: pass
        except Exception:
            pass

        # If SVG usage is disabled in the projection, bias opacity to SAT map
        try:
            if not use_svg:
                # Set slider to full SAT (100) so SVG groups are made transparent via update_sat_layers
                try:
                    self.slider_opacity.blockSignals(True)
                    self.slider_opacity.setValue(100)
                finally:
                    try: self.slider_opacity.blockSignals(False)
                    except: pass
                self.update_sat_layers()
        except Exception:
            pass
        
        # Affine matrix is now at layout_svg['A']
        affine_mat = layout_svg.get('A', [[1,0,0],[0,1,0]])

        sat_p = os.path.join(base, inputs.get('sat_path', ''))
        if os.path.exists(sat_p):
            self.sat_pixmap_item = QGraphicsPixmapItem(QPixmap(sat_p))
            self.sat_pixmap_item.setZValue(0)
            self.sat_scene.addItem(self.sat_pixmap_item)
            try:
                rect = self.sat_pixmap_item.boundingRect()
                pad = max(2000, int(max(rect.width(), rect.height()) * 0.5))
                scene_rect = QRectF(rect).adjusted(-pad, -pad, pad, pad)
                self.sat_scene.setSceneRect(scene_rect)
            except Exception: pass

        svg_file = inputs.get('layout_path')
        if svg_file:
            svg_p = os.path.join(base, svg_file)
            if os.path.exists(svg_p):
                parser = SvgGeometryParser(svg_p, affine_mat)
                z_order = {'Background': 1, 'Aesthetic': 2, 'Guidelines': 3, 'Physical': 4, 'Anchors': 5}
                for layer_name, items in parser.layer_items.items():
                    group = QGraphicsItemGroup()
                    group.setZValue(z_order.get(layer_name, 1))
                    for item in items: group.addToGroup(item)
                    self.sat_scene.addItem(group)
                    self.svg_layer_groups[layer_name] = group
                    chk = self.layer_checks.get(layer_name)
                    visible = chk.isChecked() if (chk is not None) else True
                    group.setVisible(visible)
        
        # FOV Polygon (now in homography['fov_polygon'])
        try:
            fov_pts = homography.get('fov_polygon')
            if fov_pts and isinstance(fov_pts, list) and len(fov_pts) >= 3:
                poly = QGraphicsPolygonItem(QPolygonF([QPointF(float(p[0]), float(p[1])) for p in fov_pts]))
                pen = QPen(QColor(0, 200, 0), 2)
                alpha = int((self.slider_fov_opacity.value()) * 255 / 100)
                brush = QBrush(QColor(0, 200, 0, alpha))
                poly.setPen(pen); poly.setBrush(brush)
                poly.setZValue(150)
                poly.setVisible(self.show_fov)
                self.sat_scene.addItem(poly)
                self.fov_item = poly
                
                # Camera Marker from 'parallax' block
                parallax = self.g_data.get('parallax', {})
                cx = parallax.get('x_cam_coords_sat')
                cy = parallax.get('y_cam_coords_sat')
                cz = parallax.get('z_cam_meters')
                
                if cx is not None and cy is not None:
                    cx, cy = float(cx), float(cy)
                    outer_r, inner_r = 14, 6
                    star_pts = []
                    for i in range(10):
                        angle = math.pi / 5.0 * i - math.pi/2
                        r = outer_r if (i % 2 == 0) else inner_r
                        star_pts.append(QPointF(cx + r * math.cos(angle), cy + r * math.sin(angle)))
                    
                    star_poly = QGraphicsPolygonItem(QPolygonF(star_pts))
                    star_poly.setPen(QPen(QColor(255, 215, 0), 2))
                    star_poly.setBrush(QBrush(QColor(200, 0, 0)))
                    star_poly.setZValue(200)
                    star_poly.setVisible(self.show_fov)
                    self.sat_scene.addItem(star_poly)
                    self.camera_marker_item = star_poly
                    
                    if cz is not None:
                        txt = QGraphicsSimpleTextItem(f"{loc}\n{float(cz):.2f}m")
                        fnt = txt.font(); fnt.setPointSize(10); txt.setFont(fnt)
                        txt.setBrush(QBrush(Qt.red))
                        br = txt.boundingRect()
                        txt.setPos(cx - br.width()/2.0, cy + outer_r + 4)
                        txt.setZValue(200)
                        txt.setVisible(self.show_fov)
                        self.sat_scene.addItem(txt)
                        self.camera_marker_text = txt
        except Exception: 
            self.fov_item = None

        self.update_sat_layers()

    def show_file_list_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Available Files")
        dlg.setModal(True)
        dlg.resize(700, 420)

        v = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        lines = []
        for i, full_path in enumerate(self.file_paths):
            rel = os.path.relpath(full_path, OUTPUT_DIR)
            lines.append(f"{i+1}: {rel} \n")
        te.setPlainText("\n".join(lines) if lines else "(no files found in output/)")
        v.addWidget(te)

        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        h = QHBoxLayout()
        h.addStretch()
        h.addWidget(btn)
        v.addLayout(h)

        dlg.exec_()

    def update_sat_layers(self):
        alpha = self.slider_opacity.value() / 100.0
        if getattr(self, 'sat_pixmap_item', None) is not None:
            try: self.sat_pixmap_item.setOpacity(alpha)
            except RuntimeError: self.sat_pixmap_item = None
        svg_alpha = 1.0 - alpha
        for group in self.svg_layer_groups.values(): group.setOpacity(svg_alpha)

    def toggle_svg_layer(self, name, checked):
        if name in self.svg_layer_groups: self.svg_layer_groups[name].setVisible(checked)

    def set_fov_visible(self, visible):
        self.show_fov = bool(visible)
        for item_name in ['fov_item', 'camera_marker_item', 'camera_marker_text']:
            item = getattr(self, item_name, None)
            if item:
                try: item.setVisible(self.show_fov)
                except: pass
        
        # Sync UI without signal loop
        if hasattr(self, 'chk_fov'):
            self.chk_fov.blockSignals(True)
            self.chk_fov.setChecked(self.show_fov)
            self.chk_fov.blockSignals(False)

    def update_fov_opacity(self):
        try:
            alpha = int(self.slider_fov_opacity.value() * 255 / 100)
            if getattr(self, 'fov_item', None) is not None:
                brush = self.fov_item.brush()
                col = brush.color()
                col.setAlpha(alpha)
                brush.setColor(col)
                self.fov_item.setBrush(brush)
        except: pass

    def load_roi_mask(self, data):
        # print("[ROI] load_roi_mask invoked", flush=True)
        # Clean up existing overlay if any
        if self.roi_overlay_item:
            try: self.cctv_scene.removeItem(self.roi_overlay_item)
            except: pass
            self.roi_overlay_item = None

        # Clear any existing mask
        self.roi_mask = None

        loc = data.get("location_code", "")
        roi_filename = f"roi_{loc}.png"
        p = os.path.join("location", loc, roi_filename)

        # Report path & existence
        try:
            # print(f"[ROI] looking for ROI file: {p} (exists={os.path.exists(p)})", flush=True)
            self.lbl_info.setText(f"Checking ROI: {os.path.basename(p)}")
        except Exception:
            pass

        if os.path.exists(p):
            try:
                # Load as grayscale mask
                mask_cv = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if mask_cv is None:
                    # print(f"[ROI] cv2.imread returned None for {p}", flush=True)
                    try: self.lbl_info.setText(f"ROI read failed: {os.path.basename(p)}")
                    except: pass
                    return

                # Match resolution to CCTV frames (be defensive about resolution shape)
                try:
                    res = data.get("meta", {}).get("resolution")
                    if res is None:
                        target_w, target_h = mask_cv.shape[1], mask_cv.shape[0]
                    else:
                        # Accept list/tuple/ndarray; take first two values if longer
                        if isinstance(res, (list, tuple)):
                            if len(res) >= 2:
                                target_w, target_h = int(res[0]), int(res[1])
                            else:
                                target_w, target_h = mask_cv.shape[1], mask_cv.shape[0]
                        else:
                            try:
                                # Try to iterate (e.g., numpy array)
                                rlist = list(res)
                                if len(rlist) >= 2:
                                    target_w, target_h = int(rlist[0]), int(rlist[1])
                                else:
                                    target_w, target_h = mask_cv.shape[1], mask_cv.shape[0]
                            except Exception:
                                target_w, target_h = mask_cv.shape[1], mask_cv.shape[0]
                except Exception:
                    target_w, target_h = mask_cv.shape[1], mask_cv.shape[0]

                if (mask_cv.shape[1], mask_cv.shape[0]) != (target_w, target_h):
                    # print(f"[ROI] resizing mask from {(mask_cv.shape[1], mask_cv.shape[0])} to {(target_w, target_h)}", flush=True)
                    mask_cv = cv2.resize(mask_cv, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

                # Save mask for per-frame blending in draw_cctv
                self.roi_mask = mask_cv

                # Also create a scene overlay pixmap (optional) so users can toggle it
                # Ensure mask is 2D (some ROI PNGs may have channels)
                if mask_cv.ndim == 3:
                    try:
                        if mask_cv.shape[2] == 1:
                            mask_cv = mask_cv[:, :, 0]
                        else:
                            mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_BGR2GRAY)
                    except Exception:
                        mask_cv = mask_cv[:, :, 0]

                # Use shape[:2] to guard against any remaining extra dims
                h, w = mask_cv.shape[:2]
                overlay_arr = np.zeros((h, w, 4), dtype=np.uint8)
                is_black = (mask_cv < 10)
                # Assign per-pixel RGBA safely
                overlay_arr[is_black, :] = (0, 0, 255, 100)

                # Create QImage using bytesPerLine (stride) to avoid layout issues
                try:
                    bytespp = overlay_arr.strides[0]
                    qimg = QImage(overlay_arr.data, w, h, bytespp, QImage.Format_ARGB32).copy()
                except Exception:
                    qimg = QImage(overlay_arr.data, w, h, QImage.Format_ARGB32).copy()
                pix = QPixmap.fromImage(qimg)

                self.roi_overlay_item = QGraphicsPixmapItem(pix)
                self.roi_overlay_item.setZValue(1) # Above video (0), below boxes
                # Parent the overlay to the CCTV pixmap so transforms/positioning stay in sync
                try:
                    if getattr(self, 'cctv_pixmap_item', None) is not None:
                        self.roi_overlay_item.setParentItem(self.cctv_pixmap_item)
                    else:
                        self.cctv_scene.addItem(self.roi_overlay_item)
                except Exception:
                    try: self.cctv_scene.addItem(self.roi_overlay_item)
                    except: pass

                # Ensure visibility matches checkbox
                try: self.roi_overlay_item.setVisible(self.show_roi)
                except: pass

                # Debug: compute mask stats and show them for troubleshooting
                try:
                    count_black = int(np.count_nonzero(mask_cv < 10))
                    total = int(mask_cv.size)
                    pct = 100.0 * count_black / total if total else 0.0
                    mn, mx = int(mask_cv.min()), int(mask_cv.max())
                    msg = f"ROI loaded: {os.path.basename(p)} {w}x{h} black={count_black}/{total} ({pct:.1f}%) min={mn} max={mx}"
                    # print("[ROI]", msg, flush=True)
                    try: self.lbl_info.setText(msg)
                    except: pass
                except Exception as e:
                    print(f"[ROI] stats error: {e}", flush=True)
            except Exception as e:
                import traceback
                # print(f"[ROI] exception loading ROI: {e}", flush=True)
                traceback.print_exc()
                try: self.lbl_info.setText(f"ROI load error: {e}")
                except: pass
            # Ensure we refresh one frame so the overlay/blend is visible immediately
            try:
                if self.cap:
                    self.update_frame(False)
            except Exception:
                pass
        else:
            # ROI file missing — display debug message
            msg = f"ROI not found: {p}"
            # print("[ROI]", msg, flush=True)
            try: self.lbl_info.setText(msg)
            except: pass

    def setup_video(self, data):
        # Handle path normalization
        raw_path = data.get("mp4_path", "")
        path = os.path.normpath(raw_path)
        
        if not os.path.exists(path):
            self.lbl_info.setText(f"Video not found: {path}")
            return
        
        self.cap = cv2.VideoCapture(path)
        self.target_fps = data["meta"].get("fps", 30)
        self.slider_fps.setValue(int(self.target_fps))
        self.current_frame_idx = 0

    # --- UPDATE UI ---
    def update_text_controls(self):
        self.text_color_mode = self.combo_text_color.currentText()
        self.update_ui_state()

    def update_ui_state(self):
        self.show_tracking = self.chk_tracking.isChecked()
        self.sat_box_thick = self.slider_sat_thick.value()
        self.show_sat_label = self.chk_sat_label.isChecked()
        self.sat_label_size = self.slider_sat_text.value()
        self.show_sat_box = self.chk_sat_box.isChecked()
        # --- NEW: Update State ---
        self.show_sat_coords_dot = self.chk_sat_coords.isChecked()
        self.show_sat_arrow = self.chk_sat_arrow.isChecked()
        self.speed_update_delay_frames = self.slider_speed_delay.value()
        
        self.show_3d = self.chk_3d_box.isChecked()
        self.face_opacity = self.slider_3d_alpha.value()
        self.show_label = self.chk_cctv_label.isChecked()
        self.show_roi = self.chk_roi.isChecked()
        
        # Toggle ROI overlay visibility
        if self.roi_overlay_item:
            self.roi_overlay_item.setVisible(self.show_roi)
        
        # Force a refresh so ROI visibility change is visible immediately
        try:
            if self.cap:
                self.update_frame(False)
        except: pass

    def update_fps_target(self):
        self.target_fps = self.slider_fps.value()

    def seek_frames(self, offset):
        if not self.current_json_data: return
        max_f = self.current_json_data.get("mp4_frame_count", 0)
        new_idx = int(self.current_frame_idx + offset)
        new_idx = max(0, min(new_idx, max_f - 1))
        
        self.current_frame_idx = new_idx
        if self.cap:
            try: self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            except: pass
        self.update_frame(False)

    # --- LOOP ---
    def adaptive_loop(self):
        if self.is_paused: return
        t = time.time()
        self.update_frame(True)
        wait = int(max(1, (1.0/self.target_fps - (time.time()-t))*1000))
        QTimer.singleShot(wait, self.adaptive_loop)

    def update_frame(self, advance=False):
        if not self.cap or not self.cap.isOpened(): return
        # Prefer animation count, fall back to mp4 count
        max_f = self.current_json_data.get("animation_frame_count", 
                                           self.current_json_data.get("mp4_frame_count", 0))
        
        if self.current_frame_idx >= max_f:
            self.current_frame_idx = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        if not advance: self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret: return

        objs = self.json_frame_map.get(self.current_frame_idx, [])
        self.draw_cctv(frame.copy(), objs)
        self.draw_sat(objs)
        self.progress_bar.setValue(self.current_frame_idx)
        
        if advance:
            self.current_frame_idx += 1
            t = time.time(); dt = t - self.last_real_time; self.last_real_time = t
            if dt > 0: self.actual_fps = 0.9 * self.actual_fps + 0.1*(1.0/dt)
        
        self.lbl_info.setText(f"Frame: {self.current_frame_idx}/{max_f} | FPS: {self.actual_fps:.1f}")

    # --- DRAWING ---
    def draw_cctv(self, frame, objects):
        # If we have a loaded ROI mask and the user wants to see it,
        # blend the mask into the current frame so it is always visible
        try:
            # If we have a loaded ROI mask and the user wants to see it,
            # prefer the QGraphics overlay (created in load_roi_mask) when available
            # — this avoids expensive per-frame CPU blending which causes lag.
            if getattr(self, 'roi_mask', None) is not None and self.show_roi:
                # If we created a scene overlay item, let Qt draw it (it will scale/position)
                if getattr(self, 'roi_overlay_item', None) is not None:
                    # No per-frame blending required when overlay exists
                    pass
                else:
                    # Fallback: perform a cheap, cached resize and in-place masked blend
                    # Cache a resized mask so we don't call cv2.resize every frame
                    if getattr(self, 'roi_mask_resized', None) is None or self.roi_mask_resized.shape[:2] != frame.shape[:2]:
                        try:
                            self.roi_mask_resized = cv2.resize(self.roi_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        except Exception:
                            self.roi_mask_resized = self.roi_mask

                    mask = self.roi_mask_resized
                    # Create boolean index for masked pixels (outside ROI)
                    try:
                        mask_bool = (mask < 10)
                        if mask_bool.any():
                            # Blend red onto masked pixels in-place using a simple alpha blend
                            alpha = 0.4
                            # Convert selected pixels to float, blend, cast back to uint8
                            fb = frame[mask_bool].astype(np.float32)
                            fb[:] = fb * (1.0 - alpha) + np.array([0.0, 0.0, 255.0], dtype=np.float32) * alpha
                            frame[mask_bool] = fb.astype(np.uint8)
                    except Exception:
                        pass
        except Exception:
            pass

        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch*w, QImage.Format_BGR888)
        pix = QPixmap.fromImage(qt_img)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.Antialiasing)

        for obj in objects:
            cls = obj.get("class", "?"); tid = obj.get("tracked_id")
            seed = f"{cls}_{tid}" if (self.show_tracking and tid is not None) else cls
            col = get_color_from_string(seed)
            lbl = f"{tid} {cls}" if tid is not None else cls
            
            bbox_3d = obj.get('bbox_3d')
            have_heading = obj.get('have_heading', False)
            have_measurements = obj.get('have_measurements', False)
            
            # 3D MODE: Only if flag is true AND we have box data
            # Logic: "have_heading=T, have_measurements=T -> have 3D box"
            can_draw_3d = self.show_3d and have_heading and have_measurements and bbox_3d and len(bbox_3d) == 8

            if can_draw_3d:
                try:
                    pts = [QPointF(p[0], p[1]) for p in bbox_3d]
                    faces = [
                        [0,1,2,3], [4,5,6,7], # Bot, Top
                        [0,1,5,4], [1,2,6,5], 
                        [2,3,7,6], [3,0,4,7] 
                    ]
                    pen = QPen(col, self.box_thickness)
                    painter.setPen(pen)
                    fill = QColor(col); fill.setAlpha(self.face_opacity)
                    painter.setBrush(QBrush(fill))
                    
                    for f_idx in faces:
                        poly = QPolygonF([pts[i] for i in f_idx])
                        painter.drawPolygon(poly)
                except: pass
            
            # 2D MODE: If not showing 3D, OR if 3D unavailable
            elif not self.show_3d:
                bbox = obj.get("bbox_2d")
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    rect = QRectF(x1, y1, x2-x1, y2-y1)
                    painter.setPen(QPen(col, self.box_thickness))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(rect)
                    
                    # Logic: "have_heading=F, have_measurements=T -> reference_point displayed in CCTV (only in 2D box mode)"
                    if (not have_heading) and have_measurements:
                        ref_pt = obj.get("reference_point")
                        if ref_pt:
                            rx, ry = ref_pt
                            painter.setBrush(QBrush(col))
                            painter.drawEllipse(QPointF(rx, ry), 4, 4)

                    if self.show_label:
                        painter.setPen(QPen(Qt.white))
                        fm = painter.fontMetrics()
                        tw, th = fm.width(lbl), fm.height()
                        painter.fillRect(QRectF(x1, y1-th, tw+4, th), col)
                        painter.drawText(QPointF(x1+2, y1-2), lbl)

        painter.end()
        if getattr(self, 'cctv_pixmap_item', None) is not None:
            self.cctv_pixmap_item.setPixmap(pix)

    def draw_sat(self, objects):
        for item in self.dynamic_group.childItems():
            self.dynamic_group.removeFromGroup(item); self.sat_scene.removeItem(item)
            
        # Update overlay counter
        try:
            cnt = len(objects) if objects is not None else 0
            if hasattr(self, 'sat_count_label'):
                self.sat_count_label.setText(f"Objects: {cnt}")
                self.sat_count_label.adjustSize()
                vp = self.sat_view.viewport()
                if vp: self.sat_count_label.move(max(0, vp.width() - self.sat_count_label.width() - 8), 8)
                self.sat_count_label.show()
        except: pass

        for obj in objects:
            cls = obj.get("class", "?"); tid = obj.get("tracked_id")
            seed = f"{cls}_{tid}" if (self.show_tracking and tid is not None) else cls
            col = get_color_from_string(seed)
            pen = QPen(col, self.sat_box_thick)
            brush = QBrush(QColor(col.red(), col.green(), col.blue(), 100))
            
            have_heading = obj.get('have_heading', False)
            have_measurements = obj.get('have_measurements', False)
            coord = obj.get("sat_coords") or obj.get("sat_coord") # Try plural, fallback to singular
            pts = obj.get("sat_floor_box")

            # --- 1. Floor Box (T/T only) ---
            if self.show_sat_box and have_heading and have_measurements:
                pts = obj.get("sat_floor_box")
                if pts and len(pts) >= 3:
                    poly = QGraphicsPolygonItem(QPolygonF([QPointF(p[0], p[1]) for p in pts]))
                    poly.setPen(pen); poly.setBrush(brush)
                    self.dynamic_group.addToGroup(poly)

            # --- 2. Arrow (T only) ---
            # Draw heading arrow only when object truly has a heading and it's not a default
            default_heading = obj.get('default_heading', False)
            if self.show_sat_arrow and have_heading and (not default_heading) and coord and pts and len(pts) >= 3:
                heading = obj.get("heading")
                if heading is not None:
                    rad = math.radians(heading)
                    p1 = QPointF(coord[0], coord[1])
                    p2 = QPointF(coord[0] + 40*math.cos(rad), coord[1] + 40*math.sin(rad))
                    line = QGraphicsLineItem(p1.x(), p1.y(), p2.x(), p2.y())
                    line.setPen(QPen(Qt.yellow, 2))
                    self.dynamic_group.addToGroup(line)

            # --- NEW FEATURE: Sat Coordinates Dot (Activated by Key 6) ---
            # Show SAT coordinate dot. Also allow showing when SVG is disabled and
            # 3D box mode is OFF so users can still see positions even without
            # a floor-box present.
            if self.show_sat_coords_dot and coord and ((pts and len(pts) >= 3) or (not getattr(self, 'sat_use_svg', True) and not self.show_3d)):
                # Calculate relative size
                radius = 4.0 # Default fallback
                pts = obj.get("sat_floor_box")
                if pts and len(pts) >= 3:
                    # Calculate bounding box of the floor polygon
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    w = max(xs) - min(xs)
                    h = max(ys) - min(ys)
                    # Use ~15% of the average dimension
                    avg_dim = (w + h) / 2.0
                    radius = max(3.0, avg_dim * 0.15)
                
                # Draw the dot
                dot = QGraphicsEllipseItem(coord[0]-radius, coord[1]-radius, radius*2, radius*2)
                dot.setBrush(QBrush(col))
                # Optional: Add a thin stroke to make it pop against the map
                dot.setPen(QPen(Qt.black, 1)) 
                self.dynamic_group.addToGroup(dot)

            # --- 3. Old Sat Coordinates Dot (Conditional Fallback) ---
            # Kept for backward compatibility if you still want dots for items without heading 
            # when the new toggle is OFF.
            # Legacy fallback dot: keep showing when SVG is disabled and 3D is off
            # to preserve previous behavior for non-SVG projections.
            elif (not have_heading) and have_measurements and (not self.show_3d) and coord and ((pts and len(pts) >= 3) or (not getattr(self, 'sat_use_svg', True) and not self.show_3d)):
                dot = QGraphicsEllipseItem(coord[0]-3, coord[1]-3, 6, 6)
                dot.setBrush(QBrush(col))
                self.dynamic_group.addToGroup(dot)

            # --- 4. Text Label (Universal) ---
            # Condition: Always show if coord exists AND toggle is ON
            # Text labels: show when SVG disabled and 3D off as well
            if self.show_sat_label and coord and ((pts and len(pts) >= 3) or (not getattr(self, 'sat_use_svg', True) and not self.show_3d)):
                raw_s = obj.get("speed_kmh", 0)
                disp_s = raw_s
                # Speed smoothing
                if tid is not None:
                    cache = self.speed_display_cache.get(tid, {'val': raw_s, 'last_frame': -999})
                    if (self.current_frame_idx - cache['last_frame']) >= self.speed_update_delay_frames or \
                       (self.current_frame_idx < cache['last_frame']):
                        cache['val'] = raw_s; cache['last_frame'] = self.current_frame_idx
                    self.speed_display_cache[tid] = cache
                    disp_s = cache['val']
                
                label_str = f"{cls} {disp_s:.1f}km/h"
                txt = QGraphicsSimpleTextItem(label_str)
                txt.setPos(coord[0], coord[1])
                
                if self.text_color_mode == "Black": c = Qt.black
                elif self.text_color_mode == "Yellow": c = QColor(255, 255, 143)
                else: c = Qt.white
                
                txt.setBrush(QBrush(c))
                f = txt.font(); f.setPointSize(self.sat_label_size); txt.setFont(f)
                self.dynamic_group.addToGroup(txt)

    def fit_cctv_to_viewport(self):
        try:
            if getattr(self, 'cctv_view', None) is None or getattr(self, 'cctv_pixmap_item', None) is None: return
            self.cctv_view.setTransform(QTransform())
            rect = self.cctv_pixmap_item.boundingRect()
            if not rect.isNull():
                self.cctv_scene.setSceneRect(rect)
                self.cctv_view.fitInView(rect, Qt.KeepAspectRatio)
        except: pass

    def toggle_view_layout(self):
        try:
            if not hasattr(self, 'splitter') or self.splitter is None: return
            cur = self.splitter.orientation()
            self.splitter.setOrientation(Qt.Vertical if cur == Qt.Horizontal else Qt.Horizontal)
            self.splitter.setStretchFactor(0, 1); self.splitter.setStretchFactor(1, 1)
        except: pass

    def toggle_fov(self):
        self.set_fov_visible(not self.show_fov)

    def keyPressEvent(self, e):
        ch = e.text()
        if ch == ';': self.seek_frames(-1)
        elif ch == "'": self.seek_frames(1)
        elif ch == '<': self.seek_frames(-int(self.spin_jump_frames.value()))
        elif ch == '>': self.seek_frames(int(self.spin_jump_frames.value()))
        elif e.key() == Qt.Key_Space: self.toggle_playback()
        elif e.key() == Qt.Key_R: self.current_frame_idx = 0; self.update_frame(False)
        elif e.key() == Qt.Key_BracketLeft:
            idx = self.file_combo.currentIndex()
            if idx > 0: self.file_combo.setCurrentIndex(idx - 1)
        elif e.key() == Qt.Key_BracketRight:
            idx = self.file_combo.currentIndex()
            if idx < self.file_combo.count() - 1: self.file_combo.setCurrentIndex(idx + 1)
        super().keyPressEvent(e)

    def toggle_playback(self):
        self.is_paused = not self.is_paused
        if not self.is_paused: self.last_real_time = time.time(); self.adaptive_loop()