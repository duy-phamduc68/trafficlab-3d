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
    QListWidget, QListWidgetItem, QSplitter, QMessageBox, 
    QLineEdit, QFormLayout, QAbstractItemView, QGraphicsView
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


class HomAStage(QWidget):
    def __init__(self, project_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.project_root = project_root
        
        self.anchors = []  
        self._current_cctv_cv = None 
        self._current_sat_cv = None
        self._loaded_loc = None # Track what is currently loaded in UI
        
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("background-color: #2b2b2b;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(10)

        lbl_title = QLabel("Homography Anchors")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #fff;")
        side_layout.addWidget(lbl_title)
        
        inst = QLabel(
            "Controls:\n"
            "• Left Drag: Pan Image\n"
            "• Wheel: Zoom\n"
            "• Right Click: Place Point"
        )
        inst.setStyleSheet("color: #aaa; font-size: 11px;")
        side_layout.addWidget(inst)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        side_layout.addWidget(self.list_widget)

        edit_group = QWidget()
        form_layout = QFormLayout(edit_group)
        form_layout.setContentsMargins(0,0,0,0)
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. Fountain")
        self.name_input.textChanged.connect(self._on_name_changed)
        
        lbl_name = QLabel("Name:")
        lbl_name.setStyleSheet("color: #ccc;")
        form_layout.addRow(lbl_name, self.name_input)
        side_layout.addWidget(edit_group)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add Pair")
        self.btn_add.clicked.connect(self._on_add_pair)
        self.btn_del = QPushButton("Remove")
        self.btn_del.clicked.connect(self._on_remove_pair)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_del)
        side_layout.addLayout(btn_layout)

        side_layout.addStretch()

        self.btn_compute = QPushButton("Compute Homography")
        self.btn_compute.setStyleSheet("background-color: #2a84ff; color: white; font-weight: bold; padding: 6px;")
        self.btn_compute.clicked.connect(self._on_compute)
        side_layout.addWidget(self.btn_compute)

        self.btn_proceed = QPushButton("Proceed")
        self.btn_proceed.clicked.connect(self._on_proceed)
        side_layout.addWidget(self.btn_proceed)

        main_layout.addWidget(sidebar)

        splitter = QSplitter(Qt.Horizontal)
        
        left_cont = QWidget()
        l_vbox = QVBoxLayout(left_cont)
        l_vbox.setContentsMargins(0,0,0,0)
        l_vbox.addWidget(QLabel("Undistorted CCTV (Right-click to place)"))
        self.view_cctv = RightClickImageViewer()
        self.view_cctv.clicked.connect(self._on_cctv_right_clicked)
        l_vbox.addWidget(self.view_cctv)
        
        right_cont = QWidget()
        r_vbox = QVBoxLayout(right_cont)
        r_vbox.setContentsMargins(0,0,0,0)
        r_vbox.addWidget(QLabel("Satellite / Layout (Right-click to place)"))
        self.view_sat = RightClickImageViewer()
        self.view_sat.clicked.connect(self._on_sat_right_clicked)
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
        if not host: return
        obj = getattr(host, 'inspect_obj', None)
        if not obj: return

        loc_code = obj.get('meta', {}).get('location_code')
        
        # FIX: Check if we are already loaded for this location
        if self._loaded_loc == loc_code:
            return # Don't reload/clear UI if it's the same session

        self._loaded_loc = loc_code
        self._load_images(obj)
        
        self.anchors = []
        self.list_widget.clear()
        
        hom = obj.get('homography', {})
        saved_list = hom.get('anchors_list', [])
        
        for item in saved_list:
            c_list = item.get('coords_cctv')
            s_list = item.get('coords_sat')
            c_tup = tuple(c_list) if (c_list and len(c_list)==2) else None
            s_tup = tuple(s_list) if (s_list and len(s_list)==2) else None
            
            self._add_anchor_internal(
                name=item.get('name', ''), 
                cctv_pt=c_tup, 
                sat_pt=s_tup
            )

        if self.anchors:
            self.list_widget.setCurrentRow(0)
        else:
            self._on_add_pair()

        self._refresh_markers()

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
                K_list = und.get('K')
                D_list = und.get('D', [0]*5)
                h, w = src.shape[:2]
                
                K = np.array(K_list, dtype=np.float64) if (K_list and len(K_list)==3) else \
                    np.array([[max(w,h), 0, w/2], [0, max(w,h), h/2], [0, 0, 1]], dtype=np.float64)
                D = np.array(D_list, dtype=np.float64)

                try:
                    # FIX: Use K.copy() instead of getOptimalNewCameraMatrix to match reference code logic
                    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
                    newcameramtx = K.copy()
                    
                    self._current_cctv_cv = remap_with_supersample(src, K, D, newcameramtx)
                    qimg = self._cv_to_qimage(self._current_cctv_cv)
                    self.view_cctv.load_pixmap(QPixmap.fromImage(qimg))
                    self.view_cctv.fitToView()
                except Exception as e:
                    print(f"Undistort Error: {e}")

        # Satellite
        sat_path = os.path.join(proj_root, 'location', loc_code, f'sat_{loc_code}.png')
        if os.path.isfile(sat_path):
            self._current_sat_cv = cv2.imread(sat_path)
            if self._current_sat_cv is not None:
                qimg = self._cv_to_qimage(self._current_sat_cv)
                self.view_sat.load_pixmap(QPixmap.fromImage(qimg))
                self.view_sat.fitToView()

    def _add_anchor_internal(self, name="", cctv_pt=None, sat_pt=None):
        idx = len(self.anchors)
        if not name:
            name = f"Pair {idx}"
            
        data = {
            'id': idx, 
            'name': name,
            'cctv': cctv_pt,
            'sat': sat_pt
        }
        self.anchors.append(data)
        
        item = QListWidgetItem(f"{data['id']}: {data['name']}")
        self.list_widget.addItem(item)
        return idx

    def _on_add_pair(self):
        idx = self._add_anchor_internal()
        self.list_widget.setCurrentRow(idx)
        self.name_input.setFocus()
        self.name_input.selectAll()
        self._refresh_markers()

    def _on_remove_pair(self):
        row = self.list_widget.currentRow()
        if row < 0: return
        self.list_widget.takeItem(row)
        self.anchors.pop(row)
        for i, data in enumerate(self.anchors):
            data['id'] = i
            self.list_widget.item(i).setText(f"{i}: {data['name']}")
        
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(max(0, row-1))
        self._refresh_markers()

    def _on_row_changed(self, row):
        if row < 0 or row >= len(self.anchors):
            self.name_input.setEnabled(False)
            self.name_input.setText("")
            return
        data = self.anchors[row]
        self.name_input.setEnabled(True)
        self.name_input.blockSignals(True)
        self.name_input.setText(data['name'])
        self.name_input.blockSignals(False)
        self._refresh_markers()

    def _on_name_changed(self, text):
        row = self.list_widget.currentRow()
        if row < 0: return
        self.anchors[row]['name'] = text
        self.list_widget.item(row).setText(f"{self.anchors[row]['id']}: {text}")

    def _on_cctv_right_clicked(self, x, y):
        row = self.list_widget.currentRow()
        if row < 0: return
        self.anchors[row]['cctv'] = (x, y)
        self._refresh_markers()

    def _on_sat_right_clicked(self, x, y):
        row = self.list_widget.currentRow()
        if row < 0: return
        self.anchors[row]['sat'] = (x, y)
        self._refresh_markers()

    def _refresh_markers(self):
        self._clear_scene_markers(self.view_cctv)
        self._clear_scene_markers(self.view_sat)

        sel_row = self.list_widget.currentRow()

        for i, data in enumerate(self.anchors):
            is_sel = (i == sel_row)
            color = QColor(0, 255, 0) if is_sel else QColor(255, 0, 0)
            
            if data['cctv']:
                self._draw_marker(self.view_cctv, data['cctv'][0], data['cctv'][1], str(data['id']), color)
            if data['sat']:
                self._draw_marker(self.view_sat, data['sat'][0], data['sat'][1], str(data['id']), color)

    def _clear_scene_markers(self, viewer):
        from PyQt5.QtWidgets import QGraphicsPixmapItem
        scene = viewer.scene()
        for item in scene.items():
            if not isinstance(item, QGraphicsPixmapItem):
                scene.removeItem(item)

    def _draw_marker(self, viewer, x, y, label, color):
        from PyQt5.QtWidgets import QGraphicsEllipseItem, QGraphicsSimpleTextItem
        scene = viewer.scene()
        rad = 4
        el = scene.addEllipse(x - rad, y - rad, rad*2, rad*2)
        el.setPen(QPen(color, 1))
        el.setBrush(QBrush(color))
        txt = scene.addSimpleText(label)
        txt.setBrush(QBrush(color))
        txt.setPos(x + 5, y - 10)
        txt.setFont(QFont("Arial", 10, QFont.Bold))

    def _on_compute(self):
        if not HAS_CV2: return
        
        valid_anchors = [a for a in self.anchors if a['cctv'] is not None and a['sat'] is not None]

        if len(valid_anchors) < 4:
            QMessageBox.critical(self, "Critical Error", "Not enough anchors. Please select at least 4 pairs.")
            return

        try:
            src_pts = np.array([p['cctv'] for p in valid_anchors], dtype=np.float32).reshape(-1, 1, 2)
            dst_pts = np.array([p['sat'] for p in valid_anchors], dtype=np.float32).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            report_lines = []
            if H is not None:
                matches_mask = mask.ravel().tolist()
                inliers = matches_mask.count(1)
                total = len(valid_anchors)
                
                report_lines.append(f"RANSAC Inliers: {inliers} / {total}")
                
                msg_icon = QMessageBox.Information
                msg_title = "Homography Report"
                
                if inliers < 4:
                    report_lines.append("⚠️ WARNING: RANSAC found fewer than 4 good points.\nYour projection may be unstable.")
                    msg_icon = QMessageBox.Warning
                elif inliers < total:
                    report_lines.append(f"ℹ️ Note: {total - inliers} points were ignored as outliers (likely imprecise clicks).")
                else:
                    report_lines.append("✅ Excellent: All points fit the model.")

                host = getattr(self, 'host_tab', None) or self.parent()
                if host and getattr(host, 'inspect_obj', None):
                    obj = host.inspect_obj
                    hom = obj.setdefault('homography', {})
                    hom['H'] = H.tolist()
                    
                    json_list = []
                    for data in self.anchors:
                        item = {
                            "id": data['id'],
                            "name": data['name'],
                            "coords_cctv": list(data['cctv']) if data['cctv'] else None,
                            "coords_sat": list(data['sat']) if data['sat'] else None
                        }
                        json_list.append(item)
                    hom['anchors_list'] = json_list
                    
                    host.inspect_obj = obj
                
                QMessageBox.msgbox = QMessageBox(msg_icon, msg_title, "\n".join(report_lines), QMessageBox.Ok, self)
                QMessageBox.msgbox.show()

            else:
                QMessageBox.critical(self, "Calculation Failed", "❌ CRITICAL: Homography calculation failed.\nPoints might be collinear.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_proceed(self):
        host = getattr(self, 'host_tab', None) or self.parent()
        if host and hasattr(host, 'current_step_index'):
            try:
                host.current_step_index = 5
                host._show_stage(5)
                host._update_progress_to_index(5)
            except: pass

    def _cv_to_qimage(self, cv_bgr):
        if cv_bgr is None: return None
        rgb = cv_bgr[:, :, ::-1]
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        return qimg.copy()