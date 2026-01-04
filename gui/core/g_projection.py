import numpy as np
import cv2
import math
import os
import re
import xml.etree.ElementTree as ET

class SVGParser:
    """
    Parses SVG lines/polygons for orientation guidelines.
    ROBUST VERSION: Ignores namespaces and finds deeply nested IDs.
    """
    def __init__(self, svg_path, alignment_matrix_A=None):
        self.svg_path = svg_path
        self.orientation_segments = []
        self.M_align = np.identity(3)
        if alignment_matrix_A is not None:
            self.M_align[:2, :] = np.array(alignment_matrix_A)
            
        self.valid = False
        if os.path.exists(svg_path):
            try:
                self.tree = ET.parse(svg_path)
                self.root = self.tree.getroot()
                self.orientation_segments = self._extract_segments()
                self.valid = True
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
        target_ids = ['Guidelines', 'Physical']
        
        def get_tag(el):
            return el.tag.split('}')[-1]

        target_nodes = []
        for el in self.root.iter():
            if get_tag(el) == 'g' and el.get('id') in target_ids:
                target_nodes.append(el)

        for g in target_nodes:
            self._process_node(g, np.identity(3), segs)
        return segs

    def _process_node(self, element, parent_mat, seg_list):
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
        
        if len(pts) > 0:
            homo = np.hstack([pts, np.ones((len(pts), 1))])
            t_pts = (self.M_align @ (curr_mat @ homo.T)).T[:, :2]
            for i in range(len(t_pts)-1):
                seg_list.append((t_pts[i], t_pts[i+1]))
            if tag == 'polygon':
                seg_list.append((t_pts[-1], t_pts[0]))

        for child in element:
            self._process_node(child, curr_mat, seg_list)

    def get_nearest_heading(self, pt):
        if not self.valid or not self.orientation_segments: return None
        min_d = float('inf')
        best_ang = None
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
        if best_ang is not None:
            return (best_ang + 360) % 360
        return None

class GProjection:
    def __init__(self, config_data: dict, base_dir: str = "."):
        self.config = config_data
        self.base_dir = base_dir
        self._init_matrices()
        self._init_parallax()
        self._init_svg()
        
    def _init_matrices(self):
        und = self.config.get('undistort', {})
        self.K = np.array(und.get('K'), dtype=np.float64)
        self.D = np.array(und.get('D'), dtype=np.float64)
        self.P = self.K.copy()
        
        hom = self.config.get('homography', {})
        self.H = np.array(hom.get('H'), dtype=np.float64)
        self.H_inv = np.linalg.inv(self.H)
        
    def _init_parallax(self):
        par = self.config.get('parallax', {})
        self.z_cam = par.get('z_cam_meters', 10.0)
        x_sat = par.get('x_cam_coords_sat', 0.0)
        y_sat = par.get('y_cam_coords_sat', 0.0)
        self.cam_sat = np.array([x_sat, y_sat], dtype=np.float64)
        self.px_per_m = par.get('px_per_meter', 1.0)
        if self.px_per_m <= 0.001: self.px_per_m = 1.0

    def _init_svg(self):
        self.svg_parser = None
        if self.config.get('use_svg', False):
            rel_path = self.config.get('inputs', {}).get('layout_path')
            if rel_path:
                full_path = os.path.join(self.base_dir, rel_path)
                align_A = self.config.get('layout_svg', {}).get('A')
                self.svg_parser = SVGParser(full_path, align_A)

    def get_svg_heading(self, sat_pt):
        if self.svg_parser and self.svg_parser.valid:
            return self.svg_parser.get_nearest_heading(sat_pt)
        return None

    # --- TRANSFORMATIONS ---
    def cctv_to_undistorted(self, u, v):
        src = np.array([[[u, v]]], dtype=np.float64)
        dst = cv2.undistortPoints(src, self.K, self.D, P=self.P)
        return tuple(dst[0,0])

    def undistorted_to_flat_sat(self, u_u, v_u):
        src = np.array([[[u_u, v_u]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.H)
        return tuple(dst[0,0])

    def flat_sat_to_undistorted(self, x, y):
        src = np.array([[[x, y]]], dtype=np.float64)
        dst = cv2.perspectiveTransform(src, self.H_inv)
        return tuple(dst[0,0])
    
    def undistorted_to_cctv(self, u_u, v_u):
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        x_norm = (u_u - cx) / fx
        y_norm = (v_u - cy) / fy
        obj_pts = np.array([[[x_norm, y_norm, 1.0]]], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(obj_pts, (0,0,0), (0,0,0), self.K, self.D)
        return tuple(img_pts[0,0])

    # --- PARALLAX ---
    def parallax_correct_ground_to_real(self, apparent_sat_pt, h):
        if self.z_cam == 0: return apparent_sat_pt
        A = np.array(apparent_sat_pt); C = self.cam_sat
        factor = (self.z_cam - h) / self.z_cam
        real_pt = C + (A - C) * factor
        return tuple(real_pt)

    def parallax_project_real_to_ground(self, real_sat_pt, h):
        if abs(self.z_cam - h) < 0.01: return real_sat_pt 
        R = np.array(real_sat_pt); C = self.cam_sat
        factor = self.z_cam / (self.z_cam - h)
        apparent_pt = C + (R - C) * factor
        return tuple(apparent_pt)

    def cctv_to_sat(self, u, v, h=0.0):
        u_u, v_u = self.cctv_to_undistorted(u, v)
        apparent_pt = self.undistorted_to_flat_sat(u_u, v_u)
        if h != 0: return self.parallax_correct_ground_to_real(apparent_pt, h)
        return apparent_pt

    def sat_to_cctv(self, x, y, h=0.0):
        if h != 0: flat_pt = self.parallax_project_real_to_ground((x, y), h)
        else: flat_pt = (x, y)
        u_u, v_u = self.flat_sat_to_undistorted(flat_pt[0], flat_pt[1])
        return self.undistorted_to_cctv(u_u, v_u)

    def get_ground_contact_from_box(self, rect, h_meters, ref_method="center_bottom_side", proj_method="down_h"):
        if hasattr(rect, 'x'): rx, ry, rw, rh = rect.x(), rect.y(), rect.width(), rect.height()
        else: rx, ry, rw, rh = rect
        cx = rx + rw/2
        if ref_method == "center_bottom_side": cy = ry + rh
        else: cy = ry + rh/2
            
        apparent_sat = self.cctv_to_sat(cx, cy, h=0)
        final_sat = apparent_sat
        if proj_method == "down_h": final_sat = self.parallax_correct_ground_to_real(apparent_sat, h_meters)
        elif proj_method == "down_h_2": final_sat = self.parallax_correct_ground_to_real(apparent_sat, h_meters / 2.0)
            
        gc_cctv = self.sat_to_cctv(final_sat[0], final_sat[1], h=0)
        return { "sat_coords": final_sat, "cctv_ref_point": (cx, cy), "cctv_ground_point": gc_cctv }

    def sat_floor_to_cctv_3d(self, sat_poly, obj_height_m):
        """
        Lifts a SAT floor polygon (4 points) to a 3D box (8 points) in CCTV pixel space.
        Matches legacy 'produce_stage4.py' logic.
        """
        # 1. Project Floor points (Z=0) from SAT -> CCTV Undistorted
        # Logic: SAT -> Flat -> Undistorted
        undistorted_ground_pts = []
        for p in sat_poly:
            flat_pt = p # On ground, flat = real
            u_u, v_u = self.flat_sat_to_undistorted(flat_pt[0], flat_pt[1])
            undistorted_ground_pts.append([u_u, v_u])
        
        # 2. Project Ceiling points (Z=h)
        # Logic: Parallax shift on SAT plane, then map back
        undistorted_top_pts = []
        if abs(self.z_cam - obj_height_m) < 0.01: factor = 1.0
        else: factor = self.z_cam / (self.z_cam - obj_height_m)
        
        c_sat = self.cam_sat
        
        for p in sat_poly:
            p_arr = np.array(p)
            # Find where the 'head' would appear on the ground plane (apparent position)
            # Formula reversed from correction: Apparent = C + (Real - C) * factor
            apparent_pt = c_sat + (p_arr - c_sat) * factor
            
            u_u, v_u = self.flat_sat_to_undistorted(apparent_pt[0], apparent_pt[1])
            undistorted_top_pts.append([u_u, v_u])
            
        # Combine: 4 Bottom, 4 Top
        all_undist = np.array(undistorted_ground_pts + undistorted_top_pts, dtype=np.float64)
        
        # 3. Apply Distortion (CCTV Intrinsics)
        # Convert to Normalized Ray: (u-cx)/fx
        fx, fy = self.K[0,0], self.K[1,1]
        cx, cy = self.K[0,2], self.K[1,2]
        
        obj_pts = []
        for pt in all_undist:
            x_n = (pt[0] - cx) / fx
            y_n = (pt[1] - cy) / fy
            obj_pts.append([x_n, y_n, 1.0])
            
        obj_pts = np.array(obj_pts, dtype=np.float64)
        rvec = np.zeros(3); tvec = np.zeros(3)
        
        distorted, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.D)
        return distorted.reshape(-1, 2).tolist()