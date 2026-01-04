import numpy as np
import math
import cv2
from collections import deque

class TrackSmoother:
    def __init__(self, config: dict):
        self.cfg = config
        
        # State
        self.speed_state = None          # km/h
        self.heading_vec_state = None    
        self.velocity_vec_state = None   
        
        # Params
        h_ema = self.cfg.get('heading_ema', {})
        self.h_alpha_min = h_ema.get('alpha_min', 0.05)
        self.h_alpha_max = h_ema.get('alpha_max', 0.6)
        self.h_speed_ref = h_ema.get('speed_ref', 5.0)
        
        # History
        reg_win = 8
        self.pos_history = deque(maxlen=reg_win)
        jitter_frames = self.cfg.get('heading_sat_coords_jitter_frames', 8)
        self.jitter_hist = deque(maxlen=jitter_frames)
        
        self.cosine_reject_counter = 0
        self.max_physics_speed = 200.0 # km/h

    def update(self, current_sat_pos: list, dt: float, px_per_m: float, svg_heading: float = None) -> dict:
        curr_pt = np.array(current_sat_pos)
        
        # 1. Physics Speed Gate
        if dt > 0 and len(self.pos_history) > 0:
            prev_pt = np.array(self.pos_history[-1])
            dist_px = np.linalg.norm(curr_pt - prev_pt)
            dist_m = dist_px / px_per_m  # <--- CRITICAL FIX
            inst_speed = (dist_m / dt) * 3.6
            
            if inst_speed > self.max_physics_speed:
                return {
                    "speed_kmh": self.speed_state if self.speed_state else 0.0,
                    "heading": self._vec_to_deg(self.heading_vec_state) if self.heading_vec_state is not None else None,
                    "default_heading": False
                }

        self.pos_history.append(current_sat_pos)
        self.jitter_hist.append(current_sat_pos)
        
        is_jittering = self._check_jitter(px_per_m)
        
        raw_speed_kmh = 0.0
        raw_heading_deg = None
        
        if dt > 0 and len(self.pos_history) >= 2:
            prev_pt = np.array(self.pos_history[-2])
            dist_px = np.linalg.norm(curr_pt - prev_pt)
            dist_m = dist_px / px_per_m # <--- CRITICAL FIX
            raw_speed_kmh = (dist_m / dt) * 3.6
            
            reg_heading = self._compute_regression_heading()
            if reg_heading is not None:
                raw_heading_deg = reg_heading
            elif dist_px > 0.5: # Min pixel movement
                dx, dy = curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1]
                raw_heading_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360

        # Speed Smoothing
        self.speed_state = self._smooth_speed(raw_speed_kmh)
        
        min_speed = self.cfg.get('heading_min_speed_for_update', 0.1)
        final_heading = None
        is_default = False
        
        # 2. Logic Flow
        if raw_speed_kmh > min_speed and not is_jittering and raw_heading_deg is not None:
            # Match Legacy: Pass speed in m/s to smoothing ratio if ref is low
            final_heading = self._smooth_heading(raw_heading_deg, raw_speed_kmh)
            
            if svg_heading is not None:
                final_heading = self._apply_snapping(final_heading, svg_heading)
            
            is_default = False
        else:
            if self.heading_vec_state is not None:
                final_heading = self._vec_to_deg(self.heading_vec_state)
                is_default = False
            elif svg_heading is not None:
                final_heading = svg_heading
                is_default = True
            else:
                final_heading = None
                is_default = False

        return {
            "speed_kmh": self.speed_state if self.speed_state else 0.0,
            "heading": final_heading,
            "default_heading": is_default
        }

    # --- Helpers ---
    def _check_jitter(self, px_per_m):
        if len(self.jitter_hist) < 2: return False
        rad_m = self.cfg.get('heading_sat_coords_jitter_radius', 0.6)
        rad_px = rad_m * px_per_m
        pts = np.array(self.jitter_hist)
        return np.linalg.norm(np.max(pts,0) - np.min(pts,0)) < rad_px

    def _smooth_speed(self, raw):
        alpha = self.cfg.get('speed_ema_alpha', 0.4)
        if self.speed_state is None: return raw
        return alpha * raw + (1 - alpha) * self.speed_state

    def _compute_regression_heading(self):
        if len(self.pos_history) < 3: return None
        pts = np.array(self.pos_history)
        if np.linalg.norm(pts[-1] - pts[0]) < 2.0: return None # Min pixels for regression
        [vx, vy, x, y] = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        disp = pts[-1] - pts[0]
        if np.dot(disp, [vx[0], vy[0]]) < 0: vx, vy = -vx, -vy
        return self._vec_to_deg([vx[0], vy[0]])

    def _smooth_heading(self, raw_deg, speed_kmh):
        rad = math.radians(raw_deg)
        curr_vec = np.array([math.cos(rad), math.sin(rad)])
        
        # Velocity Smoothing
        vel_alpha = 0.25
        if self.velocity_vec_state is None: self.velocity_vec_state = curr_vec
        else:
            self.velocity_vec_state = vel_alpha * curr_vec + (1 - vel_alpha) * self.velocity_vec_state
            self.velocity_vec_state /= np.linalg.norm(self.velocity_vec_state)
        curr_vec = self.velocity_vec_state
        
        if self.heading_vec_state is None:
            self.heading_vec_state = curr_vec
            return self._vec_to_deg(curr_vec)

        # Adaptive Alpha (Legacy divides speed by 3.6 for m/s ref)
        speed_ms = speed_kmh / 3.6
        ratio = min(1.0, speed_ms / self.h_speed_ref)
        alpha = self.h_alpha_min + (self.h_alpha_max - self.h_alpha_min) * ratio
        
        new_vec = alpha * curr_vec + (1 - alpha) * self.heading_vec_state
        new_vec /= np.linalg.norm(new_vec)
        
        target_deg = self._vec_to_deg(new_vec)
        prev_deg = self._vec_to_deg(self.heading_vec_state)
        delta = (target_deg - prev_deg + 180) % 360 - 180
        
        max_jump = self.cfg.get('heading_max_jump', 5)
        if abs(delta) > max_jump:
            delta = np.clip(delta, -max_jump, max_jump)
            target_deg = (prev_deg + delta + 360) % 360
            
        r_final = math.radians(target_deg)
        self.heading_vec_state = np.array([math.cos(r_final), math.sin(r_final)])
        return target_deg

    def _apply_snapping(self, current_heading, svg_heading):
        diff = (current_heading - svg_heading + 180) % 360 - 180
        if abs(diff) < 15: # 15 deg thresh
            return (current_heading - diff * 0.3 + 360) % 360
        return current_heading

    def _vec_to_deg(self, vec):
        return (math.degrees(math.atan2(vec[1], vec[0])) + 360) % 360