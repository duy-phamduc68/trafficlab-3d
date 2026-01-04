import json
import copy
import numpy as np
from datetime import datetime

SCALE = 1.5

S = np.array([
    [SCALE, 0, 0],
    [0, SCALE, 0],
    [0, 0, 1]
])

S_INV = np.linalg.inv(S)


def scale_K(K):
    K = np.array(K, dtype=float)
    return (S @ K).tolist()


def scale_point(pt):
    return [pt[0] * SCALE, pt[1] * SCALE]


def upgrade_calibration(src_path, dst_path):
    with open(src_path, "r") as f:
        src = json.load(f)

    out = {}

    # ---------- META ----------
    out["meta"] = {
        "location_code": src["meta"]["location_code"],
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }

    # ---------- INPUTS ----------
    out["inputs"] = {
        **src["inputs"],
        "roi_path": src["inputs"].get("roi_path", "")
    }

    # ---------- UNDISTORT ----------
    intr = src["camera_intrinsics"]
    out["undistort"] = {
        "resolution": [1920, 1080],
        "K": scale_K(intr["K"]),
        "D": intr["D"],
        "model": intr["model"]
    }

    # ---------- HOMOGRAPHY ----------
    H = np.array(src["ground_projection"]["homography_H"], dtype=float)
    H_new = H @ S_INV

    anchors = []
    for a in src["anchors_data"]:
        anchors.append({
            "id": a["id"],
            "name": a["name"],
            "coords_cctv": scale_point(a["raw_cctv"]),
            "coords_sat": a["sat"]
        })

    out["homography"] = {
        "H": H_new.tolist(),
        "fov_polygon": src["ground_projection"]["fov_polygon_clipped"],
        "anchors_list": anchors
    }

    # ---------- PARALLAX ----------
    gp = src["ground_projection"]
    scale = src["scale"]

    out["parallax"] = {
        "x_cam_coords_sat": gp["camera_pos_sat"][0],
        "y_cam_coords_sat": gp["camera_pos_sat"][1],
        "z_cam_meters": gp["camera_z_meters"],
        "scale": {
            "measured_px": scale["measured_px"],
            "real_m": scale["real_m"],
            "reference_anchors": scale["reference_anchors"]
        },
        "px_per_meter": scale["px_per_meter"]
    }

    # ---------- SVG / ROI FLAGS ----------
    out["use_svg"] = True
    out["layout_svg"] = {
        "A": src["layout_alignment"]["affine_matrix_svg2sat"],
        "association_pairs": src["layout_alignment"]["association_pairs"]
    }

    out["use_roi"] = True
    out["roi_method"] = "partial"
    out["ref_method"] = "center_box"
    out["proj_method"] = "down_h_2"

    # ---------- WRITE ----------
    with open(dst_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"✔ Converted calibration written to: {dst_path}")


if __name__ == "__main__":
    upgrade_calibration(
        src_path="G_projection_svg_SHARK.json",
        dst_path="G_projection_SHARK.json"
    )
