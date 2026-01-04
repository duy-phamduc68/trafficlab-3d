import json
from datetime import datetime
from typing import Dict, Any, Optional


def default_config(location_code: str = "PLACEHOLDER", timestamp: Optional[str] = None) -> Dict[str, Any]:
	"""Return a default placeholder TrafficLab JSON configuration dict.

	The structure follows the example provided by the user and is intended
	to act as a schema/placeholder to be filled in by the calibration UI.
	"""
	if timestamp is None:
		timestamp = datetime.now().replace(microsecond=0).isoformat()

	return {
		"meta": {
			"location_code": location_code,
			"timestamp": timestamp,
		},
		"inputs": {
			"cctv_path": f"cctv_{location_code}.png",
			"sat_path": f"sat_{location_code}.png",
			"layout_path": f"layout_{location_code}.svg",
			"roi_path": f"roi_{location_code}.png",
			"note": "Input paths are relative to this json file",
		},
		"undistort": {
			"resolution": [1280, 720],
			"K": [[1280.0, 0.0, 640.0], [0.0, 1280.0, 360.0], [0.0, 0.0, 1.0]],
			"D": [0.0, 0.0, 0.0, 0.0, 0.0],
			"model": "radial_tangential",
		},
		"homography": {
			"H": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
			"fov_polygon": [],
			"anchors_list": [],
		},
		"parallax": {
			"x_cam_coords_sat": 0.0,
			"y_cam_coords_sat": 0.0,
			"z_cam_meters": 0.0,
			"scale": {
				"measured_px": 0.0,
				"real_m": 0.0,
				"reference_anchors": [],
			},
			"px_per_meter": 0.0,
		},
		"use_svg": False,
		"layout_svg": {
			"A": [],
			"association_pairs": [],
		},
		"use_roi": False,
		"roi_method": "partial",
		"ref_method": "center_box",
		"proj_method": "down_h_2",
	}


def to_pretty_json(obj: Dict[str, Any]) -> str:
	return json.dumps(obj, indent=4, sort_keys=False)


def save_config(path: str, obj: Dict[str, Any]) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(obj, f, indent=4)


def load_config(path: str) -> Dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)

