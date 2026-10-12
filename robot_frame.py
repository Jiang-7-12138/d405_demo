from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from calibration import transform_points


def load_transform_json(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    mat = np.array(data["matrix"], dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {path}")
    return mat


def camera_points_to_robot_base(points_camera_m: np.ndarray, t_base_camera: np.ndarray | None) -> np.ndarray | None:
    if t_base_camera is None or points_camera_m.size == 0:
        return None
    return transform_points(t_base_camera, points_camera_m)


def camera_point_to_robot_base(point_camera_m: np.ndarray | None, t_base_camera: np.ndarray | None) -> np.ndarray | None:
    if point_camera_m is None or t_base_camera is None:
        return None
    return camera_points_to_robot_base(point_camera_m.reshape(1, 3), t_base_camera)[0]
