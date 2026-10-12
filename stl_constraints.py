from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

from calibration import camera_matrix, transform_points
from camera_d405 import CameraIntrinsics


def load_stl_vertices(path: str) -> np.ndarray:
    """Load STL vertices without external dependencies.

    Supports common binary STL and simple ASCII STL. Units are whatever the STL
    uses; the provided phantom filenames say `_mm`, so callers should scale by
    0.001 before projecting into a meter-based camera frame.
    """
    p = Path(path)
    data = p.read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + tri_count * 50
        if expected == len(data):
            vertices = []
            offset = 84
            for _ in range(tri_count):
                offset += 12  # normal
                for _ in range(3):
                    vertices.append(struct.unpack_from("<fff", data, offset))
                    offset += 12
                offset += 2
            return np.array(vertices, dtype=np.float32)

    text = data.decode("utf-8", errors="ignore")
    vertices = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.array(vertices, dtype=np.float32)


def stl_projected_mask(
    stl_path: str,
    intr: CameraIntrinsics,
    t_camera_stl: np.ndarray,
    image_shape: tuple[int, int],
    stl_scale: float = 0.001,
    hull_only: bool = True,
) -> np.ndarray:
    """Project an STL into the image as a coarse constraint mask.

    This is a geometry prior, not a renderer. It is useful once you know the STL
    pose in camera coordinates. For the current project, that pose can come from
    CAD fixture geometry or registration to the 15 mm reference marker.
    """
    vertices = load_stl_vertices(stl_path)
    if vertices.size == 0:
        return np.zeros(image_shape, dtype=bool)
    vertices_m = vertices.astype(np.float32) * float(stl_scale)
    points_camera = transform_points(t_camera_stl, vertices_m)
    points_camera = points_camera[np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 1e-5)]
    if points_camera.size == 0:
        return np.zeros(image_shape, dtype=bool)

    k = camera_matrix(intr)
    uvw = (k @ points_camera.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    h, w = image_shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[ok].astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    if uv.shape[0] < 3:
        return mask.astype(bool)
    if hull_only:
        hull = cv2.convexHull(uv.reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 255)
    else:
        mask[uv[:, 1], uv[:, 0]] = 255
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    return mask.astype(bool)


def apply_stl_constraint(mask: np.ndarray, stl_mask: np.ndarray | None) -> np.ndarray:
    if stl_mask is None or not np.any(stl_mask):
        return mask
    return mask.astype(bool) & stl_mask.astype(bool)
