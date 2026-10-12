from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from camera_d405 import CameraIntrinsics
from io_utils import imwrite_unicode


@dataclass
class ArucoPose:
    valid: bool
    t_world_camera: Optional[np.ndarray]
    rvec: Optional[np.ndarray]
    tvec: Optional[np.ndarray]
    marker_id: Optional[int]


def camera_matrix(intr: CameraIntrinsics) -> np.ndarray:
    return np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def distortion_coeffs(intr: CameraIntrinsics) -> np.ndarray:
    coeffs = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)
    if coeffs.size == 0:
        return np.zeros((5, 1), dtype=np.float64)
    return coeffs


def estimate_aruco_world_pose(
    color_bgr: np.ndarray,
    intr: CameraIntrinsics,
    marker_size_m: float = 0.015,
    marker_id: int = 0,
) -> ArucoPose:
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return ArucoPose(False, None, None, None, None)

    ids_flat = ids.reshape(-1)
    matches = np.where(ids_flat == marker_id)[0]
    if matches.size == 0:
        return ArucoPose(False, None, None, None, int(ids_flat[0]))
    i = int(matches[0])

    obj = np.array(
        [
            [-marker_size_m / 2.0, marker_size_m / 2.0, 0.0],
            [marker_size_m / 2.0, marker_size_m / 2.0, 0.0],
            [marker_size_m / 2.0, -marker_size_m / 2.0, 0.0],
            [-marker_size_m / 2.0, -marker_size_m / 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    img = corners[i].reshape(4, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, camera_matrix(intr), distortion_coeffs(intr), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return ArucoPose(False, None, None, None, marker_id)

    r_world_to_camera, _ = cv2.Rodrigues(rvec)
    t_world_to_camera = tvec.reshape(3, 1)

    t_camera_world = np.eye(4, dtype=np.float64)
    t_camera_world[:3, :3] = r_world_to_camera
    t_camera_world[:3, 3:4] = t_world_to_camera
    t_world_camera = np.linalg.inv(t_camera_world)

    return ArucoPose(True, t_world_camera, rvec, tvec, marker_id)


def transform_points(t_dst_src: np.ndarray, points_src: np.ndarray) -> np.ndarray:
    if points_src.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    ones = np.ones((points_src.shape[0], 1), dtype=np.float64)
    homo = np.hstack([points_src.astype(np.float64), ones])
    out = (t_dst_src @ homo.T).T[:, :3]
    return out.astype(np.float32)


def draw_aruco_axis(color_bgr: np.ndarray, intr: CameraIntrinsics, pose: ArucoPose, axis_len_m: float = 0.015) -> np.ndarray:
    out = color_bgr.copy()
    if pose.valid and pose.rvec is not None and pose.tvec is not None:
        cv2.drawFrameAxes(out, camera_matrix(intr), distortion_coeffs(intr), pose.rvec, pose.tvec, axis_len_m)
    return out


def create_aruco_marker_png(path: str, side_px: int = 900, marker_id: int = 0) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    imwrite_unicode(path, marker)
