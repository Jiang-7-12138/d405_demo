from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_d405 import CameraIntrinsics
from calibration import camera_matrix, distortion_coeffs


@dataclass
class CheckerboardPose:
    valid: bool
    t_world_camera: np.ndarray | None
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    corners: np.ndarray | None
    reprojection_error_px: float | None


def estimate_checkerboard_world_pose(
    color_bgr: np.ndarray,
    intr: CameraIntrinsics,
    inner_corners: tuple[int, int] = (6, 6),
    square_size_m: float = 0.020,
) -> CheckerboardPose:
    """Estimate camera pose in a checkerboard frame using known square spacing.

    The world origin is the first corner in OpenCV's detected corner sequence.
    World +X/+Y follow that sequence's grid directions. D405 intrinsics are
    used as provided by the RealSense color stream.
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    pattern = tuple(int(v) for v in inner_corners)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    found, corners = False, None

    # SB is subpixel accurate, but exposure and glare can make one preprocessing
    # choice fail. Try the original and locally contrast-normalized image first.
    for candidate in (gray, clahe):
        found, corners = cv2.findChessboardCornersSB(
            candidate, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if found and corners is not None:
            break

    # Classic detector is a fallback for board views where SB misses the quads.
    if not found or corners is None:
        classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        for candidate in (gray, clahe):
            found, corners = cv2.findChessboardCorners(candidate, pattern, flags=classic_flags)
            if found and corners is not None:
                corners = cv2.cornerSubPix(
                    candidate,
                    corners.astype(np.float32),
                    winSize=(5, 5),
                    zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3),
                )
                break

    if not found or corners is None:
        return CheckerboardPose(False, None, None, None, None, None)

    obj = np.zeros((pattern[0] * pattern[1], 3), dtype=np.float32)
    grid = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    obj[:, :2] = grid * float(square_size_m)
    image_points = corners.reshape(-1, 2).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj.astype(np.float64), image_points, camera_matrix(intr), distortion_coeffs(intr),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return CheckerboardPose(False, None, None, None, corners, None)

    rotation_camera_world, _ = cv2.Rodrigues(rvec)
    t_camera_world = np.eye(4, dtype=np.float64)
    t_camera_world[:3, :3] = rotation_camera_world
    t_camera_world[:3, 3] = tvec.reshape(3)
    t_world_camera = np.linalg.inv(t_camera_world)

    projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix(intr), distortion_coeffs(intr))
    residual = projected.reshape(-1, 2) - image_points
    reprojection_error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return CheckerboardPose(True, t_world_camera, rvec, tvec, corners, reprojection_error)


def draw_checkerboard_pose(
    color_bgr: np.ndarray,
    intr: CameraIntrinsics,
    pose: CheckerboardPose,
    inner_corners: tuple[int, int] = (6, 6),
    axis_length_m: float = 0.060,
) -> np.ndarray:
    out = color_bgr.copy()
    if pose.corners is None:
        cv2.putText(out, "Checkerboard not found", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return out
    cv2.drawChessboardCorners(out, inner_corners, pose.corners, pose.valid)
    if pose.valid and pose.rvec is not None and pose.tvec is not None:
        cv2.drawFrameAxes(out, camera_matrix(intr), distortion_coeffs(intr), pose.rvec, pose.tvec, axis_length_m, 2)
        label = f"Checkerboard World | reproj RMS {pose.reprojection_error_px:.2f}px"
        cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (40, 220, 40), 2, cv2.LINE_AA)
    return out


def save_world_camera_transform(path: str | Path, pose: CheckerboardPose, intr: CameraIntrinsics) -> None:
    if not pose.valid or pose.t_world_camera is None:
        raise ValueError("Cannot save transform: checkerboard pose is invalid")
    data = {
        "matrix": pose.t_world_camera.tolist(),
        "transform": "T_world_camera",
        "world_frame": "7x7 checkerboard inner-corner frame",
        "inner_corners": [6, 6],
        "square_size_mm": 20.0,
        "origin": "first detected inner corner; +X/+Y follow detected grid order, +Z by right-hand rule",
        "intrinsics": {
            "width": intr.width, "height": intr.height,
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "coeffs": list(intr.coeffs),
        },
        "reprojection_error_px": pose.reprojection_error_px,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
