"""Coordinate frame on the endoscope calibration cube.

The cube carries DICT_5X5_50 tags. The face that points most toward the camera
is the top. Its center is the origin. Z is the outward normal of that face
(up, off the board). X lies in the face and runs horizontally left in the
image. Y lies in the face and runs toward the bottom of the image.
X, Y, and Z are orthonormal and right-handed: X cross Y = Z.

This does not use the organ color gates, so the cube is not labeled as the
gallbladder.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MARKER_DICTIONARY = cv2.aruco.DICT_5X5_50
MARKER_SIZE_M = 0.015
AXIS_LENGTH_M = 0.046
LABEL_GAP_PX = 34


@dataclass
class MarkerFrame:
    valid: bool
    marker_id: int | None = None
    origin_px: tuple[int, int] | None = None
    origin_camera_m: np.ndarray | None = None
    x_axis: np.ndarray | None = None
    y_axis: np.ndarray | None = None
    z_axis: np.ndarray | None = None


def _camera_matrix(width: int, height: int) -> np.ndarray:
    # The endoscope has no saved intrinsics. This focal length only sets the
    # pose scale; the axis directions are fixed by the tag plane.
    focal = 700.0
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _object_corners(marker_size_m: float) -> np.ndarray:
    half = marker_size_m / 2.0
    return np.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def _image_step(point: np.ndarray, direction: np.ndarray, focal: float) -> np.ndarray:
    pz = float(point[2])
    du = focal * (pz * direction[0] - point[0] * direction[2]) / pz**2
    dv = focal * (pz * direction[1] - point[1] * direction[2]) / pz**2
    return np.array([du, dv], dtype=np.float64)


def _axes_on_face(normal: np.ndarray, origin: np.ndarray, focal: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X left in the image, Y down in the image, both in the face plane."""
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent = np.cross(z_axis, helper)
    tangent /= max(np.linalg.norm(tangent), 1e-12)
    bitangent = np.cross(z_axis, tangent)
    basis = np.column_stack(
        [_image_step(origin, tangent, focal), _image_step(origin, bitangent, focal)]
    )
    weights = np.linalg.solve(basis, np.array([-1.0, 0.0]))
    x_axis = tangent * weights[0] + bitangent * weights[1]
    x_axis /= max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= max(np.linalg.norm(y_axis), 1e-12)
    return x_axis, y_axis, z_axis


def detect_marker_frame(color_bgr: np.ndarray, marker_size_m: float = MARKER_SIZE_M) -> MarkerFrame:
    """Find the cube's top tag and build the image-aligned frame."""
    height, width = color_bgr.shape[:2]
    camera = _camera_matrix(width, height)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return MarkerFrame(False)

    object_corners = _object_corners(marker_size_m)
    distortion = np.zeros(5, dtype=np.float64)
    best: tuple[float, MarkerFrame] | None = None
    for index, marker_corners in enumerate(corners):
        image_points = marker_corners.reshape(-1, 1, 2).astype(np.float32)
        cv2.cornerSubPix(
            gray,
            image_points,
            (3, 3),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        )
        image_points = image_points.reshape(4, 2).astype(np.float64)
        solution_count, rvecs, tvecs, _errors = cv2.solvePnPGeneric(
            object_corners,
            image_points,
            camera,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if solution_count < 1:
            continue
        # IPPE returns two poses with the same camera-facing score. One Z
        # points up in the image and the other points down. Taking whichever
        # solvePnP returned first made Z flip from frame to frame.
        focal = float(camera[0, 0])
        chosen: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for rvec, tvec in zip(rvecs, tvecs):
            rotation, _ = cv2.Rodrigues(rvec)
            origin = tvec.reshape(3)
            normal = rotation[:, 2]
            toward_camera = -origin / max(np.linalg.norm(origin), 1e-12)
            score = float(normal @ toward_camera)
            image_z = _image_step(origin, normal, focal)
            if score <= 0.35 or image_z[1] >= 0.0:
                continue
            if chosen is None or score > chosen[0]:
                chosen = (score, origin, normal)
        if chosen is None:
            continue
        _score, origin, normal = chosen
        x_axis, y_axis, z_axis = _axes_on_face(normal, origin, focal)
        center = image_points.mean(axis=0)
        pose = MarkerFrame(
            True,
            int(ids[index].item()),
            (int(round(float(center[0]))), int(round(float(center[1])))),
            origin,
            x_axis,
            y_axis,
            z_axis,
        )
        if best is None or _score > best[0]:
            best = (_score, pose)
    if best is None:
        return MarkerFrame(False)
    return best[1]


def draw_marker_frame(color_bgr: np.ndarray, pose: MarkerFrame, axis_length_m: float = AXIS_LENGTH_M) -> np.ndarray:
    """Draw X left, Y down, and Z up. Does not paint the cube as an organ."""
    if not pose.valid or pose.origin_camera_m is None or pose.x_axis is None or pose.y_axis is None or pose.z_axis is None:
        return color_bgr
    height, width = color_bgr.shape[:2]
    camera = _camera_matrix(width, height)
    distortion = np.zeros(5, dtype=np.float64)
    origin = pose.origin_camera_m.reshape(1, 3)
    tips = np.vstack(
        [
            origin,
            origin + pose.x_axis.reshape(1, 3) * axis_length_m,
            origin + pose.y_axis.reshape(1, 3) * axis_length_m,
            origin + pose.z_axis.reshape(1, 3) * axis_length_m,
        ]
    )
    projected, _ = cv2.projectPoints(tips, np.zeros(3), np.zeros(3), camera, distortion)
    points = projected.reshape(4, 2)
    start = np.round(points[0]).astype(int)
    colors = ((0, 0, 255), (0, 200, 0), (255, 180, 0))
    names = ("X", "Y", "Z")
    for offset, color, name in zip(points[1:], colors, names):
        end = np.round(offset).astype(int)
        cv2.arrowedLine(color_bgr, tuple(start), tuple(end), (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.18)
        cv2.arrowedLine(color_bgr, tuple(start), tuple(end), color, 2, cv2.LINE_AA, tipLength=0.18)
        direction = (end - start).astype(np.float64)
        length = float(np.linalg.norm(direction))
        unit = direction / length if length > 1.0 else np.array([1.0, 0.0])
        (text_w, text_h), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        anchor = end.astype(np.float64) + unit * LABEL_GAP_PX
        text_x = int(round(anchor[0] - text_w / 2.0))
        text_y = int(round(anchor[1] + text_h / 2.0))
        text_x = max(0, min(width - text_w - 2, text_x))
        text_y = max(text_h + 2, min(height - 2, text_y))
        cv2.putText(color_bgr, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return color_bgr


class MarkerFrameHold:
    """Keep Z from twitching when the tag corners move by a pixel or two."""

    def __init__(self, deadband_deg: float = 8.0, hold_frames: int = 8) -> None:
        self.deadband_deg = float(deadband_deg)
        self.hold_frames = int(hold_frames)
        self._pose = MarkerFrame(False)
        self._missing = 0

    def update(self, pose: MarkerFrame) -> MarkerFrame:
        if not pose.valid or pose.z_axis is None or pose.origin_px is None:
            self._missing += 1
            if self._missing > self.hold_frames:
                self._pose = MarkerFrame(False)
            return self._pose
        self._missing = 0
        if self._pose.valid and self._pose.z_axis is not None and self._pose.origin_px is not None:
            dot = float(np.clip(self._pose.z_axis @ pose.z_axis, -1.0, 1.0))
            angle = float(np.degrees(np.arccos(dot)))
            shift = float(np.hypot(pose.origin_px[0] - self._pose.origin_px[0], pose.origin_px[1] - self._pose.origin_px[1]))
            if angle > 35.0 or (angle < self.deadband_deg and shift < 8.0):
                return self._pose
        self._pose = pose
        return pose
