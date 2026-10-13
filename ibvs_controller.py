from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IBVSCommand:
    valid: bool
    reason: str
    error_px: tuple[float, float]
    error_norm_px: float
    camera_linear_m_s: tuple[float, float, float]
    base_linear_m_s: tuple[float, float, float]


class PointIBVSController:
    """Translation-only image-based servo for a tracked tool-tip point.

    The camera is fixed. `tool_pixel` must be the current image location of
    the moving tool tip, while `target_pixel` is the desired image location.
    The returned velocity is a suggestion in robot-base coordinates; this
    class never communicates with or commands a robot.
    """

    def __init__(
        self,
        gain: float = 0.25,
        max_linear_speed_m_s: float = 0.005,
        deadband_px: float = 2.0,
        max_error_px: float = 120.0,
        min_depth_m: float = 0.05,
        max_depth_m: float = 0.70,
    ) -> None:
        if gain <= 0 or max_linear_speed_m_s <= 0:
            raise ValueError("gain and max_linear_speed_m_s must be positive")
        if deadband_px < 0 or max_error_px <= deadband_px:
            raise ValueError("max_error_px must be greater than non-negative deadband_px")
        if min_depth_m <= 0 or max_depth_m <= min_depth_m:
            raise ValueError("invalid depth range")
        self.gain = float(gain)
        self.max_linear_speed_m_s = float(max_linear_speed_m_s)
        self.deadband_px = float(deadband_px)
        self.max_error_px = float(max_error_px)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

    def compute(
        self,
        tool_pixel: tuple[float, float] | np.ndarray,
        target_pixel: tuple[float, float] | np.ndarray,
        tool_depth_m: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        t_base_camera: np.ndarray,
    ) -> IBVSCommand:
        """Compute a bounded base-frame linear velocity from pixel error.

        `t_base_camera` maps camera-frame coordinates into robot-base frame.
        No angular velocity is generated. A robot Jacobian/driver must still
        turn this Cartesian reference into safe joint commands.
        """
        tool = np.asarray(tool_pixel, dtype=np.float64).reshape(-1)
        target = np.asarray(target_pixel, dtype=np.float64).reshape(-1)
        transform = np.asarray(t_base_camera, dtype=np.float64)
        error = target - tool if tool.size == 2 and target.size == 2 else np.zeros(2)

        def stop(reason: str) -> IBVSCommand:
            return IBVSCommand(False, reason, tuple(error), float(np.linalg.norm(error)), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        if tool.size != 2 or target.size != 2 or not np.all(np.isfinite([*tool, *target])):
            return stop("invalid_image_feature")
        if not np.isfinite(tool_depth_m) or not self.min_depth_m <= tool_depth_m <= self.max_depth_m:
            return stop("invalid_tool_depth")
        if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0 or fy <= 0:
            return stop("invalid_intrinsics")
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            return stop("invalid_transform")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            return stop("invalid_transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
            return stop("invalid_rotation")

        error_norm = float(np.linalg.norm(error))
        if error_norm > self.max_error_px:
            return stop("error_out_of_range")
        if error_norm <= self.deadband_px:
            return IBVSCommand(True, "within_deadband", tuple(error), error_norm, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

        z = float(tool_depth_m)
        x = (float(tool[0]) - cx) / fx
        y = (float(tool[1]) - cy) / fy
        # Point-feature interaction matrix for translation of the observed
        # point in camera coordinates, with the camera itself stationary.
        interaction = np.array([[1.0 / z, 0.0, -x / z], [0.0, 1.0 / z, -y / z]])
        desired_feature_rate = self.gain * np.array([error[0] / fx, error[1] / fy])
        camera_velocity = np.linalg.pinv(interaction) @ desired_feature_rate
        speed = float(np.linalg.norm(camera_velocity))
        if speed > self.max_linear_speed_m_s:
            camera_velocity *= self.max_linear_speed_m_s / speed

        base_velocity = rotation @ camera_velocity
        return IBVSCommand(
            True,
            "tracking",
            (float(error[0]), float(error[1])),
            error_norm,
            tuple(float(v) for v in camera_velocity),
            tuple(float(v) for v in base_velocity),
        )
