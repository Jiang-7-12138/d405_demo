from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    coeffs: Tuple[float, ...]


@dataclass
class CameraFrame:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    depth_colormap: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_ms: float


class RealSenseD405:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, use_filters: bool = False):
        self.width = width
        self.height = height
        self.fps = fps
        self.use_filters = use_filters
        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self.depth_scale = 0.001
        self.spatial_filter = None
        self.temporal_filter = None
        self.hole_filling_filter = None

    @staticmethod
    def list_devices() -> list[tuple[str, str]]:
        ctx = rs.context()
        devices = []
        for dev in ctx.devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            devices.append((name, serial))
        return devices

    @staticmethod
    def reset_devices() -> None:
        ctx = rs.context()
        for dev in ctx.devices:
            dev.hardware_reset()

    def start(self) -> None:
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        try:
            profile = self.pipeline.start(config)
        except Exception:
            self.pipeline = None
            raise

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.align = rs.align(rs.stream.color)

        if self.use_filters:
            self.spatial_filter = rs.spatial_filter()
            self.spatial_filter.set_option(rs.option.filter_magnitude, 2)
            self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
            self.temporal_filter = rs.temporal_filter()
            self.hole_filling_filter = rs.hole_filling_filter()

        # Warm up auto-exposure and stereo matching. If this fails, the device
        # was enumerated but is not actually delivering frames.
        try:
            for _ in range(15):
                self.pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError as exc:
            self.stop()
            raise RuntimeError(
                "RealSense device was found, but no frames arrived. Close other camera/RealSense programs, "
                "unplug/replug the D405, try another USB port, or rerun with --reset-camera."
            ) from exc

    def stop(self) -> None:
        if self.pipeline is not None:
            self.pipeline.stop()
        self.pipeline = None
        self.align = None

    def read(self) -> CameraFrame:
        if self.pipeline is None or self.align is None:
            raise RuntimeError("Camera is not started")

        frames = self.pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to get synchronized color/depth frames")

        if self.use_filters:
            if self.spatial_filter is not None:
                depth_frame = self.spatial_filter.process(depth_frame)
            if self.temporal_filter is not None:
                depth_frame = self.temporal_filter.process(depth_frame)
            if self.hole_filling_filter is not None:
                depth_frame = self.hole_filling_filter.process(depth_frame)

        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * float(self.depth_scale)

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        intrinsics = CameraIntrinsics(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            coeffs=tuple(intr.coeffs),
        )

        import cv2

        clipped = np.clip(depth_m, 0.0, 0.45)
        depth_8u = cv2.convertScaleAbs(clipped, alpha=255.0 / 0.45)
        depth_colormap = cv2.applyColorMap(depth_8u, cv2.COLORMAP_TURBO)
        depth_colormap[depth_m <= 0] = (0, 0, 0)

        return CameraFrame(
            color_bgr=color,
            depth_m=depth_m,
            depth_colormap=depth_colormap,
            intrinsics=intrinsics,
            timestamp_ms=frames.get_timestamp(),
        )
