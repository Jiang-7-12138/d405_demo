from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

from camera_d405 import CameraIntrinsics


@dataclass
class PointStats:
    valid: bool
    centroid_camera_m: Optional[np.ndarray]
    median_camera_m: Optional[np.ndarray]
    valid_points: int
    total_pixels: int
    valid_ratio: float
    depth_median_m: Optional[float]
    depth_mad_m: Optional[float]


def pixel_to_camera_xyz(u: int, v: int, depth_m: float, intr: CameraIntrinsics) -> Optional[np.ndarray]:
    if depth_m <= 0 or not np.isfinite(depth_m):
        return None
    x = (float(u) - intr.cx) * depth_m / intr.fx
    y = (float(v) - intr.cy) * depth_m / intr.fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float32)


def mask_to_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    min_depth_m: float = 0.03,
    max_depth_m: float = 0.45,
    max_points: int = 50000,
) -> np.ndarray:
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    valid = mask & np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    if xs.size > max_points:
        idx = np.linspace(0, xs.size - 1, max_points).astype(np.int64)
        xs = xs[idx]
        ys = ys[idx]

    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - intr.cx) * z / intr.fx
    y = (ys.astype(np.float32) - intr.cy) * z / intr.fy
    return np.column_stack((x, y, z)).astype(np.float32)


def robust_mask_stats(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    min_depth_m: float = 0.03,
    max_depth_m: float = 0.45,
) -> PointStats:
    total_pixels = int(np.count_nonzero(mask))
    points = mask_to_points(mask, depth_m, intr, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    valid_points = int(points.shape[0])
    valid_ratio = float(valid_points / total_pixels) if total_pixels else 0.0
    if valid_points < 20:
        return PointStats(False, None, None, valid_points, total_pixels, valid_ratio, None, None)

    z = points[:, 2]
    z_med = float(np.median(z))
    mad = float(np.median(np.abs(z - z_med)))
    keep = np.abs(z - z_med) <= max(0.004, 3.0 * mad)
    filtered = points[keep]
    if filtered.shape[0] < 20:
        filtered = points

    centroid = np.mean(filtered, axis=0)
    median = np.median(filtered, axis=0)
    return PointStats(
        True,
        centroid.astype(np.float32),
        median.astype(np.float32),
        int(filtered.shape[0]),
        total_pixels,
        valid_ratio,
        z_med,
        mad,
    )


def roi_mask(shape: Tuple[int, int], p0: tuple[int, int], p1: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x0, y0 = p0
    x1, y1 = p1
    x0, x1 = sorted((max(0, min(w - 1, x0)), max(0, min(w - 1, x1))))
    y0, y1 = sorted((max(0, min(h - 1, y0)), max(0, min(h - 1, y1))))
    mask = np.zeros((h, w), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask


def depth_to_points_sampled(
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    color_bgr: Optional[np.ndarray] = None,
    stride: int = 4,
    min_depth_m: float = 0.03,
    max_depth_m: float = 0.45,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    ys = np.arange(0, depth_m.shape[0], max(1, stride), dtype=np.int32)
    xs = np.arange(0, depth_m.shape[1], max(1, stride), dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth_m[grid_y, grid_x]
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), None

    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    points = np.column_stack((x, y, z)).astype(np.float32)

    colors = None
    if color_bgr is not None:
        colors = color_bgr[grid_y[valid], grid_x[valid]].astype(np.uint8)
    return points, colors


def fill_small_depth_holes(depth_m: np.ndarray, max_hole_size: int = 120) -> np.ndarray:
    """Fill only small internal invalid-depth holes for visualization.

    This is deliberately separate from measurement depth. It should not be
    used to compute lesion coordinates because interpolated values are not
    original sensor measurements.
    """
    depth = depth_m.astype(np.float32, copy=True)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return depth

    # A closed valid mask identifies holes surrounded by measured depth.
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    candidate = closed & ~valid
    if not np.any(candidate):
        return depth

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    fill_mask = np.zeros(candidate.shape, dtype=np.uint8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area <= max_hole_size and width <= 20 and height <= 20:
            fill_mask[labels == label] = 255

    if np.any(fill_mask):
        depth = cv2.inpaint(depth, fill_mask, 3, cv2.INPAINT_NS)
    return depth
