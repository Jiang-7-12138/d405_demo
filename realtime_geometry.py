from __future__ import annotations
import cv2
import numpy as np
from calibration import transform_points

def _points(mask, depth, intr, lo, hi):
    valid = np.asarray(mask, bool) & np.isfinite(depth) & (depth >= lo) & (depth <= hi)
    y, x = np.nonzero(valid)
    if not x.size: return np.empty((0, 3), np.float32), x, y
    z = depth[y, x].astype(np.float32)
    return np.column_stack(((x-intr.cx)*z/intr.fx, (y-intr.cy)*z/intr.fy, z)).astype(np.float32), x, y

def _bounds(points, transform, margin):
    if len(points) < 10: return None
    lo, hi = np.percentile(points, [5, 95], axis=0); lo -= margin; hi += margin
    center = np.median(points, axis=0)
    result = {"center_m": center.tolist(), "aabb_min_m": lo.tolist(), "aabb_max_m": hi.tolist(), "extent_m": (hi-lo).tolist(), "bounds_method": "5th-95th percentile with margin"}
    if transform is not None:
        base = {"center_m": transform_points(transform, center.reshape(1,3))[0].tolist()}
        corners = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        mapped = transform_points(transform, corners)
        base["aabb_min_m"], base["aabb_max_m"] = mapped.min(axis=0).tolist(), mapped.max(axis=0).tolist()
        result["base"] = base
    return result

def _encode_mask_rle(mask):
    flat = np.asarray(mask, dtype=np.uint8).reshape(-1)
    counts = []
    value, run = 0, 0
    for pixel in flat:
        pixel = int(pixel != 0)
        if pixel == value:
            run += 1
        else:
            counts.append(run)
            value, run = pixel, 1
    counts.append(run)
    return counts

def _lesion_instances(mask, depth, intr, min_depth, max_depth):
    if mask is None:
        return []
    binary = np.asarray(mask, dtype=np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    instances = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < 1:
            continue
        component = labels[y:y+h, x:x+w] == label
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour_px = [[(int(px) + x, int(py) + y) for px, py in contour[:, 0, :]] for contour in contours]
        points, _, _ = _points(labels == label, depth, intr, min_depth, max_depth)
        center_3d = np.median(points, axis=0).tolist() if len(points) >= 10 else None
        instances.append({
            "id": len(instances) + 1,
            "center_px": [float(centroids[label][0]), float(centroids[label][1])],
            "bbox_px": {"x": x, "y": y, "width": w, "height": h},
            "area_px": area,
            "contours_px": contour_px,
            "mask_rle": {"encoding": "row-major alternating zero/one run lengths; starts with zero", "size": [h, w], "counts": _encode_mask_rle(component)},
            "center_camera_m": center_3d,
            "valid_depth_pixels": int(len(points)),
            "valid_3d": center_3d is not None,
        })
    return instances

def build_vision_result(timestamp_ms, masks, depth, intr, min_depth, max_depth, base_from_camera, obstacle_margin=.005):
    lesion_instances = _lesion_instances(masks.get("lesion"), depth, intr, min_depth, max_depth)
    obstacles = {}
    for name, mask in masks.items():
        if name == "lesion":
            continue
        if mask is None or not np.any(mask):
            obstacles[name] = {"valid": False, "reason": "no mask"}
            continue
        points, xs, ys = _points(mask, depth, intr, min_depth, max_depth)
        bounds = _bounds(points, base_from_camera, obstacle_margin)
        if bounds is None:
            obstacles[name] = {"valid": False, "reason": "fewer than 10 strict depth points"}
            continue
        obstacles[name] = {
            "valid": True,
            "center_px": [float(np.mean(xs)), float(np.mean(ys))],
            "center_camera_m": np.median(points, axis=0).tolist(),
            "bounds": bounds,
            "valid_depth_pixels": int(len(points)),
        }
    return {
        "schema_version": 2,
        "timestamp_ms": float(timestamp_ms),
        "units": "m",
        "camera_frame": "camera",
        "robot_base_frame": "robot_base" if base_from_camera is not None else None,
        "lesion": {"detected": bool(lesion_instances), "instances": lesion_instances},
        "obstacles": obstacles,
    }
