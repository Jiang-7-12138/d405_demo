from __future__ import annotations

import cv2
import numpy as np


def refine_mask_by_depth_band(
    mask: np.ndarray,
    depth_m: np.ndarray,
    band_m: float = 0.018,
    min_valid_pixels: int = 80,
) -> np.ndarray:
    """Keep mask pixels near the mask's median depth.

    This removes background/tray pixels that are connected in RGB but sit on a
    different depth layer. It is intentionally conservative and safe to disable
    until real D405 frames are available for tuning.
    """
    if mask is None or not np.any(mask):
        return mask
    valid = mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0)
    if int(np.count_nonzero(valid)) < min_valid_pixels:
        return mask.astype(bool)
    z_med = float(np.median(depth_m[valid]))
    refined = mask.astype(bool) & np.isfinite(depth_m) & (np.abs(depth_m - z_med) <= band_m)
    if int(np.count_nonzero(refined)) < min_valid_pixels:
        return mask.astype(bool)
    return _clean(refined, open_size=3, close_size=7)


def remove_dominant_plane(
    mask: np.ndarray,
    depth_m: np.ndarray,
    plane_band_m: float = 0.006,
    max_plane_fraction: float = 0.65,
) -> np.ndarray:
    """Remove a dominant flat depth layer from a mask.

    This is a lightweight substitute for full RANSAC plane fitting. For the tray
    and table, many connected false-positive pixels often share a very narrow
    depth band. If that band dominates the candidate, remove it.
    """
    if mask is None or not np.any(mask):
        return mask
    valid = mask.astype(bool) & np.isfinite(depth_m) & (depth_m > 0)
    zs = depth_m[valid]
    if zs.size < 100:
        return mask.astype(bool)
    hist, edges = np.histogram(zs, bins=48)
    idx = int(np.argmax(hist))
    mode_z = float((edges[idx] + edges[idx + 1]) * 0.5)
    plane = valid & (np.abs(depth_m - mode_z) <= plane_band_m)
    if np.count_nonzero(plane) / max(1, np.count_nonzero(mask)) > max_plane_fraction:
        refined = mask.astype(bool) & ~plane
        if np.count_nonzero(refined) >= 50:
            return _clean(refined, open_size=3, close_size=7)
    return mask.astype(bool)


def refine_all_masks_by_depth(masks: dict[str, np.ndarray], depth_m: np.ndarray) -> dict[str, np.ndarray]:
    refined: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        if name in {"kidney"}:
            refined[name] = mask
            continue
        out = refine_mask_by_depth_band(mask, depth_m)
        if name in {"bowel", "vessel", "gallbladder"}:
            out = remove_dominant_plane(out, depth_m)
        refined[name] = out
    return refined


def _clean(mask: np.ndarray, open_size: int, close_size: int) -> np.ndarray:
    out = mask.astype(np.uint8) * 255
    if open_size > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    if close_size > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    return out.astype(bool)
