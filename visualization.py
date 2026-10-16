from __future__ import annotations

import cv2
import numpy as np


def overlay_mask(color_bgr: np.ndarray, mask: np.ndarray, color=(0, 255, 255), alpha: float = 0.35) -> np.ndarray:
    out = color_bgr.copy()
    if mask is None or not np.any(mask):
        return out
    layer = np.zeros_like(out)
    layer[mask.astype(bool)] = color
    return cv2.addWeighted(out, 1.0, layer, alpha, 0.0)


def draw_labeled_mask(
    color_bgr: np.ndarray,
    mask: np.ndarray,
    label: str = "lesion",
    fill_color=(0, 255, 255),
    contour_color=(0, 0, 255),
    alpha: float = 0.28,
    center_px: tuple[int, int] | None = None,
) -> np.ndarray:
    out = overlay_mask(color_bgr, mask, color=fill_color, alpha=alpha)
    if mask is None or not np.any(mask):
        return out

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out

    cv2.drawContours(out, contours, -1, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.drawContours(out, contours, -1, contour_color, 2, cv2.LINE_AA)

    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    draw_lesion_size_circle(out, mask, center_px=center_px, color=contour_color)
    if center_px is not None:
        cx, cy = int(center_px[0]), int(center_px[1])
        cv2.drawMarker(out, (cx, cy), contour_color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    elif moments["m00"]:
        cx = int(round(moments["m10"] / moments["m00"]))
        cy = int(round(moments["m01"] / moments["m00"]))
        cv2.drawMarker(out, (cx, cy), contour_color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    else:
        x, y, w, h = cv2.boundingRect(largest)
        cx, cy = x + w // 2, y + h // 2

    x, y, w, h = cv2.boundingRect(largest)
    text_x = max(0, min(color_bgr.shape[1] - 1, x))
    text_y = max(22, y - 8)
    text = f"{label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(out, (text_x - 3, text_y - th - 7), (text_x + tw + 5, text_y + 5), (0, 0, 0), -1)
    cv2.putText(out, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def lesion_area_radius_px(mask: np.ndarray) -> int | None:
    """Radius of a circle with the same area as the detected lesion mask."""
    if mask is None or not np.any(mask):
        return None
    area = float(np.count_nonzero(mask))
    if area < 1.0:
        return None
    return max(4, int(round(np.sqrt(area / np.pi))))


def draw_lesion_size_circle(
    image: np.ndarray,
    mask: np.ndarray,
    center_px: tuple[int, int] | None = None,
    color: tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Draw a size circle. center_px should be the depth-based point when available."""
    radius = lesion_area_radius_px(mask)
    if radius is None:
        return
    if center_px is None:
        center_px = mask_centroid(mask)
    if center_px is None:
        return
    center = (int(center_px[0]), int(center_px[1]))
    cv2.circle(image, center, radius, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.circle(image, center, radius, color, 2, cv2.LINE_AA)


def smooth_closed_contour(contour: np.ndarray, radius: float = 18.0) -> np.ndarray:
    """Round a closed outline so ridges on the colon do not become a sawtooth."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 8:
        return contour
    closed = np.vstack([pts, pts[:1]])
    segment = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    distance = np.concatenate([[0.0], np.cumsum(segment)])
    total = float(distance[-1])
    if total < radius * 4:
        return contour
    samples = np.arange(0.0, total, 2.0)
    xs = np.interp(samples, distance, closed[:, 0])
    ys = np.interp(samples, distance, closed[:, 1])
    win = max(3, int(round(radius)) | 1)
    pad = win // 2
    kernel = np.ones(win, dtype=np.float32) / float(win)

    def circular_mean(values: np.ndarray) -> np.ndarray:
        extended = np.concatenate([values[-pad:], values, values[:pad]])
        return np.convolve(extended, kernel, mode="valid")[: len(values)]

    smoothed = np.stack([circular_mean(xs), circular_mean(ys)], axis=1)
    return np.round(smoothed).astype(np.int32).reshape(-1, 1, 2)


def mask_interior_point(mask: np.ndarray, edge_margin: int = 36) -> tuple[int, int] | None:
    """A point on the object itself, not in the hollow or on a cropped image edge."""
    if mask is None or not np.any(mask):
        return None
    search = mask.astype(bool)
    if edge_margin > 0 and min(search.shape) > edge_margin * 2:
        inset = search.copy()
        inset[:edge_margin, :] = False
        inset[-edge_margin:, :] = False
        inset[:, :edge_margin] = False
        inset[:, -edge_margin:] = False
        if np.any(inset):
            search = inset
    dist = cv2.distanceTransform(search.astype(np.uint8), cv2.DIST_L2, 5)
    _min_val, _max_val, _min_loc, max_loc = cv2.minMaxLoc(dist)
    return int(max_loc[0]), int(max_loc[1])


def mask_centroid(mask: np.ndarray) -> tuple[int, int] | None:
    if mask is None or not np.any(mask):
        return None
    moments = cv2.moments(mask.astype(np.uint8) * 255)
    if not moments["m00"]:
        return None
    cx = int(round(moments["m10"] / moments["m00"]))
    cy = int(round(moments["m01"] / moments["m00"]))
    return cx, cy


def draw_named_masks(
    color_bgr: np.ndarray,
    masks: dict[str, np.ndarray],
    colors: dict[str, tuple[int, int, int]] | None = None,
    centers: dict[str, tuple[int, int] | None] | None = None,
    label_names: set[str] | None = None,
    center_names: set[str] | None = None,
    alpha: float = 0.22,
) -> np.ndarray:
    out = color_bgr.copy()
    default_colors = {
        "liver": (0, 170, 255),
        "kidney": (180, 80, 255),
        "bowel": (255, 130, 80),
        "omentum": (0, 220, 255),
        "gallbladder": (0, 220, 0),
        "vessel": (255, 0, 0),
        "lesion": (0, 0, 255),
    }
    if colors:
        default_colors.update(colors)

    for name, mask in masks.items():
        if mask is None or not np.any(mask):
            continue
        color = default_colors.get(name, (255, 255, 255))
        layer = np.zeros_like(out)
        layer[mask.astype(bool)] = color
        out = cv2.addWeighted(out, 1.0, layer, alpha, 0.0)

        mask_u8 = mask.astype(np.uint8) * 255
        contour_method = cv2.CHAIN_APPROX_NONE if name == "lesion" else cv2.CHAIN_APPROX_SIMPLE
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, contour_method)
        if name != "lesion":
            contours = [c for c in contours if cv2.contourArea(c) >= 40]
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        if name == "lesion":
            anchor = None
            if centers is not None:
                anchor = centers.get(name)
            draw_lesion_size_circle(out, mask, center_px=anchor, color=color)
        else:
            drawn = [smooth_closed_contour(c, radius=42.0) if name == "bowel" else c for c in contours]
            cv2.drawContours(out, drawn, -1, color, 2, cv2.LINE_AA)

        if label_names is not None and name not in label_names:
            continue
        x, y, w, h = cv2.boundingRect(largest)
        show_coordinates = center_names is None or name in center_names
        if show_coordinates:
            centroid = centers.get(name) if centers is not None else mask_centroid(mask)
            if centroid is None:
                centroid = (x + w // 2, y + h // 2)
            cx, cy = centroid
            cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            label = f"{name} ({cx},{cy})"
            text_x = max(0, min(color_bgr.shape[1] - 1, cx + 8))
            text_y = max(18, cy - 8)
        else:
            label = name
            if name == "bowel":
                held = centers.get(name) if centers is not None else None
                label_at = held if held is not None else mask_interior_point(mask)
            else:
                label_at = None
            if label_at is None:
                moments = cv2.moments(largest)
                if moments["m00"]:
                    label_at = (int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"])))
                else:
                    label_at = (x + w // 2, y + h // 2)
            text_x = int(label_at[0]) + 8
            text_y = max(18, int(label_at[1]) - 8)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        if name == "bowel" and not show_coordinates:
            text_x = int(label_at[0]) - tw // 2
            text_y = int(label_at[1]) + th // 2
        text_x = max(0, min(color_bgr.shape[1] - tw - 4, text_x))
        text_y = max(th + 6, min(color_bgr.shape[0] - 6, text_y))
        cv2.rectangle(out, (text_x - 3, text_y - th - 6), (text_x + tw + 4, text_y + 5), (0, 0, 0), -1)
        cv2.putText(out, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_text_lines(img: np.ndarray, lines: list[str], x: int = 10, y: int = 24) -> np.ndarray:
    out = img.copy()
    for i, line in enumerate(lines):
        yy = y + i * 22
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def hstack_resize(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.hstack([left, right])


def colorize_validity(depth_m: np.ndarray, min_depth_m: float = 0.03, max_depth_m: float = 0.45) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    out = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    out[valid] = (60, 200, 60)
    out[~valid] = (40, 40, 40)
    return out
