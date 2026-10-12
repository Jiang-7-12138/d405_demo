from __future__ import annotations

import cv2
import numpy as np


def draw_pointcloud_views(
    points_m: np.ndarray,
    size: tuple[int, int] = (480, 640),
    highlight_points_m: np.ndarray | None = None,
    highlight_center_m: np.ndarray | None = None,
) -> np.ndarray:
    h, w = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if points_m.size == 0:
        cv2.putText(canvas, "no valid points", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        return canvas

    left = np.zeros((h, w // 2, 3), dtype=np.uint8)
    right = np.zeros((h, w - w // 2, 3), dtype=np.uint8)

    _draw_projection(left, points_m[:, 0], points_m[:, 2], "top: X-Z", "X mm", "Z mm")
    _draw_projection(right, points_m[:, 0], -points_m[:, 1], "front: X-Y", "X mm", "Y mm")
    if highlight_points_m is not None and highlight_points_m.size > 0:
        _draw_projection_points(left, points_m[:, 0], points_m[:, 2], highlight_points_m[:, 0], highlight_points_m[:, 2], (0, 0, 255), radius=1)
        _draw_projection_points(right, points_m[:, 0], -points_m[:, 1], highlight_points_m[:, 0], -highlight_points_m[:, 1], (0, 0, 255), radius=1)
    if highlight_center_m is not None and highlight_center_m.size == 3 and np.all(np.isfinite(highlight_center_m)):
        center = highlight_center_m.reshape(1, 3)
        _draw_projection_points(left, points_m[:, 0], points_m[:, 2], center[:, 0], center[:, 2], (0, 0, 255), radius=6)
        _draw_projection_points(right, points_m[:, 0], -points_m[:, 1], center[:, 0], -center[:, 1], (0, 0, 255), radius=6)
    return np.hstack([left, right])


def draw_3d_reconstruction(
    points_m: np.ndarray,
    colors_bgr: np.ndarray | None = None,
    size: tuple[int, int] = (480, 640),
    highlight_points_m: np.ndarray | None = None,
    highlight_center_m: np.ndarray | None = None,
    title: str = "3D reconstruction results",
    point_radius: int = 1,
    fixed_span_m: float | None = None,
    background: str = "blue",
) -> np.ndarray:
    """Render a camera-frame point cloud into a 3D-looking preview panel.

    This does not replace the interactive Open3D window. It gives the main
    OpenCV dashboard a stable, screenshot-friendly view similar to stereo
    reconstruction demos: colored points in a real metric coordinate frame,
    with a floor grid and XYZ axes for orientation.
    """
    h, w = size
    canvas = _white_background(h, w) if background == "white" else _blue_gradient(h, w)
    dark_text = background == "white"
    text_color = (35, 35, 35) if dark_text else (245, 245, 245)
    sub_text_color = (85, 85, 85) if dark_text else (220, 220, 220)
    if points_m.size == 0:
        cv2.putText(canvas, "no valid 3D points", (22, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        return canvas

    display_points = _camera_to_display(points_m)
    center = np.median(display_points, axis=0)
    spans = np.percentile(display_points, 99, axis=0) - np.percentile(display_points, 1, axis=0)
    scene_span = float(max(fixed_span_m if fixed_span_m is not None else np.max(spans), 0.08))
    scale = 0.78 * min(w, h) / scene_span
    rot = _view_rotation(yaw_deg=-34.0, pitch_deg=-20.0)

    def project(p_display: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = (p_display - center) @ rot.T
        xy = np.column_stack((w * 0.52 + q[:, 0] * scale, h * 0.56 - q[:, 1] * scale))
        return xy, q[:, 2]

    _draw_metric_grid(canvas, center, scene_span, rot, scale, light=dark_text)
    _draw_axes(canvas, center, scene_span, rot, scale)

    pts = display_points
    cols = colors_bgr
    if pts.shape[0] > 55000:
        idx = np.linspace(0, pts.shape[0] - 1, 55000).astype(np.int64)
        pts = pts[idx]
        if cols is not None:
            cols = cols[idx]

    xy, depth = project(pts)
    order = np.argsort(depth)
    xy = xy[order]
    if cols is not None:
        cols = cols[order]
    else:
        z = points_m[:, 2]
        if z.shape[0] != xy.shape[0]:
            z = z[np.linspace(0, z.shape[0] - 1, xy.shape[0]).astype(np.int64)]
        zn = (z - np.nanmin(z)) / max(1e-6, float(np.nanmax(z) - np.nanmin(z)))
        cols = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
        cols = cols[order]

    ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    xy_i = xy[ok].astype(np.int32)
    cols_i = cols[ok]
    radius = max(0, int(point_radius))
    if radius <= 0:
        for (x, y), c in zip(xy_i, cols_i):
            canvas[y, x] = tuple(int(v) for v in c)
    else:
        for (x, y), c in zip(xy_i, cols_i):
            cv2.circle(canvas, (int(x), int(y)), radius, tuple(int(v) for v in c), -1, cv2.LINE_AA)

    if highlight_points_m is not None and highlight_points_m.size > 0:
        hp = _camera_to_display(highlight_points_m)
        hxy, _ = project(hp)
        for x, y in hxy.astype(np.int32):
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(canvas, (int(x), int(y)), 2, (0, 0, 255), -1, cv2.LINE_AA)

    if highlight_center_m is not None and highlight_center_m.size == 3 and np.all(np.isfinite(highlight_center_m)):
        cpt = _camera_to_display(highlight_center_m.reshape(1, 3))
        cxy, _ = project(cpt)
        x, y = cxy[0].astype(np.int32)
        if 0 <= x < w and 0 <= y < h:
            cv2.drawMarker(canvas, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

    cv2.putText(canvas, title, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "fixed metric camera frame: X right, Y up, Z forward", (14, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, sub_text_color, 1, cv2.LINE_AA)
    return canvas


def _draw_projection(img: np.ndarray, a_m: np.ndarray, b_m: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    h, w = img.shape[:2]
    margin = 28
    a_mm = a_m * 1000.0
    b_mm = b_m * 1000.0
    a_min, a_max = _robust_range(a_mm)
    b_min, b_max = _robust_range(b_mm)

    px = ((a_mm - a_min) / max(1e-6, a_max - a_min) * (w - 2 * margin) + margin).astype(np.int32)
    py = (h - margin - (b_mm - b_min) / max(1e-6, b_max - b_min) * (h - 2 * margin)).astype(np.int32)
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px = px[ok]
    py = py[ok]
    if px.size > 8000:
        idx = np.linspace(0, px.size - 1, 8000).astype(np.int32)
        px = px[idx]
        py = py[idx]

    img[py, px] = (0, 220, 255)
    cv2.rectangle(img, (margin, margin), (w - margin, h - margin), (80, 80, 80), 1)
    cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, f"{xlabel}: {a_min:.0f}..{a_max:.0f}", (10, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(img, f"{ylabel}: {b_min:.0f}..{b_max:.0f}", (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1, cv2.LINE_AA)


def _draw_projection_points(
    img: np.ndarray,
    base_a_m: np.ndarray,
    base_b_m: np.ndarray,
    a_m: np.ndarray,
    b_m: np.ndarray,
    color: tuple[int, int, int],
    radius: int = 1,
) -> None:
    h, w = img.shape[:2]
    margin = 28
    base_a_mm = base_a_m * 1000.0
    base_b_mm = base_b_m * 1000.0
    a_min, a_max = _robust_range(base_a_mm)
    b_min, b_max = _robust_range(base_b_mm)

    a_mm = a_m * 1000.0
    b_mm = b_m * 1000.0
    px = ((a_mm - a_min) / max(1e-6, a_max - a_min) * (w - 2 * margin) + margin).astype(np.int32)
    py = (h - margin - (b_mm - b_min) / max(1e-6, b_max - b_min) * (h - 2 * margin)).astype(np.int32)
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px = px[ok]
    py = py[ok]
    if px.size > 10000:
        idx = np.linspace(0, px.size - 1, 10000).astype(np.int32)
        px = px[idx]
        py = py[idx]
    for x, y in zip(px, py):
        cv2.circle(img, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)


def _robust_range(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(values, [1, 99])
    if hi - lo < 1.0:
        mid = float((hi + lo) / 2.0)
        return mid - 0.5, mid + 0.5
    pad = float((hi - lo) * 0.08)
    return float(lo - pad), float(hi + pad)


def _blue_gradient(h: int, w: int) -> np.ndarray:
    top = np.array([56, 92, 128], dtype=np.float32)
    bottom = np.array([15, 35, 58], dtype=np.float32)
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    col = (top * (1.0 - t) + bottom * t).astype(np.uint8)
    return np.repeat(col[:, None, :], w, axis=1)


def _white_background(h: int, w: int) -> np.ndarray:
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _camera_to_display(points_m: np.ndarray) -> np.ndarray:
    points = points_m.astype(np.float32, copy=True)
    points[:, 1] *= -1.0
    return points


def _view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    ry = np.array([
        [np.cos(yaw), 0.0, np.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-np.sin(yaw), 0.0, np.cos(yaw)],
    ], dtype=np.float32)
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(pitch), -np.sin(pitch)],
        [0.0, np.sin(pitch), np.cos(pitch)],
    ], dtype=np.float32)
    return rx @ ry


def _project_display(points: np.ndarray, center: np.ndarray, rot: np.ndarray, scale: float, w: int, h: int) -> np.ndarray:
    q = (points - center) @ rot.T
    return np.column_stack((w * 0.52 + q[:, 0] * scale, h * 0.56 - q[:, 1] * scale))


def _draw_metric_grid(canvas: np.ndarray, center: np.ndarray, scene_span: float, rot: np.ndarray, scale: float, light: bool = False) -> None:
    h, w = canvas.shape[:2]
    floor_y = center[1] - scene_span * 0.38
    step = _nice_step(scene_span / 5.0)
    extent = step * 4
    color = (205, 205, 205) if light else (92, 118, 138)
    xs = np.arange(center[0] - extent, center[0] + extent + step * 0.5, step)
    zs = np.arange(center[2] - extent, center[2] + extent + step * 0.5, step)
    for x in xs:
        pts = np.array([[x, floor_y, zs[0]], [x, floor_y, zs[-1]]], dtype=np.float32)
        xy = _project_display(pts, center, rot, scale, w, h).astype(np.int32)
        cv2.line(canvas, tuple(xy[0]), tuple(xy[1]), color, 1, cv2.LINE_AA)
    for z in zs:
        pts = np.array([[xs[0], floor_y, z], [xs[-1], floor_y, z]], dtype=np.float32)
        xy = _project_display(pts, center, rot, scale, w, h).astype(np.int32)
        cv2.line(canvas, tuple(xy[0]), tuple(xy[1]), color, 1, cv2.LINE_AA)


def _draw_axes(canvas: np.ndarray, center: np.ndarray, scene_span: float, rot: np.ndarray, scale: float) -> None:
    h, w = canvas.shape[:2]
    axis_len = max(_nice_step(scene_span / 3.0), 0.03)
    origin = center + np.array([-scene_span * 0.42, -scene_span * 0.30, -scene_span * 0.32], dtype=np.float32)
    axes = [
        (np.array([axis_len, 0.0, 0.0], dtype=np.float32), (30, 70, 240), "X"),
        (np.array([0.0, axis_len, 0.0], dtype=np.float32), (40, 210, 70), "Y"),
        (np.array([0.0, 0.0, axis_len], dtype=np.float32), (230, 130, 40), "Z"),
    ]
    for vec, color, label in axes:
        pts = np.array([origin, origin + vec], dtype=np.float32)
        xy = _project_display(pts, center, rot, scale, w, h).astype(np.int32)
        cv2.arrowedLine(canvas, tuple(xy[0]), tuple(xy[1]), color, 2, cv2.LINE_AA, tipLength=0.18)
        cv2.putText(canvas, label, tuple(xy[1] + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def _nice_step(value: float) -> float:
    if value <= 0:
        return 0.05
    exp = np.floor(np.log10(value))
    frac = value / (10 ** exp)
    if frac < 1.5:
        nice = 1.0
    elif frac < 3.5:
        nice = 2.0
    elif frac < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10 ** exp))


def write_ply_ascii(path: str, points_m: np.ndarray, colors_bgr: np.ndarray | None = None) -> None:
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points_m.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors_bgr is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors_bgr is None:
            for p in points_m:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(points_m, colors_bgr):
                b, g, r = int(c[0]), int(c[1]), int(c[2])
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")
