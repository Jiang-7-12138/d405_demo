from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_d405 import CameraIntrinsics
from io_utils import imwrite_unicode
from pointcloud import robust_mask_stats
from pointcloud_viewer import write_ply_ascii


@dataclass
class LesionResult:
    lesion_id: int
    row: int
    col: int
    pixel_xy: tuple[int, int]
    radius_px: int
    valid_3d: bool
    valid_points: int
    camera_xyz_m: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect the 3x3 red lesion targets in an offline D405 RGB-D reconstruction."
    )
    parser.add_argument("--npz", default=None, help="Offline reconstruction .npz. Defaults to latest captures/offline_recon/*.npz.")
    parser.add_argument("--intrinsics", default=None, help="Matching *_intrinsics.json. Defaults from --npz stem.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to the npz directory.")
    parser.add_argument("--stem", default=None, help="Output filename prefix. Defaults to npz stem.")
    parser.add_argument("--min-depth-mm", type=float, default=120.0)
    parser.add_argument("--max-depth-mm", type=float, default=700.0)
    parser.add_argument("--radius-scale", type=float, default=0.30, help="Lesion disk radius as a fraction of grid spacing.")
    parser.add_argument("--highlight-color", type=int, nargs=3, default=[255, 255, 0], metavar=("B", "G", "R"), help="BGR color for lesion points. Default cyan, not red.")
    parser.add_argument("--ply-coordinates", choices=["display", "camera"], default="display")
    parser.add_argument("--preview", action="store_true", help="Open highlighted PLY in Open3D after writing.")
    return parser.parse_args()


def latest_npz(base: Path) -> Path:
    candidates = sorted((base / "captures" / "offline_recon").glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise SystemExit("No offline reconstruction .npz found. Run capture_reconstruction.py first.")
    return candidates[0]


def load_intrinsics(path: Path) -> CameraIntrinsics:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CameraIntrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        coeffs=tuple(float(x) for x in data.get("coeffs", [])),
    )


def find_red_board(color_bgr: np.ndarray) -> tuple[int, int, int, int]:
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (((h <= 12) | (h >= 168)) & (s >= 35) & (v >= 60)).astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    contours, _ = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 2000]
    if not contours:
        raise SystemExit("Could not find the red 3x3 lesion board in the fused color image.")
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h_box = cv2.boundingRect(contour)
    return int(x), int(y), int(w), int(h_box)


def refine_center(color_bgr: np.ndarray, expected_xy: tuple[float, float], search_radius: int) -> tuple[int, int]:
    x0, y0 = expected_xy
    h_img, w_img = color_bgr.shape[:2]
    x1 = max(0, int(round(x0 - search_radius)))
    x2 = min(w_img, int(round(x0 + search_radius + 1)))
    y1 = max(0, int(round(y0 - search_radius)))
    y2 = min(h_img, int(round(y0 + search_radius + 1)))
    patch = color_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return int(round(x0)), int(round(y0))

    # Lesion domes/rings are local red high-saturation structures on a red base.
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((h <= 14) | (h >= 166)) & (s >= 35) & (v >= 45)
    local_dark = v.astype(np.int16) < (cv2.GaussianBlur(v, (0, 0), 9).astype(np.int16) - 3)
    local_sat = s.astype(np.int16) > (cv2.GaussianBlur(s, (0, 0), 9).astype(np.int16) + 2)
    candidate = (red & (local_dark | local_sat)).astype(np.uint8) * 255

    yy, xx = np.indices(candidate.shape)
    cx_local = x0 - x1
    cy_local = y0 - y1
    rr = (xx - cx_local) ** 2 + (yy - cy_local) ** 2
    candidate[rr > search_radius ** 2] = 0
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        best = None
        best_score = -1.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20:
                continue
            m = cv2.moments(contour)
            if not m["m00"]:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            dist = np.hypot(cx - cx_local, cy - cy_local)
            score = float(area - 2.0 * dist)
            if score > best_score:
                best_score = score
                best = (cx, cy)
        if best is not None:
            return int(round(x1 + best[0])), int(round(y1 + best[1]))
    return int(round(x0)), int(round(y0))


def make_lesion_masks(color_bgr: np.ndarray, radius_scale: float) -> tuple[list[np.ndarray], list[tuple[int, int]], tuple[int, int, int, int]]:
    x, y, w, h = find_red_board(color_bgr)
    # Fractions were chosen from the physical 3x3 lesion board layout. They are
    # robust to small board translations because the board bbox is detected.
    xs = [x + w * f for f in (0.29, 0.58, 0.86)]
    ys = [y + h * f for f in (0.22, 0.48, 0.73)]
    grid_spacing = float(min(np.diff(xs).mean(), np.diff(ys).mean()))
    radius = max(8, int(round(grid_spacing * radius_scale)))
    search_radius = max(radius + 8, int(round(grid_spacing * 0.42)))

    masks: list[np.ndarray] = []
    centers: list[tuple[int, int]] = []
    h_img, w_img = color_bgr.shape[:2]
    yy, xx = np.indices((h_img, w_img))
    for cy in ys:
        for cx in xs:
            refined = refine_center(color_bgr, (cx, cy), search_radius)
            centers.append(refined)
            mask = ((xx - refined[0]) ** 2 + (yy - refined[1]) ** 2) <= radius ** 2
            masks.append(mask)
    return masks, centers, (x, y, w, h)


def camera_points_to_display(points_m: np.ndarray) -> np.ndarray:
    display = points_m.astype(np.float32, copy=True)
    if display.size > 0:
        display[:, 1] *= -1.0
    return display


def build_colored_point_cloud(
    depth_m: np.ndarray,
    color_bgr: np.ndarray,
    intr: CameraIntrinsics,
    lesion_union: np.ndarray,
    highlight_bgr: tuple[int, int, int],
    min_depth_m: float,
    max_depth_m: float,
    ply_coordinates: str,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape
    ys, xs = np.indices((h, w))
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    u = xs[valid].astype(np.float32)
    v = ys[valid].astype(np.float32)
    z = depth_m[valid].astype(np.float32)
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    points = np.column_stack((x, y, z)).astype(np.float32)
    if ply_coordinates == "display":
        points = camera_points_to_display(points)
    colors = color_bgr[ys[valid], xs[valid]].astype(np.uint8)
    lesion_flat = lesion_union[valid]
    colors[lesion_flat] = np.array(highlight_bgr, dtype=np.uint8)
    return points, colors


def draw_overlay(color_bgr: np.ndarray, masks: list[np.ndarray], centers: list[tuple[int, int]], board_bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = color_bgr.copy()
    x, y, w, h = board_bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2, cv2.LINE_AA)
    for idx, (mask, center) in enumerate(zip(masks, centers), start=1):
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.drawContours(out, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(out, center, (255, 255, 0), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        label = f"lesion {idx}"
        x, y, w, h = cv2.boundingRect(mask_u8)
        tx = max(0, min(color_bgr.shape[1] - 1, x - 3))
        ty = max(18, y - 7)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 2)
        cv2.rectangle(out, (tx - 2, ty - th - 5), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    npz_path = Path(args.npz) if args.npz else latest_npz(base)
    if not npz_path.is_absolute():
        npz_path = base / npz_path
    intr_path = Path(args.intrinsics) if args.intrinsics else npz_path.with_name(f"{npz_path.stem}_intrinsics.json")
    if not intr_path.is_absolute():
        intr_path = base / intr_path
    out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or f"{npz_path.stem}_lesions"

    data = np.load(npz_path)
    color = data["color_bgr"]
    depth = data["export_depth_m"] if "export_depth_m" in data.files else data["depth_m"]
    intr = load_intrinsics(intr_path)

    masks, centers, board_bbox = make_lesion_masks(color, args.radius_scale)
    lesion_union = np.zeros(color.shape[:2], dtype=bool)
    for mask in masks:
        lesion_union |= mask

    results: list[LesionResult] = []
    for idx, (mask, center) in enumerate(zip(masks, centers), start=1):
        stats = robust_mask_stats(
            mask,
            depth,
            intr,
            min_depth_m=args.min_depth_mm / 1000.0,
            max_depth_m=args.max_depth_mm / 1000.0,
        )
        xyz = stats.median_camera_m if stats.valid else np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        row = (idx - 1) // 3 + 1
        col = (idx - 1) % 3 + 1
        results.append(LesionResult(idx, row, col, center, int(round(np.sqrt(np.count_nonzero(mask) / np.pi))), bool(stats.valid), int(stats.valid_points), xyz))

    overlay = draw_overlay(color, masks, centers, board_bbox)
    points, colors = build_colored_point_cloud(
        depth,
        color,
        intr,
        lesion_union,
        tuple(int(x) for x in args.highlight_color),
        args.min_depth_mm / 1000.0,
        args.max_depth_mm / 1000.0,
        args.ply_coordinates,
    )

    overlay_path = out_dir / f"{stem}_overlay.png"
    mask_path = out_dir / f"{stem}_mask.png"
    ply_path = out_dir / f"{stem}_highlight_cloud.ply"
    csv_path = out_dir / f"{stem}_centers_3d.csv"
    imwrite_unicode(str(overlay_path), overlay)
    imwrite_unicode(str(mask_path), lesion_union.astype(np.uint8) * 255)
    write_ply_ascii(str(ply_path), points, colors)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lesion_id", "row", "col", "pixel_x", "pixel_y", "valid_3d", "valid_points", "camera_x_m", "camera_y_m", "camera_z_m", "camera_x_mm", "camera_y_mm", "camera_z_mm"])
        for r in results:
            x, y, z = r.camera_xyz_m
            writer.writerow([
                r.lesion_id,
                r.row,
                r.col,
                r.pixel_xy[0],
                r.pixel_xy[1],
                int(r.valid_3d),
                r.valid_points,
                f"{x:.6f}",
                f"{y:.6f}",
                f"{z:.6f}",
                f"{x * 1000.0:.2f}",
                f"{y * 1000.0:.2f}",
                f"{z * 1000.0:.2f}",
            ])

    print(f"input_npz: {npz_path}")
    print(f"wrote {overlay_path}")
    print(f"wrote {ply_path}")
    print(f"wrote {csv_path}")
    for r in results:
        xyz_mm = r.camera_xyz_m * 1000.0
        print(
            f"lession {r.lesion_id}: px={r.pixel_xy} valid={r.valid_3d} pts={r.valid_points} "
            f"camera_mm=({xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f})"
        )

    if args.preview:
        try:
            import open3d as o3d  # type: ignore
        except Exception as exc:
            raise SystemExit("Open3D is not installed. Run: python -m pip install -r requirements-3d.txt") from exc
        pcd = o3d.io.read_point_cloud(str(ply_path))
        o3d.visualization.draw_geometries([pcd], window_name="Offline lesions highlighted")


if __name__ == "__main__":
    main()
