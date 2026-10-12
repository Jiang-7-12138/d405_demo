from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from camera_d405 import CameraIntrinsics, RealSenseD405
from checkerboard_calibration import estimate_checkerboard_world_pose
from io_utils import imwrite_unicode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one aligned D405 RGB-D frame and report all checkerboard corner depths."
    )
    parser.add_argument("--out-dir", default="captures/checkerboard", help="Output directory for this one-shot measurement.")
    parser.add_argument("--stem", default=None, help="Output filename prefix. Defaults to timestamp.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--discard", type=int, default=15, help="Warmup frames discarded before taking the single measurement.")
    parser.add_argument("--rs-filters", action="store_true", help="Enable RealSense depth filters for the captured frame.")
    parser.add_argument("--depth-window", type=int, default=5, help="Odd pixel window for robust depth sampling around each corner.")
    parser.add_argument("--min-depth-mm", type=float, default=50.0)
    parser.add_argument("--max-depth-mm", type=float, default=700.0)
    return parser.parse_args()


def camera_xyz_from_pixel(u: float, v: float, depth_m: float, intr: CameraIntrinsics) -> np.ndarray:
    return np.array(
        [(u - intr.cx) * depth_m / intr.fx, (v - intr.cy) * depth_m / intr.fy, depth_m],
        dtype=np.float64,
    )


def sample_corner_depth(
    depth_m: np.ndarray,
    u: float,
    v: float,
    window: int,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[float, int]:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    x, y = int(round(u)), int(round(v))
    y1, y2 = max(0, y - radius), min(depth_m.shape[0], y + radius + 1)
    x1, x2 = max(0, x - radius), min(depth_m.shape[1], x + radius + 1)
    patch = depth_m[y1:y2, x1:x2]
    valid = patch[np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)]
    if valid.size == 0:
        return float("nan"), 0
    return float(np.median(valid)), int(valid.size)


def analyze_frame(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    depth_window: int = 5,
    min_depth_m: float = 0.05,
    max_depth_m: float = 0.70,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pose = estimate_checkerboard_world_pose(color_bgr, intr)
    if not pose.valid or pose.corners is None or pose.t_world_camera is None:
        raise RuntimeError("Checkerboard not detected. Keep the complete 7x7 board visible, sharp, and glare-free, then retry.")

    corners = pose.corners.reshape(-1, 2)
    records: list[dict[str, object]] = []
    for index, (u, v) in enumerate(corners):
        row, col = divmod(index, 6)
        depth, valid_samples = sample_corner_depth(
            depth_m, float(u), float(v), depth_window, min_depth_m, max_depth_m
        )
        camera_xyz = camera_xyz_from_pixel(float(u), float(v), depth, intr) if np.isfinite(depth) else np.full(3, np.nan)
        known_world_xyz = np.array([col * 0.020, row * 0.020, 0.0], dtype=np.float64)
        if np.isfinite(depth):
            measured_world_xyz = (pose.t_world_camera @ np.r_[camera_xyz, 1.0])[:3]
        else:
            measured_world_xyz = np.full(3, np.nan)
        records.append({
            "corner_id": index,
            "row": row,
            "col": col,
            "pixel_x": float(u),
            "pixel_y": float(v),
            "known_world_x_m": float(known_world_xyz[0]),
            "known_world_y_m": float(known_world_xyz[1]),
            "known_world_z_m": float(known_world_xyz[2]),
            "depth_m": depth,
            "depth_valid_samples": valid_samples,
            "camera_x_m": float(camera_xyz[0]),
            "camera_y_m": float(camera_xyz[1]),
            "camera_z_m": float(camera_xyz[2]),
            "world_x_from_depth_m": float(measured_world_xyz[0]),
            "world_y_from_depth_m": float(measured_world_xyz[1]),
            "world_z_from_depth_m": float(measured_world_xyz[2]),
        })

    metadata = {
        "T_world_camera": pose.t_world_camera.tolist(),
        "reprojection_error_px": pose.reprojection_error_px,
        "inner_corners": [6, 6],
        "square_size_mm": 20.0,
        "depth_window_px": depth_window if depth_window % 2 else depth_window + 1,
        "depth_source": "D405 depth aligned to color; median of valid pixels around each corner",
        "intrinsics": {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.cx,
            "cy": intr.cy,
            "coeffs": list(intr.coeffs),
        },
    }
    return metadata, records


def main() -> None:
    args = parse_args()
    out_dir = Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or time.strftime("checkerboard_%Y%m%d_%H%M%S")

    camera = RealSenseD405(args.width, args.height, args.fps, use_filters=args.rs_filters)
    camera.start()
    try:
        for _ in range(max(0, args.discard)):
            camera.read()
        frame = camera.read()
    finally:
        camera.stop()

    metadata, records = analyze_frame(
        frame.color_bgr,
        frame.depth_m,
        frame.intrinsics,
        depth_window=args.depth_window,
        min_depth_m=args.min_depth_mm / 1000.0,
        max_depth_m=args.max_depth_mm / 1000.0,
    )

    overlay = frame.color_bgr.copy()
    corners = np.array([[[r["pixel_x"], r["pixel_y"]]] for r in records], dtype=np.float32)
    cv2.drawChessboardCorners(overlay, (6, 6), corners, True)
    for record in records:
        u, v = int(round(record["pixel_x"])), int(round(record["pixel_y"]))
        label = f'{record["corner_id"]}:{record["depth_m"] * 1000:.0f}mm' if np.isfinite(record["depth_m"]) else f'{record["corner_id"]}:NA'
        cv2.putText(overlay, label, (u + 3, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1, cv2.LINE_AA)

    color_path = out_dir / f"{stem}_color.png"
    overlay_path = out_dir / f"{stem}_corners_depth.png"
    depth_path = out_dir / f"{stem}_depth.npy"
    csv_path = out_dir / f"{stem}_corners.csv"
    json_path = out_dir / f"{stem}_T_world_camera.json"
    imwrite_unicode(str(color_path), frame.color_bgr)
    imwrite_unicode(str(overlay_path), overlay)
    np.save(depth_path, frame.depth_m)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    valid_depths = sum(np.isfinite(record["depth_m"]) for record in records)
    print(f"Detected {len(records)}/36 checkerboard corners; valid depth at {valid_depths}/36 corners")
    print(f"Reprojection RMS: {metadata['reprojection_error_px']:.3f} px")
    print(f"Saved corner/depth overlay: {overlay_path}")
    print(f"Saved per-corner pixel, depth, camera XYZ, and world coordinates: {csv_path}")
    print(f"Saved T_world_camera and intrinsics: {json_path}")
    print(f"Saved aligned raw depth: {depth_path}")
    for record in records:
        print(
            f'corner {record["corner_id"]:02d} pixel=({record["pixel_x"]:.2f},{record["pixel_y"]:.2f}) '
            f'depth={record["depth_m"] * 1000.0:.1f} mm samples={record["depth_valid_samples"]}'
        )


if __name__ == "__main__":
    main()
