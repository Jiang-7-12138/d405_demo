from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from camera_d405 import RealSenseD405
from io_utils import imwrite_unicode
from open3d_live_viewer import Open3DUnavailable, clean_point_cloud_open3d
from pointcloud import depth_to_points_sampled, fill_small_depth_holes
from pointcloud_viewer import write_ply_ascii


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture several D405 RGB-D frames and build a cleaner offline 3D reconstruction."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=60, help="Number of RGB-D frames used for temporal fusion.")
    parser.add_argument("--discard", type=int, default=15, help="Warmup frames discarded before capture.")
    parser.add_argument("--out-dir", default="captures/offline_recon")
    parser.add_argument("--stem", default=None, help="Output filename prefix. Defaults to timestamp.")
    parser.add_argument("--rs-filters", action="store_true", help="Enable RealSense spatial/temporal/hole-filling filters while capturing.")
    parser.add_argument("--reset-camera", action="store_true", help="Hardware-reset RealSense devices before starting capture.")
    parser.add_argument("--min-depth-mm", type=float, default=30.0)
    parser.add_argument("--max-depth-mm", type=float, default=900.0)
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.45,
        help="Keep a pixel only if this fraction of captured frames has valid depth there.",
    )
    parser.add_argument(
        "--max-depth-mad-mm",
        type=float,
        default=12.0,
        help="Reject pixels whose multi-frame depth median absolute deviation is larger than this, in mm.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Point cloud sampling stride for the saved PLY.")
    parser.add_argument("--fill-small-holes", action="store_true", help="Fill only small internal holes in the fused depth before exporting PLY.")
    parser.add_argument("--smooth-depth", action="store_true", help="Smooth fused depth with an edge-preserving bilateral filter before exporting PLY.")
    parser.add_argument("--smooth-sigma-mm", type=float, default=8.0, help="Depth-domain bilateral smoothing sigma in mm.")
    parser.add_argument("--smooth-sigma-px", type=float, default=5.0, help="Image-domain bilateral smoothing sigma in pixels.")
    parser.add_argument("--save-stack", action="store_true", help="Also save the full raw depth stack. This can create a large .npz file.")
    parser.add_argument("--clean", action="store_true", help="Write an additional Open3D-cleaned PLY.")
    parser.add_argument("--voxel-mm", type=float, default=1.0)
    parser.add_argument("--preview", action="store_true", help="Open the cleaned/raw PLY in Open3D after capture.")
    parser.add_argument(
        "--ply-coordinates",
        choices=["display", "camera"],
        default="display",
        help="Coordinate convention for saved PLY. 'display' flips RealSense +Y down to +Y up for Open3D viewing; 'camera' keeps raw RealSense camera coordinates.",
    )
    return parser.parse_args()


def masked_median_depth(depth_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(depth_stack) & (depth_stack > 0)
    valid_count = valid.sum(axis=0).astype(np.uint16)
    has_depth = valid_count > 0
    fused = np.zeros(depth_stack.shape[1:], dtype=np.float32)
    mad = np.zeros(depth_stack.shape[1:], dtype=np.float32)
    if np.any(has_depth):
        masked = np.where(valid[:, has_depth], depth_stack[:, has_depth], np.nan)
        fused_values = np.nanmedian(masked, axis=0).astype(np.float32)
        mad_values = np.nanmedian(np.abs(masked - fused_values[None, :]), axis=0).astype(np.float32)
        fused[has_depth] = np.where(np.isfinite(fused_values), fused_values, 0.0)
        mad[has_depth] = np.where(np.isfinite(mad_values), mad_values, 0.0)
    return fused, valid_count, mad


def smooth_depth_bilateral(depth_m: np.ndarray, sigma_depth_mm: float, sigma_space_px: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return depth_m
    filled = depth_m.astype(np.float32, copy=True)
    median_m = float(np.median(filled[valid]))
    filled[~valid] = median_m
    smoothed_mm = cv2.bilateralFilter(
        filled * 1000.0,
        d=0,
        sigmaColor=max(0.1, float(sigma_depth_mm)),
        sigmaSpace=max(0.1, float(sigma_space_px)),
    )
    smoothed = (smoothed_mm / 1000.0).astype(np.float32)
    smoothed[~valid] = 0.0
    return smoothed


def normalize_to_u8(values: np.ndarray, max_value: float | None = None) -> np.ndarray:
    if max_value is None:
        max_value = float(np.nanpercentile(values, 99)) if np.any(values) else 1.0
    max_value = max(max_value, 1e-6)
    return np.clip(values.astype(np.float32) / max_value * 255.0, 0, 255).astype(np.uint8)


def write_intrinsics(path: Path, intrinsics, capture_info: dict[str, object]) -> None:
    data = {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "coeffs": list(intrinsics.coeffs),
        **capture_info,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def camera_points_to_display(points_m: np.ndarray) -> np.ndarray:
    display = points_m.astype(np.float32, copy=True)
    if display.size > 0:
        display[:, 1] *= -1.0
    return display


def main() -> None:
    args = parse_args()
    out_dir = Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or time.strftime("recon_%Y%m%d_%H%M%S")

    devices = RealSenseD405.list_devices()
    if not devices:
        raise SystemExit("No RealSense device found.")
    print("RealSense devices:")
    for name, serial in devices:
        print(f"  {name} serial={serial}")

    if args.reset_camera:
        print("Resetting RealSense device. Wait a few seconds...")
        RealSenseD405.reset_devices()
        time.sleep(5.0)

    camera = RealSenseD405(args.width, args.height, args.fps, use_filters=args.rs_filters)
    camera.start()
    try:
        for _ in range(max(0, args.discard)):
            camera.read()

        depths: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        intrinsics = None
        print(f"Capturing {args.frames} frames. Keep camera and object still...")
        for i in range(max(1, args.frames)):
            frame = camera.read()
            depths.append(frame.depth_m.copy())
            colors.append(frame.color_bgr.copy())
            intrinsics = frame.intrinsics
            if (i + 1) % 10 == 0 or i == args.frames - 1:
                print(f"  {i + 1}/{args.frames}")
    finally:
        camera.stop()

    if intrinsics is None:
        raise SystemExit("No frames captured.")

    depth_stack = np.stack(depths, axis=0)
    color_stack = np.stack(colors, axis=0)
    fused_depth, valid_count, depth_mad = masked_median_depth(depth_stack)
    fused_color = np.median(color_stack, axis=0).clip(0, 255).astype(np.uint8)

    min_valid_count = max(1, int(np.ceil(max(0.0, min(1.0, args.min_valid_ratio)) * len(depths))))
    stable = valid_count >= min_valid_count
    if args.max_depth_mad_mm > 0:
        stable &= depth_mad <= (args.max_depth_mad_mm / 1000.0)
    fused_depth = np.where(stable, fused_depth, 0.0).astype(np.float32)

    export_depth = fused_depth
    if args.fill_small_holes:
        export_depth = fill_small_depth_holes(fused_depth)
    if args.smooth_depth:
        export_depth = smooth_depth_bilateral(export_depth, args.smooth_sigma_mm, args.smooth_sigma_px)

    points, point_colors = depth_to_points_sampled(
        export_depth,
        intrinsics,
        color_bgr=fused_color,
        stride=args.stride,
        min_depth_m=args.min_depth_mm / 1000.0,
        max_depth_m=args.max_depth_mm / 1000.0,
    )
    ply_points = camera_points_to_display(points) if args.ply_coordinates == "display" else points

    color_path = out_dir / f"{stem}_color.png"
    depth_vis_path = out_dir / f"{stem}_depth_vis.png"
    valid_path = out_dir / f"{stem}_valid_count.png"
    mad_path = out_dir / f"{stem}_depth_mad_mm.png"
    npz_path = out_dir / f"{stem}.npz"
    intr_path = out_dir / f"{stem}_intrinsics.json"
    ply_path = out_dir / f"{stem}_cloud.ply"

    clipped = np.clip(fused_depth, 0.0, args.max_depth_mm / 1000.0)
    depth_u8 = cv2.convertScaleAbs(clipped, alpha=255.0 / max(args.max_depth_mm / 1000.0, 1e-6))
    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_vis[fused_depth <= 0] = (0, 0, 0)

    imwrite_unicode(str(color_path), fused_color)
    imwrite_unicode(str(depth_vis_path), depth_vis)
    imwrite_unicode(str(valid_path), cv2.applyColorMap(normalize_to_u8(valid_count, args.frames), cv2.COLORMAP_VIRIDIS))
    imwrite_unicode(str(mad_path), cv2.applyColorMap(normalize_to_u8(depth_mad * 1000.0, 10.0), cv2.COLORMAP_MAGMA))

    if args.save_stack:
        np.savez_compressed(
            npz_path,
            color_bgr=fused_color,
            depth_m=fused_depth,
            export_depth_m=export_depth,
            valid_count=valid_count,
            depth_mad_m=depth_mad,
            depth_stack_m=depth_stack,
        )
    else:
        np.savez_compressed(
            npz_path,
            color_bgr=fused_color,
            depth_m=fused_depth,
            export_depth_m=export_depth,
            valid_count=valid_count,
            depth_mad_m=depth_mad,
        )

    write_intrinsics(
        intr_path,
        intrinsics,
        {
            "frames": int(args.frames),
            "discard": int(args.discard),
            "rs_filters": bool(args.rs_filters),
            "min_depth_mm": float(args.min_depth_mm),
            "max_depth_mm": float(args.max_depth_mm),
            "stride": int(args.stride),
            "fill_small_holes": bool(args.fill_small_holes),
            "smooth_depth": bool(args.smooth_depth),
            "smooth_sigma_mm": float(args.smooth_sigma_mm),
            "smooth_sigma_px": float(args.smooth_sigma_px),
            "min_valid_ratio": float(args.min_valid_ratio),
            "min_valid_count": int(min_valid_count),
            "max_depth_mad_mm": float(args.max_depth_mad_mm),
            "ply_coordinates": args.ply_coordinates,
            "camera_frame_note": "RealSense camera frame is X right, Y down, Z forward.",
            "display_frame_note": "Display/Open3D frame flips Y upward: X right, Y up, Z forward.",
        },
    )
    write_ply_ascii(str(ply_path), ply_points, point_colors)

    clean_path = None
    view_path = ply_path
    if args.clean and points.size > 0:
        try:
            clean_points, clean_colors = clean_point_cloud_open3d(
                ply_points,
                point_colors,
                voxel_size_m=args.voxel_mm / 1000.0,
                remove_outliers=True,
            )
            clean_path = out_dir / f"{stem}_cloud_clean.ply"
            write_ply_ascii(str(clean_path), clean_points, clean_colors)
            view_path = clean_path
        except Open3DUnavailable as exc:
            print(f"Open3D cleanup skipped: {exc}")

    print("Saved:")
    for path in [color_path, depth_vis_path, valid_path, mad_path, npz_path, intr_path, ply_path, clean_path]:
        if path is not None:
            print(f"  {path}")
    print(f"Point count: {points.shape[0]}")

    if args.preview and view_path.exists():
        try:
            import open3d as o3d  # type: ignore
        except Exception as exc:
            raise SystemExit("Open3D is not installed. Run: python -m pip install -r requirements-3d.txt") from exc
        pcd = o3d.io.read_point_cloud(str(view_path))
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        o3d.visualization.draw_geometries([pcd, axis], window_name="Offline D405 reconstruction")


if __name__ == "__main__":
    main()
