from __future__ import annotations

import argparse
import ctypes
import csv
import json
from collections import deque
import os
import time
from pathlib import Path

import cv2
import numpy as np

from calibration import create_aruco_marker_png, draw_aruco_axis, estimate_aruco_world_pose, transform_points
from camera_d405 import RealSenseD405
from capture_reconstruction import masked_median_depth, smooth_depth_bilateral
from checkerboard_calibration import (
    draw_checkerboard_pose,
    estimate_checkerboard_world_pose,
    save_world_camera_transform,
)
from detect_offline_lesions import draw_overlay as draw_offline_lesion_overlay, make_lesion_masks
from depth_refine import refine_all_masks_by_depth
from io_utils import imwrite_unicode
from open3d_live_viewer import Open3DLiveViewer, Open3DUnavailable, clean_point_cloud_open3d
from pointcloud import depth_to_points_sampled, fill_small_depth_holes, mask_to_points, pixel_to_camera_xyz, robust_mask_stats, roi_mask
from pointcloud_viewer import draw_3d_reconstruction, write_ply_ascii
from robot_frame import camera_point_to_robot_base, load_transform_json
from realtime_geometry import build_vision_result
from segmenter import HSVSegmenter, PhantomLesionSegmenter, PhantomMaterialMaskSegmenter, PhantomMaterialSegmenter
from visualization import draw_labeled_mask, draw_named_masks, draw_text_lines, hstack_resize, mask_centroid, overlay_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intel RealSense D405 RGB-D demo for minimally invasive robot vision.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--record-dir", default="captures", help="Directory for screenshots, npz frames and csv logs.")
    parser.add_argument("--aruco", action="store_true", help="Estimate temporary World pose from a 15 mm ArUco marker id 0.")
    parser.add_argument("--marker-size-mm", type=float, default=15.0)
    parser.add_argument(
        "--chessboard-world",
        action="store_true",
        help="Use the 7x7, 20 mm checkerboard as World and estimate T_world_camera from its 6x6 inner corners.",
    )
    parser.add_argument("--min-depth-mm", type=float, default=30.0, help="Minimum valid depth for ROI/mask statistics.")
    parser.add_argument("--max-depth-mm", type=float, default=700.0, help="Maximum valid depth for ROI/mask statistics.")
    parser.add_argument("--pc-min-depth-mm", type=float, default=None, help="Minimum depth for displayed point cloud. Defaults to --min-depth-mm.")
    parser.add_argument("--pc-max-depth-mm", type=float, default=None, help="Maximum depth for displayed point cloud. Defaults to --max-depth-mm.")
    parser.add_argument("--make-marker", action="store_true", help="Create aruco_marker_id0.png and exit.")
    parser.add_argument("--pc-stride", type=int, default=4, help="Point cloud sampling stride. Smaller is denser but slower.")
    parser.add_argument("--recon-point-radius", type=int, default=1, help="Point radius for the dashboard 3D reconstruction preview.")
    parser.add_argument(
        "--pc-color-mode",
        choices=["rgb", "depth", "none"],
        default="rgb",
        help="Point cloud color source: RGB texture, depth colormap, or no color.",
    )
    parser.add_argument("--rs-filters", action="store_true", help="Enable RealSense spatial/temporal/hole-filling depth filters.")
    parser.add_argument("--depth-temporal-frames", type=int, default=20, help="Median-fuse this many recent depth frames for display and 3D measurements.")
    parser.add_argument("--depth-min-valid-ratio", type=float, default=0.35, help="Keep fused depth only where this fraction of buffered frames is valid for measurements.")
    parser.add_argument("--depth-max-mad-mm", type=float, default=15.0, help="Reject fused measurement pixels whose temporal depth MAD exceeds this value.")
    parser.add_argument("--depth-display-min-valid-ratio", type=float, default=0.20, help="Valid-frame ratio used only for the preview point cloud.")
    parser.add_argument("--depth-display-max-mad-mm", type=float, default=25.0, help="Temporal MAD used only for the preview point cloud.")
    parser.add_argument("--depth-smooth", action="store_true", help="Apply edge-preserving bilateral smoothing to fused depth.")
    parser.add_argument("--depth-smooth-sigma-mm", type=float, default=8.0, help="Depth sigma for bilateral smoothing.")
    parser.add_argument("--display-hole-size", type=int, default=900, help="Largest display-only depth hole area to inpaint in pixels.")
    parser.add_argument(
        "--coordinate-depth-source",
        choices=["strict", "display"],
        default="display",
        help="Depth map used for target coordinate output. 'display' is less holey; use 'strict' for rawer measurements.",
    )
    parser.add_argument("--recon-fixed-span-mm", type=float, default=360.0, help="Fixed 3D preview span in millimeters; keeps display scale stable as distance changes.")
    parser.add_argument("--reset-camera", action="store_true", help="Hardware-reset RealSense devices before starting the live demo.")
    parser.add_argument("--open3d", action="store_true", help="Show a true interactive 3D point cloud window with Open3D.")
    parser.add_argument("--open3d-clean", action="store_true", help="Clean Open3D display/save point clouds with voxel downsampling and outlier removal.")
    parser.add_argument("--voxel-mm", type=float, default=1.5, help="Voxel size for Open3D cleanup, in millimeters.")
    parser.add_argument("--use-depth-refine", action="store_true", help="Refine all class masks using D405 depth layers.")
    parser.add_argument("--base-from-camera", default=None, help="JSON file containing T_base_camera for robot-base output.")
    parser.add_argument("--vision-output-jsonl", default="vision_detections.jsonl", help="Append automatic vision results here; empty disables file output.")
    parser.add_argument("--vision-output-hz", type=float, default=5.0, help="Maximum vision output rate.")
    parser.add_argument("--obstacle-margin-mm", type=float, default=5.0, help="Inflation added to each organ 3D bound.")
    parser.add_argument(
        "--mask-mode",
        choices=["none", "roi", "hsv", "lesion", "offline-lesions", "liver", "kidney", "bowel", "omentum", "gallbladder", "vessel", "all"],
        default="all",
    )
    parser.add_argument("--lesion-hue1", type=int, nargs=2, default=[0, 14], metavar=("LOW", "HIGH"))
    parser.add_argument("--lesion-hue2", type=int, nargs=2, default=[155, 179], metavar=("LOW", "HIGH"))
    parser.add_argument("--lesion-s-min", type=int, default=35)
    parser.add_argument("--lesion-v-min", type=int, default=55)
    parser.add_argument("--lesion-v-max", type=int, default=255)
    parser.add_argument("--lesion-min-area", type=float, default=120.0)
    parser.add_argument("--lesion-max-area", type=float, default=90000.0)
    parser.add_argument("--lesion-min-circularity", type=float, default=0.12)
    parser.add_argument("--offline-lesion-radius-scale", type=float, default=0.30, help="Radius of each offline 3x3 lesion region as a fraction of grid spacing.")
    parser.add_argument(
        "--lesion-profile",
        choices=["auto", "official", "pink", "pink-real", "painted"],
        default="painted",
        help="Lesion detector profile. Painted targets the current brown-red raised lesion; use auto/official/pink for older phantoms.",
    )
    parser.add_argument(
        "--material-profile",
        choices=["legacy", "painted"],
        default="painted",
        help="Material thresholds for --mask-mode all/liver/kidney/bowel/omentum/gallbladder/vessel.",
    )
    parser.add_argument(
        "--coordinate-classes",
        default="all",
        help="Comma-separated classes used for 3D coordinate output in --mask-mode all. Default: all.",
    )
    return parser.parse_args()


class DemoState:
    def __init__(self) -> None:
        self.mouse_xy: tuple[int, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_end: tuple[int, int] | None = None
        self.roi_mask: np.ndarray | None = None
        self.mask_mode = "none"
        self.freeze_mask = False
        self.last_mask: np.ndarray | None = None
        self.last_world_pose = None
        self.depth_history = deque(maxlen=5)


def save_csv_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s", "frame_timestamp_ms", "mode", "valid", "points", "valid_ratio",
            "camera_x_m", "camera_y_m", "camera_z_m", "world_x_m", "world_y_m", "world_z_m",
        ])


def save_entities_csv_header(path: Path) -> None:
    expected = [
        "time_s", "frame_timestamp_ms", "class", "valid", "confidence", "mask_pixels",
        "valid_depth_points", "valid_depth_ratio", "pixel_x", "pixel_y",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "area_px", "geometry",
        "camera_x_m", "camera_y_m", "camera_z_m", "world_x_m", "world_y_m", "world_z_m",
        "base_x_m", "base_y_m", "base_z_m",
    ]
    if path.exists():
        first = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:1]
        if first and first[0].split(",") == expected:
            return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(expected)


def append_stats(path: Path, timestamp_ms: float, mode: str, stats, world_xyz) -> None:
    cam = stats.median_camera_m if stats.valid else [np.nan, np.nan, np.nan]
    world = world_xyz if world_xyz is not None else [np.nan, np.nan, np.nan]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{time.time():.3f}", f"{timestamp_ms:.3f}", mode, int(stats.valid), stats.valid_points,
            f"{stats.valid_ratio:.4f}", f"{cam[0]:.6f}", f"{cam[1]:.6f}", f"{cam[2]:.6f}",
            f"{world[0]:.6f}", f"{world[1]:.6f}", f"{world[2]:.6f}",
        ])


def append_offline_lesion_stats(path: Path, timestamp_ms: float, centers, lesion_stats, world_pose, base_from_camera) -> None:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "time_s", "frame_timestamp_ms", "lesion_id", "row", "col", "pixel_x", "pixel_y",
                "valid_3d", "valid_depth_points", "camera_x_m", "camera_y_m", "camera_z_m",
                "world_x_m", "world_y_m", "world_z_m", "base_x_m", "base_y_m", "base_z_m",
            ])

    now = f"{time.time():.3f}"
    rows = []
    for lesion_id, stats in lesion_stats:
        center = centers[lesion_id - 1]
        nan_xyz = [np.nan, np.nan, np.nan]
        camera_xyz = stats.median_camera_m if stats.valid else nan_xyz
        world_xyz = transform_points(world_pose, camera_xyz.reshape(1, 3))[0] if stats.valid and world_pose is not None else nan_xyz
        base_xyz = camera_point_to_robot_base(camera_xyz, base_from_camera) if stats.valid and base_from_camera is not None else nan_xyz
        rows.append([
            now, f"{timestamp_ms:.3f}", lesion_id, (lesion_id - 1) // 3 + 1, (lesion_id - 1) % 3 + 1,
            center[0], center[1], int(stats.valid), stats.valid_points,
            *[f"{value:.6f}" for value in camera_xyz],
            *[f"{value:.6f}" for value in world_xyz],
            *[f"{value:.6f}" for value in base_xyz],
        ])
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def mask_geometry(mask: np.ndarray) -> dict[str, object]:
    if mask is None or not np.any(mask):
        return {
            "center_px": None,
            "bbox": (0, 0, 0, 0),
            "area_px": 0,
            "geometry": "empty",
            "confidence": 0.0,
        }
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 20]
    if not contours:
        return {
            "center_px": None,
            "bbox": (0, 0, 0, 0),
            "area_px": int(np.count_nonzero(mask)),
            "geometry": "mask",
            "confidence": 0.2,
        }
    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    x, y, w, h = cv2.boundingRect(largest)
    moments = cv2.moments(largest)
    center = None
    if moments["m00"]:
        center = (int(round(moments["m10"] / moments["m00"])), int(round(moments["m01"] / moments["m00"])))
    fill = area / max(1.0, float(w * h))
    perimeter = float(cv2.arcLength(largest, True))
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 1.0 else 0.0
    if circularity > 0.45 and 0.6 <= (w / max(1.0, h)) <= 1.7:
        shape = "blob"
    elif max(w, h) / max(1.0, min(w, h)) > 2.2:
        shape = "elongated"
    else:
        shape = "region"
    confidence = float(np.clip(0.25 + 0.45 * min(fill, 1.0) + 0.30 * min(circularity, 1.0), 0.0, 1.0))
    return {
        "center_px": center,
        "bbox": (int(x), int(y), int(w), int(h)),
        "area_px": int(np.count_nonzero(mask)),
        "geometry": f"{shape};fill={fill:.3f};circularity={circularity:.3f}",
        "confidence": confidence,
    }


def append_entity_states(
    path: Path | None,
    timestamp_ms: float,
    masks: dict[str, np.ndarray],
    depth,
    intrinsics,
    min_depth_m: float,
    max_depth_m: float,
    world_pose,
    base_from_camera,
    coordinate_names: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    rows = []
    states: dict[str, dict[str, object]] = {}
    now = time.time()
    names = ("lesion", "liver", "kidney", "bowel", "omentum", "gallbladder", "vessel")
    if coordinate_names is None:
        coordinate_names = set(names)
    for name in names:
        mask = masks.get(name)
        geom = mask_geometry(mask)
        if mask is None or not np.any(mask) or name not in coordinate_names:
            stats = None
            world_xyz = None
            base_xyz = None
            cam = [np.nan, np.nan, np.nan]
            valid = False
            valid_points = 0
            valid_ratio = 0.0
        else:
            stats = robust_mask_stats(mask, depth, intrinsics, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
            valid = bool(stats.valid)
            valid_points = int(stats.valid_points)
            valid_ratio = float(stats.valid_ratio)
            cam = stats.median_camera_m if stats.valid else [np.nan, np.nan, np.nan]
            world_xyz = None
            base_xyz = None
            if stats.valid and world_pose is not None:
                world_xyz = transform_points(world_pose, stats.median_camera_m.reshape(1, 3))[0]
            if stats.valid and base_from_camera is not None:
                base_xyz = camera_point_to_robot_base(stats.median_camera_m, base_from_camera)
        center = geom["center_px"]
        px, py = center if center is not None else ("", "")
        bbox = geom["bbox"]
        world = world_xyz if world_xyz is not None else [np.nan, np.nan, np.nan]
        base = base_xyz if base_xyz is not None else [np.nan, np.nan, np.nan]
        if name in coordinate_names:
            rows.append([
                f"{now:.3f}", f"{timestamp_ms:.3f}", name, int(valid), f"{geom['confidence']:.3f}",
                int(geom["area_px"]), valid_points, f"{valid_ratio:.4f}", px, py,
                bbox[0], bbox[1], bbox[2], bbox[3], int(geom["area_px"]), geom["geometry"],
                f"{cam[0]:.6f}", f"{cam[1]:.6f}", f"{cam[2]:.6f}",
                f"{world[0]:.6f}", f"{world[1]:.6f}", f"{world[2]:.6f}",
                f"{base[0]:.6f}", f"{base[1]:.6f}", f"{base[2]:.6f}",
            ])
        states[name] = {
            "valid": valid,
            "confidence": geom["confidence"],
            "center_px": center,
            "bbox": bbox,
            "area_px": geom["area_px"],
            "geometry": geom["geometry"],
            "valid_points": valid_points,
            "valid_depth_ratio": valid_ratio,
            "camera_m": np.array(cam, dtype=np.float32),
            "world_m": np.array(world, dtype=np.float32),
            "base_m": np.array(base, dtype=np.float32),
        }
    if path is not None:
        with path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
    return states


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    record_dir = base / args.record_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    pc_min_depth_mm = args.pc_min_depth_mm if args.pc_min_depth_mm is not None else args.min_depth_mm
    pc_max_depth_mm = args.pc_max_depth_mm if args.pc_max_depth_mm is not None else args.max_depth_mm
    material_names = ("liver", "kidney", "bowel", "omentum", "gallbladder", "vessel")
    entity_names = ("lesion", *material_names)
    coordinate_classes = set(entity_names) if args.coordinate_classes.strip().lower() == "all" else {
        item.strip() for item in args.coordinate_classes.split(",") if item.strip()
    }

    if args.make_marker:
        marker_path = base / "aruco_marker_id0.png"
        create_aruco_marker_png(str(marker_path))
        print(f"Wrote {marker_path}")
        return

    devices = RealSenseD405.list_devices()
    if not devices:
        raise SystemExit("No RealSense device found. Check USB cable, RealSense SDK and camera permissions.")
    print("RealSense devices:")
    for name, serial in devices:
        print(f"  {name} serial={serial}")

    if args.reset_camera:
        print("Resetting RealSense device. Wait a few seconds...")
        RealSenseD405.reset_devices()
        time.sleep(5.0)

    camera = RealSenseD405(args.width, args.height, args.fps, use_filters=args.rs_filters)
    state = DemoState()
    state.depth_history = deque(maxlen=max(1, int(args.depth_temporal_frames)))
    state.mask_mode = args.mask_mode
    generic_segmenter = HSVSegmenter()
    lesion_segmenter = PhantomLesionSegmenter(profile=args.lesion_profile)
    all_segmenter = PhantomMaterialSegmenter(profile=args.material_profile)
    material_segmenters = {
        name: PhantomMaterialMaskSegmenter(name, profile=args.material_profile)
        for name in material_names
    }
    csv_path = record_dir / "measurements.csv"
    entities_csv_path = record_dir / "entity_states.csv"
    vision_output_path = record_dir / args.vision_output_jsonl if args.vision_output_jsonl else None
    if vision_output_path is not None:
        vision_output_path.parent.mkdir(parents=True, exist_ok=True)
    vision_output_period = 1.0 / max(args.vision_output_hz, 0.1)
    last_vision_output_time = 0.0
    save_csv_header(csv_path)
    save_entities_csv_header(entities_csv_path)
    base_from_camera = load_transform_json(args.base_from_camera)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            state.mouse_xy = (x, y)
            if state.drag_start is not None:
                state.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            state.mouse_xy = (x, y)
            state.drag_start = (x, y)
            state.drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            state.mouse_xy = (x, y)
            state.drag_end = (x, y)
            state.drag_start = None

    o3d_viewer = None
    if args.open3d:
        try:
            o3d_viewer = Open3DLiveViewer()
        except Open3DUnavailable as exc:
            print(f"Open3D viewer disabled: {exc}")

    camera.start()
    win = "D405 RGB-D demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("Keys: q/ESC quit | s save | r log 3D | h HSV | l lesion | m ROI | n none | c clear ROI | a ArUco | w save checkerboard T_world_camera")
    print("Material modes from CLI: --mask-mode lesion|offline-lesions|liver|kidney|bowel|omentum|gallbladder|vessel|all")
    print("Current brown-red lesion profile: painted (override with --lesion-profile auto|official|pink)")
    try:
        while True:
            frame = camera.read()
            color = frame.color_bgr
            depth_raw = frame.depth_m
            state.depth_history.append(depth_raw)
            depth_stack = np.stack([d.astype(np.float32, copy=False) for d in state.depth_history], axis=0)
            fused_depth, valid_count, depth_mad = masked_median_depth(depth_stack)
            min_valid_count = max(1, int(np.ceil(args.depth_min_valid_ratio * len(state.depth_history))))
            stable = valid_count >= min_valid_count
            if args.depth_max_mad_mm > 0:
                stable &= depth_mad <= (args.depth_max_mad_mm / 1000.0)
            depth = np.where(stable, fused_depth, 0.0).astype(np.float32)

            # Keep a separate, more permissive map for visualization. The strict
            # map above remains the source for target coordinates and robot data.
            display_min_count = max(1, int(np.ceil(args.depth_display_min_valid_ratio * len(state.depth_history))))
            display_stable = valid_count >= display_min_count
            if args.depth_display_max_mad_mm > 0:
                display_stable &= depth_mad <= (args.depth_display_max_mad_mm / 1000.0)
            display_depth = np.where(display_stable, fused_depth, 0.0).astype(np.float32)
            if args.depth_smooth:
                depth = smooth_depth_bilateral(depth, args.depth_smooth_sigma_mm, 5.0)
                display_depth = smooth_depth_bilateral(display_depth, args.depth_smooth_sigma_mm, 5.0)
            # Small-hole completion improves real-time preview. Coordinates use
            # this map by default; switch to --coordinate-depth-source strict if
            # you only want high-confidence measured pixels.
            display_depth = fill_small_depth_holes(display_depth, max_hole_size=args.display_hole_size)
            coordinate_depth = display_depth if args.coordinate_depth_source == "display" else depth
            h, w = depth.shape
            if args.pc_color_mode == "rgb":
                point_colors = color
            elif args.pc_color_mode == "depth":
                point_colors = frame.depth_colormap
            else:
                point_colors = None
            pc_points, pc_colors = depth_to_points_sampled(
                display_depth,
                frame.intrinsics,
                color_bgr=point_colors,
                stride=args.pc_stride,
                min_depth_m=pc_min_depth_mm / 1000.0,
                max_depth_m=pc_max_depth_mm / 1000.0,
            )

            display_pc_points = pc_points
            display_pc_colors = pc_colors
            if args.open3d_clean and pc_points.size > 0:
                try:
                    display_pc_points, display_pc_colors = clean_point_cloud_open3d(
                        pc_points,
                        pc_colors,
                        voxel_size_m=args.voxel_mm / 1000.0,
                        remove_outliers=True,
                    )
                except Open3DUnavailable as exc:
                    print(f"Open3D cleanup disabled: {exc}")
                    args.open3d_clean = False

            if state.drag_start is not None and state.drag_end is not None:
                state.roi_mask = roi_mask((h, w), state.drag_start, state.drag_end)

            active_mask = None
            offline_lesion_masks = None
            offline_lesion_centers = None
            offline_lesion_bbox = None
            offline_lesion_stats = None
            all_masks = None
            mode = "none"
            if state.mask_mode == "offline-lesions":
                mode = "offline-lesions"
                try:
                    offline_lesion_masks, offline_lesion_centers, offline_lesion_bbox = make_lesion_masks(
                        color, args.offline_lesion_radius_scale
                    )
                except SystemExit:
                    # The detector may miss a frame while the board is occluded or out of view.
                    offline_lesion_masks = None
                    offline_lesion_centers = None
                    offline_lesion_bbox = None
                if offline_lesion_masks:
                    active_mask = np.logical_or.reduce(offline_lesion_masks)
            elif state.mask_mode == "all":
                if not state.freeze_mask or state.last_mask is None:
                    all_masks = all_segmenter.predict_all(color)
                    if args.use_depth_refine:
                        all_masks = refine_all_masks_by_depth(all_masks, depth)
                    state.last_mask = all_masks.get("lesion")
                else:
                    all_masks = all_segmenter.predict_all(color)
                    if args.use_depth_refine:
                        all_masks = refine_all_masks_by_depth(all_masks, depth)
                active_mask = all_masks.get("lesion") if all_masks is not None else None
                mode = "all"
            elif state.mask_mode in ("hsv", "lesion", *material_names):
                if not state.freeze_mask or state.last_mask is None:
                    if state.mask_mode == "lesion":
                        segmenter = lesion_segmenter
                    elif state.mask_mode in material_segmenters:
                        segmenter = material_segmenters[state.mask_mode]
                    else:
                        segmenter = generic_segmenter
                    masks = segmenter.predict(color)
                    state.last_mask = masks[0] if masks else None
                active_mask = state.last_mask
                if active_mask is not None and state.roi_mask is not None:
                    active_mask = active_mask & state.roi_mask
                mode = state.mask_mode
            elif state.mask_mode == "roi" and state.roi_mask is not None:
                active_mask = state.roi_mask
                mode = "roi"

            pose = None
            checkerboard_pose = None
            shown_color = color.copy()
            if args.chessboard_world:
                checkerboard_pose = estimate_checkerboard_world_pose(color, frame.intrinsics)
                if checkerboard_pose.valid:
                    state.last_world_pose = checkerboard_pose.t_world_camera
                shown_color = draw_checkerboard_pose(shown_color, frame.intrinsics, checkerboard_pose)
            elif args.aruco:
                pose = estimate_aruco_world_pose(color, frame.intrinsics, args.marker_size_mm / 1000.0)
                if pose.valid:
                    state.last_world_pose = pose.t_world_camera
                shown_color = draw_aruco_axis(shown_color, frame.intrinsics, pose)

            stats = None
            world_xyz = None
            active_points = None
            lesion_center_camera_m = None
            all_entity_states = None
            if active_mask is not None:
                stats = robust_mask_stats(
                    active_mask,
                    coordinate_depth,
                    frame.intrinsics,
                    min_depth_m=args.min_depth_mm / 1000.0,
                    max_depth_m=args.max_depth_mm / 1000.0,
                )
                active_points = mask_to_points(
                    active_mask,
                    coordinate_depth,
                    frame.intrinsics,
                    min_depth_m=args.min_depth_mm / 1000.0,
                    max_depth_m=args.max_depth_mm / 1000.0,
                )
                if mode == "offline-lesions" and offline_lesion_masks is not None:
                    shown_color = draw_offline_lesion_overlay(
                        shown_color,
                        offline_lesion_masks,
                        offline_lesion_centers or [],
                        offline_lesion_bbox,
                    )
                elif mode == "all" and all_masks is not None:
                    draw_centers = {name: mask_geometry(mask)["center_px"] for name, mask in all_masks.items()}
                    shown_color = draw_named_masks(shown_color, all_masks, centers=draw_centers, label_names=coordinate_classes)
                elif mode == "lesion":
                    shown_color = draw_labeled_mask(shown_color, active_mask, label="lesion")
                else:
                    shown_color = overlay_mask(shown_color, active_mask)
                if stats.valid:
                    lesion_center_camera_m = stats.median_camera_m
                    pt = stats.median_camera_m.reshape(1, 3)
                    u = int(round((stats.median_camera_m[0] * frame.intrinsics.fx / stats.median_camera_m[2]) + frame.intrinsics.cx))
                    v = int(round((stats.median_camera_m[1] * frame.intrinsics.fy / stats.median_camera_m[2]) + frame.intrinsics.cy))
                    cv2.circle(shown_color, (u, v), 6, (0, 0, 255), -1)
                    if state.last_world_pose is not None:
                        world_xyz = transform_points(state.last_world_pose, pt)[0]

            if mode == "offline-lesions" and offline_lesion_masks is not None:
                offline_lesion_stats = []
                for lesion_id, lesion_mask in enumerate(offline_lesion_masks, start=1):
                    lesion_stats = robust_mask_stats(
                        lesion_mask,
                        coordinate_depth,
                        frame.intrinsics,
                        min_depth_m=args.min_depth_mm / 1000.0,
                        max_depth_m=args.max_depth_mm / 1000.0,
                    )
                    offline_lesion_stats.append((lesion_id, lesion_stats))

            if mode == "all" and all_masks is not None:
                all_entity_states = append_entity_states(
                    None,
                    frame.timestamp_ms,
                    all_masks,
                    coordinate_depth,
                    frame.intrinsics,
                    args.min_depth_mm / 1000.0,
                    args.max_depth_mm / 1000.0,
                    state.last_world_pose,
                    base_from_camera,
                    coordinate_classes,
                )
                lesion_state = all_entity_states.get("lesion")
                if lesion_state and lesion_state["valid"]:
                    lesion_center_camera_m = lesion_state["camera_m"]
                now = time.monotonic()
                if vision_output_path is not None and now - last_vision_output_time >= vision_output_period:
                    vision_result = build_vision_result(
                        frame.timestamp_ms, all_masks, depth, frame.intrinsics,
                        args.min_depth_mm / 1000.0, args.max_depth_mm / 1000.0,
                        base_from_camera, args.obstacle_margin_mm / 1000.0,
                    )
                    with vision_output_path.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(vision_result, separators=(",", ":"), allow_nan=False) + "\n")
                    last_vision_output_time = now

            lines = [
                f"intr fx={frame.intrinsics.fx:.1f} fy={frame.intrinsics.fy:.1f} cx={frame.intrinsics.cx:.1f} cy={frame.intrinsics.cy:.1f}",
                f"depth scale={camera.depth_scale:.6f} m/unit | target depth={args.min_depth_mm:.0f}-{args.max_depth_mm:.0f} mm | pointcloud={pc_min_depth_mm:.0f}-{pc_max_depth_mm:.0f} mm | mode={mode} | pc_color={args.pc_color_mode} | aruco={'on' if args.aruco else 'off'}",
            ]
            if state.mouse_xy is not None:
                x, y = state.mouse_xy
                if 0 <= x < w and 0 <= y < h:
                    xyz = pixel_to_camera_xyz(x, y, float(coordinate_depth[y, x]), frame.intrinsics)
                    if xyz is None:
                        lines.append(f"cursor ({x},{y}) depth invalid")
                    else:
                        lines.append(f"cursor ({x},{y}) depth={coordinate_depth[y, x]*1000:.1f} mm camera=({xyz[0]*1000:.1f},{xyz[1]*1000:.1f},{xyz[2]*1000:.1f}) mm")
                        cv2.drawMarker(shown_color, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
            if stats is not None:
                if stats.valid:
                    cam = stats.median_camera_m * 1000.0
                    lines.append(f"{mode} valid pts={stats.valid_points}/{stats.total_pixels} median camera=({cam[0]:.1f},{cam[1]:.1f},{cam[2]:.1f}) mm")
                    if world_xyz is not None:
                        ww = world_xyz * 1000.0
                        lines.append(f"world from ArUco=({ww[0]:.1f},{ww[1]:.1f},{ww[2]:.1f}) mm")
                else:
                    lines.append(f"{mode} invalid: pts={stats.valid_points}/{stats.total_pixels}, no reliable depth")
            if pose is not None:
                lines.append("ArUco pose valid" if pose.valid else "ArUco pose not found")
            if args.chessboard_world:
                if checkerboard_pose is not None and checkerboard_pose.valid:
                    lines.append(f"checkerboard World reprojection RMS={checkerboard_pose.reprojection_error_px:.2f}px")
                else:
                    lines.append("checkerboard World not found")
            if mode == "offline-lesions" and offline_lesion_stats is not None:
                valid_ids = [str(lesion_id) for lesion_id, lesion_stats in offline_lesion_stats if lesion_stats.valid]
                lines.append(f"offline lesions 3D valid: {len(valid_ids)}/9" + (" [" + ",".join(valid_ids) + "]" if valid_ids else ""))
            elif state.mask_mode == "offline-lesions":
                lines.append("red 3x3 lesion board not detected")
            if mode == "all" and all_masks is not None:
                centers = []
                for name in coordinate_classes:
                    center = mask_centroid(all_masks.get(name))
                    if center is not None:
                        centers.append(f"{name}=({center[0]},{center[1]})")
                if centers:
                    lines.append("centers px: " + " | ".join(centers[:3]))
                    if len(centers) > 3:
                        lines.append("centers px: " + " | ".join(centers[3:]))
                if all_entity_states is not None:
                    valid3d = [name for name, state_info in all_entity_states.items() if state_info["valid"]]
                    lines.append("3D valid: " + (", ".join(valid3d) if valid3d else "none"))

            shown_color = draw_text_lines(shown_color, lines)
            top = hstack_resize(shown_color, frame.depth_colormap)
            reconstruction_view = draw_3d_reconstruction(
                display_pc_points,
                display_pc_colors,
                size=top.shape[:2],
                highlight_points_m=active_points,
                highlight_center_m=lesion_center_camera_m,
                point_radius=args.recon_point_radius,
                fixed_span_m=args.recon_fixed_span_mm / 1000.0,
                background="white",
            )
            if o3d_viewer is not None:
                o3d_viewer.update(display_pc_points, display_pc_colors, highlight_points_m=active_points)
            canvas = np.vstack([top, reconstruction_view])
            cv2.imshow(win, canvas)

            key_code = cv2.waitKeyEx(1)
            key = (key_code & 0xFF) if key_code >= 0 else -1
            if os.name == "nt":
                user32 = ctypes.windll.user32
                for virtual_key in (0x1B, ord("q"), ord("w"), ord("s"), ord("r"), ord("h"), ord("l"), ord("m"), ord("n"), ord("c"), ord("a")):
                    if user32.GetAsyncKeyState(virtual_key) & 1:
                        key = virtual_key
                        break
            if key in (27, ord("q")):
                break
            if key == ord("h"):
                state.mask_mode = "none" if state.mask_mode == "hsv" else "hsv"
                state.last_mask = None
            if key == ord("l"):
                state.mask_mode = "none" if state.mask_mode == "lesion" else "lesion"
                state.last_mask = None
            if key == ord("m"):
                state.mask_mode = "roi"
                state.last_mask = None
            if key == ord("n"):
                state.mask_mode = "none"
                state.last_mask = None
            if key == ord("c"):
                state.roi_mask = None
                state.last_mask = None
            if key == ord("a"):
                args.aruco = not args.aruco
            if key == ord("w"):
                if args.chessboard_world and checkerboard_pose is not None and checkerboard_pose.valid:
                    transform_path = record_dir / "T_world_camera_checkerboard.json"
                    save_world_camera_transform(transform_path, checkerboard_pose, frame.intrinsics)
                    print(f"Saved checkerboard T_world_camera to {transform_path}")
                else:
                    print("Cannot save checkerboard transform: enable --chessboard-world and keep the full board visible.")
            if key == ord("s"):
                stem = time.strftime("frame_%Y%m%d_%H%M%S")
                imwrite_unicode(str(record_dir / f"{stem}_color.png"), color)
                imwrite_unicode(str(record_dir / f"{stem}_depth_vis.png"), frame.depth_colormap)
                np.savez_compressed(record_dir / f"{stem}.npz", color_bgr=color, depth_m=depth_raw, depth_fused_m=depth)
                write_ply_ascii(str(record_dir / f"{stem}_cloud.ply"), display_pc_points, display_pc_colors)
                if active_points is not None and active_points.size > 0:
                    write_ply_ascii(str(record_dir / f"{stem}_{mode}_target_cloud.ply"), active_points)
                print(f"Saved {stem} to {record_dir}")
            if key == ord("r"):
                if stats is not None and mode != "offline-lesions":
                    append_stats(csv_path, frame.timestamp_ms, mode, stats, world_xyz)
                    print(f"Logged {mode} measurement to {csv_path}")
                if mode == "offline-lesions" and offline_lesion_stats is not None:
                    lesion_csv_path = record_dir / "offline_lesions_3d.csv"
                    append_offline_lesion_stats(
                        lesion_csv_path,
                        frame.timestamp_ms,
                        offline_lesion_centers,
                        offline_lesion_stats,
                        state.last_world_pose,
                        base_from_camera,
                    )
                    valid_count = sum(int(lesion_stats.valid) for _, lesion_stats in offline_lesion_stats)
                    print(f"Logged {valid_count}/9 lesion 3D results to {lesion_csv_path}")
                if mode == "all" and all_masks is not None:
                    append_entity_states(
                        entities_csv_path,
                        frame.timestamp_ms,
                        all_masks,
                        coordinate_depth,
                        frame.intrinsics,
                        args.min_depth_mm / 1000.0,
                        args.max_depth_mm / 1000.0,
                        state.last_world_pose,
                        base_from_camera,
                        coordinate_classes,
                    )
                    print(f"Logged coordinate targets {sorted(coordinate_classes)} to {entities_csv_path}")
    finally:
        if o3d_viewer is not None:
            o3d_viewer.close()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
