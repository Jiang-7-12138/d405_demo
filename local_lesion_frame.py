from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_d405 import CameraIntrinsics
from detect_offline_lesions import latest_npz, load_intrinsics, make_lesion_masks
from io_utils import imwrite_unicode
from pointcloud import mask_to_points, robust_mask_stats
from pointcloud_viewer import write_ply_ascii


@dataclass
class LocalFrameResult:
    lesion_id: int
    center_px: tuple[int, int]
    origin_camera_m: np.ndarray
    x_axis_camera: np.ndarray
    y_axis_camera: np.ndarray
    z_axis_camera: np.ndarray
    T_local_camera: np.ndarray
    T_camera_local: np.ndarray
    lesion_points_camera: np.ndarray
    lesion_points_local: np.ndarray
    context_points_camera: np.ndarray
    plane_points_camera: np.ndarray
    height_min_m: float
    height_max_m: float
    height_range_m: float
    size_xyz_m: np.ndarray
    voxel_grid: np.ndarray | None
    voxel_origin_local_m: np.ndarray | None
    voxel_size_m: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local 3D coordinate frame for the best offline lesion target.")
    parser.add_argument("--npz", default=None, help="Offline reconstruction .npz. Defaults to latest captures/offline_recon/*.npz.")
    parser.add_argument("--intrinsics", default=None, help="Matching *_intrinsics.json. Defaults from --npz stem.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to the npz directory.")
    parser.add_argument("--stem", default=None, help="Output prefix. Defaults to npz stem + _local_lesion.")
    parser.add_argument("--lesion-id", type=int, default=0, help="1-9 target id. 0 chooses the best-scored target automatically.")
    parser.add_argument("--min-depth-mm", type=float, default=120.0)
    parser.add_argument("--max-depth-mm", type=float, default=700.0)
    parser.add_argument("--radius-scale", type=float, default=0.30)
    parser.add_argument("--context-scale", type=float, default=2.2, help="Radius around target used for local output/context point cloud.")
    parser.add_argument("--plane-inner-scale", type=float, default=1.25, help="Exclude lesion core inside radius*scale when fitting plane.")
    parser.add_argument("--plane-outer-scale", type=float, default=2.5, help="Outer radius for background plane fitting.")
    parser.add_argument("--ransac-threshold-mm", type=float, default=2.5)
    parser.add_argument("--ransac-iterations", type=int, default=250)
    parser.add_argument("--voxel-mm", type=float, default=1.5, help="Voxel size for local voxel model. <=0 disables voxel output.")
    parser.add_argument("--approach-height-mm", type=float, default=15.0, help="Pre-cut approach distance above lesion apex along local +Z.")
    parser.add_argument("--cut-depth-mm", type=float, default=1.0, help="Cut target offset below the robust apex along local Z.")
    parser.add_argument("--apex-percentile", type=float, default=97.0, help="Robust top percentile used to estimate lesion apex instead of a single noisy max point.")
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def points_from_mask(depth_m: np.ndarray, intr: CameraIntrinsics, mask: np.ndarray, min_depth_m: float, max_depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)
    z = depth_m[ys, xs].astype(np.float32)
    x = (xs.astype(np.float32) - intr.cx) * z / intr.fx
    y = (ys.astype(np.float32) - intr.cy) * z / intr.fy
    return np.column_stack((x, y, z)).astype(np.float32), np.column_stack((xs, ys)).astype(np.int32)


def disk_mask(shape: tuple[int, int], center: tuple[int, int], radius_px: float) -> np.ndarray:
    yy, xx = np.indices(shape)
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2) <= radius_px ** 2


def ring_mask(shape: tuple[int, int], center: tuple[int, int], inner_px: float, outer_px: float) -> np.ndarray:
    yy, xx = np.indices(shape)
    rr = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return (rr >= inner_px ** 2) & (rr <= outer_px ** 2)


def robust_plane_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.median(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    return centroid.astype(np.float32), normal.astype(np.float32)


def fit_plane_ransac(points: np.ndarray, threshold_m: float, iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] < 30:
        raise RuntimeError(f"Not enough plane points: {points.shape[0]}")
    rng = np.random.default_rng(7)
    best_inliers = None
    best_count = -1
    n = points.shape[0]
    for _ in range(max(1, iterations)):
        ids = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        dist = np.abs((points - p0) @ normal)
        inliers = dist <= threshold_m
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 20:
        centroid, normal = robust_plane_pca(points)
        dist = np.abs((points - centroid) @ normal)
        best_inliers = dist <= max(threshold_m * 2.0, 0.005)
    inlier_points = points[best_inliers]
    if inlier_points.shape[0] < 20:
        inlier_points = points
    centroid, normal = robust_plane_pca(inlier_points)
    return centroid, normal, inlier_points


def build_axes(normal: np.ndarray, center_xyz: np.ndarray, center_px: tuple[int, int], intr: CameraIntrinsics, lesion_points: np.ndarray, plane_origin: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_axis = normal / max(np.linalg.norm(normal), 1e-12)
    signed = (lesion_points - plane_origin) @ z_axis
    if np.percentile(signed, 90) < abs(np.percentile(signed, 10)):
        z_axis = -z_axis
        signed = -signed

    origin = center_xyz - ((center_xyz - plane_origin) @ z_axis) * z_axis
    # Camera pixel +x direction projected onto the local plane.
    z_ref = max(float(center_xyz[2]), 1e-6)
    p_right = np.array([
        (center_px[0] + 1.0 - intr.cx) * z_ref / intr.fx,
        (center_px[1] - intr.cy) * z_ref / intr.fy,
        z_ref,
    ], dtype=np.float32)
    p_center = np.array([
        (center_px[0] - intr.cx) * z_ref / intr.fx,
        (center_px[1] - intr.cy) * z_ref / intr.fy,
        z_ref,
    ], dtype=np.float32)
    x_axis = p_right - p_center
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-12)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-12)
    return origin.astype(np.float32), x_axis.astype(np.float32), y_axis.astype(np.float32), z_axis.astype(np.float32)


def make_transforms(origin: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_camera_local = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32)
    t_camera_local = origin.astype(np.float32)
    T_camera_local = np.eye(4, dtype=np.float32)
    T_camera_local[:3, :3] = r_camera_local
    T_camera_local[:3, 3] = t_camera_local
    T_local_camera = np.eye(4, dtype=np.float32)
    T_local_camera[:3, :3] = r_camera_local.T
    T_local_camera[:3, 3] = -r_camera_local.T @ t_camera_local
    return T_local_camera, T_camera_local


def transform_points(points: np.ndarray, T_local_camera: np.ndarray) -> np.ndarray:
    return (points @ T_local_camera[:3, :3].T) + T_local_camera[:3, 3]


def choose_best_lesion(masks: list[np.ndarray], centers: list[tuple[int, int]], depth: np.ndarray, intr: CameraIntrinsics, min_depth_m: float, max_depth_m: float) -> tuple[int, list[dict[str, float]]]:
    scores = []
    for i, mask in enumerate(masks):
        stats = robust_mask_stats(mask, depth, intr, min_depth_m, max_depth_m)
        valid = mask & np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
        z = depth[valid]
        mad = float(np.median(np.abs(z - np.median(z)))) if z.size else 999.0
        score = float(stats.valid_points) / (1.0 + 300.0 * mad)
        scores.append({"lesion_id": i + 1, "score": score, "valid_points": float(stats.valid_points), "depth_mad_m": mad})
    best = max(range(len(scores)), key=lambda idx: scores[idx]["score"])
    return best, scores


def voxelize(points_local: np.ndarray, voxel_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.min(points_local, axis=0)
    hi = np.max(points_local, axis=0)
    dims = np.maximum(1, np.ceil((hi - lo) / voxel_size_m).astype(int) + 1)
    if np.any(dims > 400):
        raise RuntimeError(f"Voxel grid too large: {dims.tolist()}, increase --voxel-mm")
    ids = np.floor((points_local - lo) / voxel_size_m).astype(int)
    ids = np.clip(ids, 0, dims - 1)
    grid = np.zeros(tuple(int(x) for x in dims), dtype=bool)
    grid[ids[:, 0], ids[:, 1], ids[:, 2]] = True
    return grid, lo.astype(np.float32)


def make_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation.astype(np.float32)
    pose[:3, 3] = translation.astype(np.float32)
    return pose


def local_to_camera_point(point_local: np.ndarray, T_camera_local: np.ndarray) -> np.ndarray:
    return T_camera_local[:3, :3] @ point_local + T_camera_local[:3, 3]


def local_rotation_to_camera(rotation_local: np.ndarray, T_camera_local: np.ndarray) -> np.ndarray:
    return T_camera_local[:3, :3] @ rotation_local


def estimate_cut_plan(
    lesion_local: np.ndarray,
    T_camera_local: np.ndarray,
    apex_percentile: float,
    approach_height_m: float,
    cut_depth_m: float,
) -> dict[str, object]:
    if lesion_local.shape[0] < 30:
        raise RuntimeError("Not enough local lesion points for cut planning")

    apex_threshold = float(np.percentile(lesion_local[:, 2], np.clip(apex_percentile, 50.0, 100.0)))
    top_points = lesion_local[lesion_local[:, 2] >= apex_threshold]
    if top_points.shape[0] < 5:
        top_points = lesion_local[np.argsort(lesion_local[:, 2])[-5:]]
    apex_local = np.median(top_points, axis=0).astype(np.float32)

    footprint = lesion_local[:, :2]
    footprint_center = np.median(footprint, axis=0)
    centered = footprint - footprint_center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    cut_x = vh[0].astype(np.float32)
    if np.linalg.norm(cut_x) < 1e-8:
        cut_x = np.array([1.0, 0.0], dtype=np.float32)
    cut_x = cut_x / max(np.linalg.norm(cut_x), 1e-12)
    cut_y = np.array([-cut_x[1], cut_x[0]], dtype=np.float32)

    # Tool frame convention for the recommendation:
    # local tool X = suggested cut direction in the fitted XY plane.
    # local tool Y = lateral in-plane direction.
    # local tool Z = approach direction pointing toward the tissue, i.e. -local Z.
    tool_x_local = np.array([cut_x[0], cut_x[1], 0.0], dtype=np.float32)
    tool_y_local = np.array([cut_y[0], cut_y[1], 0.0], dtype=np.float32)
    tool_z_local = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    R_tool_local = np.column_stack([tool_x_local, tool_y_local, tool_z_local]).astype(np.float32)

    cut_target_local = apex_local.copy()
    cut_target_local[2] = max(0.0, float(apex_local[2] - cut_depth_m))
    pre_cut_local = cut_target_local + np.array([0.0, 0.0, approach_height_m], dtype=np.float32)
    base_center_local = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    R_tool_camera = local_rotation_to_camera(R_tool_local, T_camera_local)
    cut_target_camera = local_to_camera_point(cut_target_local, T_camera_local)
    pre_cut_camera = local_to_camera_point(pre_cut_local, T_camera_local)
    apex_camera = local_to_camera_point(apex_local, T_camera_local)
    base_center_camera = local_to_camera_point(base_center_local, T_camera_local)

    cut_pose_camera = make_pose(R_tool_camera, cut_target_camera)
    pre_cut_pose_camera = make_pose(R_tool_camera, pre_cut_camera)

    footprint_extent = np.percentile(footprint @ np.column_stack([cut_x, cut_y]), 99, axis=0) - np.percentile(footprint @ np.column_stack([cut_x, cut_y]), 1, axis=0)
    yaw_in_local_xy_deg = float(np.degrees(np.arctan2(tool_x_local[1], tool_x_local[0])))
    approach_angle_to_plane_deg = 90.0
    return {
        "apex_local_m": apex_local.tolist(),
        "apex_camera_m": apex_camera.tolist(),
        "base_center_local_m": base_center_local.tolist(),
        "base_center_camera_m": base_center_camera.tolist(),
        "cut_target_local_m": cut_target_local.tolist(),
        "cut_target_camera_m": cut_target_camera.tolist(),
        "pre_cut_local_m": pre_cut_local.tolist(),
        "pre_cut_camera_m": pre_cut_camera.tolist(),
        "R_tool_local": R_tool_local.tolist(),
        "R_tool_camera": R_tool_camera.tolist(),
        "cut_pose_camera": cut_pose_camera.tolist(),
        "pre_cut_pose_camera": pre_cut_pose_camera.tolist(),
        "cut_direction_local_xy": tool_x_local.tolist(),
        "cut_lateral_local_xy": tool_y_local.tolist(),
        "tool_approach_direction_local": tool_z_local.tolist(),
        "tool_approach_direction_camera": R_tool_camera[:, 2].tolist(),
        "yaw_in_local_xy_deg": yaw_in_local_xy_deg,
        "approach_angle_to_plane_deg": approach_angle_to_plane_deg,
        "footprint_extent_major_minor_mm": (footprint_extent * 1000.0).tolist(),
        "apex_height_mm": float(apex_local[2] * 1000.0),
        "cut_depth_mm": float(cut_depth_m * 1000.0),
        "approach_height_mm": float(approach_height_m * 1000.0),
        "convention": "Tool +Z points along the approach direction toward tissue (-local Z). Tool +X is the suggested in-plane cut direction.",
    }


def write_local_csv(path: Path, points_local: np.ndarray) -> None:
    header = "local_x_m,local_y_m,local_z_m\n"
    np.savetxt(path, points_local, delimiter=",", header=header.strip(), comments="", fmt="%.7f")


def write_cut_plan_csv(path: Path, cut_plan: dict[str, object]) -> None:
    rows = []
    for key in ("apex_camera_m", "base_center_camera_m", "cut_target_camera_m", "pre_cut_camera_m", "tool_approach_direction_camera"):
        value = np.array(cut_plan[key], dtype=np.float32)
        rows.append([key, f"{value[0]:.7f}", f"{value[1]:.7f}", f"{value[2]:.7f}", f"{value[0]*1000.0:.3f}", f"{value[1]*1000.0:.3f}", f"{value[2]*1000.0:.3f}"])
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("name,x_m,y_m,z_m,x_mm,y_mm,z_mm\n")
        for row in rows:
            f.write(",".join(row) + "\n")
        f.write(f"yaw_in_local_xy_deg,{cut_plan['yaw_in_local_xy_deg']:.6f},,,,,\n")
        f.write(f"approach_angle_to_plane_deg,{cut_plan['approach_angle_to_plane_deg']:.6f},,,,,\n")
        f.write(f"apex_height_mm,{cut_plan['apex_height_mm']:.6f},,,,,\n")
        f.write(f"cut_depth_mm,{cut_plan['cut_depth_mm']:.6f},,,,,\n")
        f.write(f"approach_height_mm,{cut_plan['approach_height_mm']:.6f},,,,,\n")


def write_height_map(path: Path, points_local: np.ndarray, pixel_mm: float = 0.5) -> None:
    xy = points_local[:, :2]
    z = points_local[:, 2]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    step = pixel_mm / 1000.0
    w = int(max(1, np.ceil((hi[0] - lo[0]) / step) + 1))
    h = int(max(1, np.ceil((hi[1] - lo[1]) / step) + 1))
    if w > 2000 or h > 2000:
        step = max((hi[0] - lo[0]) / 1200.0, (hi[1] - lo[1]) / 1200.0, step)
        w = int(max(1, np.ceil((hi[0] - lo[0]) / step) + 1))
        h = int(max(1, np.ceil((hi[1] - lo[1]) / step) + 1))
    img = np.full((h, w), np.nan, dtype=np.float32)
    ix = np.clip(((xy[:, 0] - lo[0]) / step).astype(int), 0, w - 1)
    iy = np.clip(((xy[:, 1] - lo[1]) / step).astype(int), 0, h - 1)
    for x, y, zz in zip(ix, iy, z):
        if not np.isfinite(img[y, x]) or zz > img[y, x]:
            img[y, x] = zz
    valid = np.isfinite(img)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if np.any(valid):
        zmin = float(np.nanpercentile(img, 2))
        zmax = float(np.nanpercentile(img, 98))
        norm = np.clip((img - zmin) / max(zmax - zmin, 1e-6), 0, 1)
        out = cv2.applyColorMap((np.nan_to_num(norm) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        out[~valid] = (0, 0, 0)
    imwrite_unicode(str(path), cv2.flip(out, 0))


def write_overlay(path: Path, color: np.ndarray, selected_mask: np.ndarray, context_mask: np.ndarray, center: tuple[int, int], lesion_id: int) -> None:
    out = color.copy()
    layer = np.zeros_like(out)
    layer[context_mask] = (0, 180, 255)
    layer[selected_mask] = (255, 255, 0)
    out = cv2.addWeighted(out, 1.0, layer, 0.28, 0.0)
    contours, _ = cv2.findContours(selected_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.drawContours(out, contours, -1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(out, center, (255, 255, 0), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
    cv2.putText(out, f"selected lession {lesion_id}", (max(5, center[0] - 70), max(24, center[1] - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, f"selected lession {lesion_id}", (max(5, center[0] - 70), max(24, center[1] - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    imwrite_unicode(str(path), out)


def save_open3d_visuals(out_dir: Path, stem: str, result: LocalFrameResult, color_bgr: np.ndarray, context_colors_bgr: np.ndarray, voxel_size_m: float) -> None:
    local_colors = np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (result.lesion_points_local.shape[0], 1))
    plane_local = transform_points(result.plane_points_camera, result.T_local_camera)
    plane_colors = np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (plane_local.shape[0], 1))
    context_local = transform_points(result.context_points_camera, result.T_local_camera)
    write_ply_ascii(str(out_dir / f"{stem}_local_lesion_points.ply"), result.lesion_points_local, local_colors)
    write_ply_ascii(str(out_dir / f"{stem}_local_context_plane.ply"), np.vstack([context_local, result.lesion_points_local]), np.vstack([context_colors_bgr, local_colors]))
    write_ply_ascii(str(out_dir / f"{stem}_local_plane_inliers.ply"), plane_local, plane_colors)

    if result.voxel_grid is not None and result.voxel_origin_local_m is not None and voxel_size_m > 0:
        ids = np.argwhere(result.voxel_grid)
        centers = result.voxel_origin_local_m + (ids.astype(np.float32) + 0.5) * voxel_size_m
        colors = np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (centers.shape[0], 1))
        write_ply_ascii(str(out_dir / f"{stem}_local_voxels_centers.ply"), centers.astype(np.float32), colors)


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
    stem = args.stem or f"{npz_path.stem}_local_lesion"

    data = np.load(npz_path)
    color = data["color_bgr"]
    depth = data["export_depth_m"] if "export_depth_m" in data.files else data["depth_m"]
    intr = load_intrinsics(intr_path)
    masks, centers, _ = make_lesion_masks(color, args.radius_scale)
    min_depth_m = args.min_depth_mm / 1000.0
    max_depth_m = args.max_depth_mm / 1000.0
    selected_idx, scores = choose_best_lesion(masks, centers, depth, intr, min_depth_m, max_depth_m)
    if args.lesion_id:
        if not 1 <= args.lesion_id <= len(masks):
            raise SystemExit("--lesion-id must be 1..9 or 0 for auto")
        selected_idx = args.lesion_id - 1

    selected_mask = masks[selected_idx]
    center = centers[selected_idx]
    radius_px = np.sqrt(np.count_nonzero(selected_mask) / np.pi)
    context_mask = disk_mask(depth.shape, center, radius_px * args.context_scale)
    plane_mask = ring_mask(depth.shape, center, radius_px * args.plane_inner_scale, radius_px * args.plane_outer_scale)
    context_points, context_pixels = points_from_mask(depth, intr, context_mask, min_depth_m, max_depth_m)
    lesion_points = mask_to_points(selected_mask, depth, intr, min_depth_m, max_depth_m, max_points=200000)
    plane_points, _ = points_from_mask(depth, intr, plane_mask, min_depth_m, max_depth_m)
    if lesion_points.shape[0] < 30:
        raise SystemExit(f"Selected lesion has too few 3D points: {lesion_points.shape[0]}")
    if plane_points.shape[0] < 30:
        raise SystemExit(f"Plane-fitting ring has too few 3D points: {plane_points.shape[0]}")

    plane_origin, plane_normal, plane_inliers = fit_plane_ransac(plane_points, args.ransac_threshold_mm / 1000.0, args.ransac_iterations)
    center_stats = robust_mask_stats(selected_mask, depth, intr, min_depth_m, max_depth_m)
    if not center_stats.valid:
        raise SystemExit("Selected lesion center depth is invalid")
    origin, x_axis, y_axis, z_axis = build_axes(plane_normal, center_stats.median_camera_m, center, intr, lesion_points, plane_origin)
    T_local_camera, T_camera_local = make_transforms(origin, x_axis, y_axis, z_axis)
    lesion_local = transform_points(lesion_points, T_local_camera)
    context_local = transform_points(context_points, T_local_camera)
    heights = lesion_local[:, 2]
    size_xyz = np.percentile(lesion_local, 99, axis=0) - np.percentile(lesion_local, 1, axis=0)

    voxel_grid = None
    voxel_origin = None
    voxel_size_m = args.voxel_mm / 1000.0
    if args.voxel_mm > 0:
        voxel_grid, voxel_origin = voxelize(lesion_local, voxel_size_m)
    cut_plan = estimate_cut_plan(
        lesion_local,
        T_camera_local,
        apex_percentile=args.apex_percentile,
        approach_height_m=args.approach_height_mm / 1000.0,
        cut_depth_m=args.cut_depth_mm / 1000.0,
    )

    result = LocalFrameResult(
        lesion_id=selected_idx + 1,
        center_px=center,
        origin_camera_m=origin,
        x_axis_camera=x_axis,
        y_axis_camera=y_axis,
        z_axis_camera=z_axis,
        T_local_camera=T_local_camera,
        T_camera_local=T_camera_local,
        lesion_points_camera=lesion_points,
        lesion_points_local=lesion_local,
        context_points_camera=context_points,
        plane_points_camera=plane_inliers,
        height_min_m=float(np.min(heights)),
        height_max_m=float(np.max(heights)),
        height_range_m=float(np.max(heights) - np.min(heights)),
        size_xyz_m=size_xyz.astype(np.float32),
        voxel_grid=voxel_grid,
        voxel_origin_local_m=voxel_origin,
        voxel_size_m=voxel_size_m if args.voxel_mm > 0 else None,
    )

    overlay_path = out_dir / f"{stem}_selected_overlay.png"
    write_overlay(overlay_path, color, selected_mask, context_mask, center, result.lesion_id)
    write_height_map(out_dir / f"{stem}_height_map.png", lesion_local)
    write_local_csv(out_dir / f"{stem}_local_points.csv", lesion_local)
    context_colors = color[context_pixels[:, 1], context_pixels[:, 0]].astype(np.uint8) if context_pixels.size else np.empty((0, 3), dtype=np.uint8)
    save_open3d_visuals(out_dir, stem, result, color, context_colors, voxel_size_m)

    metadata = {
        "input_npz": str(npz_path),
        "selected_lesion_id": result.lesion_id,
        "selection_scores": scores,
        "center_pixel_xy": list(center),
        "origin_camera_m": result.origin_camera_m.tolist(),
        "x_axis_camera": result.x_axis_camera.tolist(),
        "y_axis_camera": result.y_axis_camera.tolist(),
        "z_axis_camera": result.z_axis_camera.tolist(),
        "T_local_camera": result.T_local_camera.tolist(),
        "T_camera_local": result.T_camera_local.tolist(),
        "height_min_mm": result.height_min_m * 1000.0,
        "height_max_mm": result.height_max_m * 1000.0,
        "height_range_mm": result.height_range_m * 1000.0,
        "size_xyz_mm_p1_p99": (result.size_xyz_m * 1000.0).tolist(),
        "lesion_points": int(result.lesion_points_local.shape[0]),
        "plane_inliers": int(result.plane_points_camera.shape[0]),
        "voxel_size_mm": args.voxel_mm if args.voxel_mm > 0 else None,
        "voxel_occupied": int(np.count_nonzero(voxel_grid)) if voxel_grid is not None else None,
        "cut_plan": cut_plan,
        "note": "Camera frame uses RealSense X right, Y down, Z forward. Local frame origin is lesion center projected to fitted surrounding plane; local +Z points toward lesion protrusion.",
    }
    (out_dir / f"{stem}_local_frame.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_cut_plan_csv(out_dir / f"{stem}_cut_plan.csv", cut_plan)
    if voxel_grid is not None:
        np.savez_compressed(out_dir / f"{stem}_voxels.npz", voxel_grid=voxel_grid, voxel_origin_local_m=voxel_origin, voxel_size_m=voxel_size_m)

    print(f"input_npz: {npz_path}")
    print(f"selected lession {result.lesion_id} center_px={center}")
    print(f"height mm: min={result.height_min_m*1000:.2f}, max={result.height_max_m*1000:.2f}, range={result.height_range_m*1000:.2f}")
    print(f"size xyz p1-p99 mm: {(result.size_xyz_m * 1000.0).round(2).tolist()}")
    print(f"T_local_camera saved: {out_dir / f'{stem}_local_frame.json'}")
    print(
        "cut target camera mm: "
        f"{(np.array(cut_plan['cut_target_camera_m']) * 1000.0).round(2).tolist()} | "
        f"apex height={cut_plan['apex_height_mm']:.2f} mm | "
        f"yaw local XY={cut_plan['yaw_in_local_xy_deg']:.1f} deg"
    )
    print(f"local lesion PLY: {out_dir / f'{stem}_local_lesion_points.ply'}")
    print(f"local context PLY: {out_dir / f'{stem}_local_context_plane.ply'}")
    print(f"height map: {out_dir / f'{stem}_height_map.png'}")

    if args.preview:
        try:
            import open3d as o3d  # type: ignore
        except Exception as exc:
            raise SystemExit("Open3D is not installed. Run: python -m pip install -r requirements-3d.txt") from exc
        pcd = o3d.io.read_point_cloud(str(out_dir / f"{stem}_local_context_plane.ply"))
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0.0, 0.0, 0.0])
        o3d.visualization.draw_geometries([pcd, axis], window_name="Local lesion frame")


if __name__ == "__main__":
    main()
