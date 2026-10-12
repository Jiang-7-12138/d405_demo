from __future__ import annotations

import argparse

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a saved PLY point cloud in a true 3D Open3D window.")
    parser.add_argument("ply", help="Path to .ply point cloud saved by run_demo.py")
    parser.add_argument("--clean", action="store_true", help="Apply voxel downsampling and statistical outlier removal before viewing.")
    parser.add_argument("--voxel-mm", type=float, default=1.5, help="Voxel size for --clean, in millimeters.")
    parser.add_argument(
        "--camera-coordinates",
        action="store_true",
        help="Input PLY is in raw RealSense camera coordinates, so flip +Y down to +Y up for viewing.",
    )
    parser.add_argument(
        "--grid",
        choices=["scene", "camera", "none"],
        default="scene",
        help="Reference grid placement. 'scene' places it under the point cloud; 'camera' keeps it at camera origin.",
    )
    parser.add_argument("--grid-step-mm", type=float, default=25.0, help="Reference grid spacing in millimeters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:
        raise SystemExit("Open3D is not installed. Run: python -m pip install -r requirements-3d.txt") from exc

    pcd = o3d.io.read_point_cloud(args.ply)
    if pcd.is_empty():
        raise SystemExit(f"No points found in {args.ply}")

    if args.camera_coordinates:
        points = np.asarray(pcd.points)
        points[:, 1] *= -1.0
        pcd.points = o3d.utility.Vector3dVector(points)

    if args.clean:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_mm / 1000.0)
        if len(pcd.points) >= 50:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.8)
        if len(pcd.points) >= 30:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )

    geometries = [pcd]
    if args.grid == "scene":
        axis, grid = make_scene_reference(o3d, np.asarray(pcd.points), step_m=args.grid_step_mm / 1000.0)
        geometries.extend([axis, grid])
    elif args.grid == "camera":
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0.0, 0.0, 0.0])
        grid = make_reference_grid(o3d, size_m=0.50, step_m=args.grid_step_mm / 1000.0, y_m=0.0)
        geometries.extend([axis, grid])
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Saved D405 point cloud",
        width=1000,
        height=800,
    )


def make_scene_reference(o3d, points: np.ndarray, step_m: float = 0.025):
    lo = np.percentile(points, 1, axis=0)
    hi = np.percentile(points, 99, axis=0)
    pad = max(step_m * 2.0, 0.03)
    x_min, x_max = float(lo[0] - pad), float(hi[0] + pad)
    z_min, z_max = float(lo[2] - pad), float(hi[2] + pad)
    y_floor = float(lo[1] - step_m)
    origin = [x_min, y_floor, z_min]
    size = max(0.04, min(0.12, max(x_max - x_min, z_max - z_min) * 0.25))
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    grid = make_reference_grid(o3d, x_min=x_min, x_max=x_max, z_min=z_min, z_max=z_max, step_m=step_m, y_m=y_floor)
    return axis, grid


def make_reference_grid(
    o3d,
    size_m: float | None = None,
    step_m: float = 0.05,
    y_m: float = 0.0,
    x_min: float | None = None,
    x_max: float | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
):
    if size_m is not None:
        half = size_m / 2.0
        x_min, x_max = -half, half
        z_min, z_max = -half, half
    if x_min is None or x_max is None or z_min is None or z_max is None:
        raise ValueError("Grid bounds are not fully specified")
    xs = np.arange(x_min, x_max + step_m * 0.5, step_m)
    zs = np.arange(z_min, z_max + step_m * 0.5, step_m)
    points = []
    lines = []
    for x in xs:
        points.append([float(x), y_m, float(z_min)])
        points.append([float(x), y_m, float(z_max)])
        lines.append([len(points) - 2, len(points) - 1])
    for z in zs:
        points.append([float(x_min), y_m, float(z)])
        points.append([float(x_max), y_m, float(z)])
        lines.append([len(points) - 2, len(points) - 1])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    grid.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.32, 0.32, 0.32]]), (len(lines), 1)))
    return grid


if __name__ == "__main__":
    main()
