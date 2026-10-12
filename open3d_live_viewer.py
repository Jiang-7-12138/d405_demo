from __future__ import annotations

import numpy as np


class Open3DUnavailable(RuntimeError):
    pass


class Open3DLiveViewer:
    """Small non-blocking Open3D point cloud viewer.

    The main OpenCV demo can keep running while this window shows the actual
    3D point cloud. Open3D is intentionally imported lazily so the rest of the
    project still runs on machines without it installed.
    """

    def __init__(self, window_name: str = "D405 3D point cloud", max_points: int = 120000) -> None:
        try:
            import open3d as o3d  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local env
            raise Open3DUnavailable("Open3D is not installed. Run: python -m pip install open3d") from exc

        self.o3d = o3d
        self.max_points = int(max_points)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name, width=1000, height=800)
        self.pcd = o3d.geometry.PointCloud()
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0.0, 0.0, 0.0])
        self.grid = self._make_reference_grid(size_m=0.40, step_m=0.05)
        self._added = False

    def update(
        self,
        points_m: np.ndarray,
        colors_bgr: np.ndarray | None = None,
        highlight_points_m: np.ndarray | None = None,
    ) -> None:
        if points_m is None or points_m.size == 0:
            self.vis.poll_events()
            self.vis.update_renderer()
            return

        points = points_m
        colors = colors_bgr
        if highlight_points_m is not None and highlight_points_m.size > 0:
            points = np.vstack([points_m, highlight_points_m])
            red = np.tile(np.array([[0, 0, 255]], dtype=np.uint8), (highlight_points_m.shape[0], 1))
            if colors_bgr is None:
                base = np.tile(np.array([[180, 220, 220]], dtype=np.uint8), (points_m.shape[0], 1))
                colors = np.vstack([base, red])
            else:
                colors = np.vstack([colors_bgr, red])

        if points.shape[0] > self.max_points:
            idx = np.linspace(0, points.shape[0] - 1, self.max_points).astype(np.int64)
            points = points[idx]
            if colors is not None:
                colors = colors[idx]

        # RealSense camera coordinates use +Y down; Open3D display coordinates
        # use +Y up. Flip only the visualization copy.
        display_points = points.astype(np.float64).copy()
        display_points[:, 1] *= -1.0
        self.pcd.points = self.o3d.utility.Vector3dVector(display_points)
        if colors is not None:
            rgb = colors[:, ::-1].astype(np.float64) / 255.0
            self.pcd.colors = self.o3d.utility.Vector3dVector(rgb)

        if not self._added:
            self.vis.add_geometry(self.pcd)
            self.vis.add_geometry(self.axis)
            self.vis.add_geometry(self.grid)
            render = self.vis.get_render_option()
            render.point_size = 2.0
            render.background_color = np.asarray([0.02, 0.02, 0.02])
            self._added = True
        else:
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self) -> None:
        self.vis.destroy_window()

    def _make_reference_grid(self, size_m: float = 0.40, step_m: float = 0.05):
        """Create an X-Z floor grid in Open3D display coordinates.

        RealSense camera points are displayed as X right, Y up, Z forward
        after flipping camera +Y down to display +Y up.
        """
        lines = []
        points = []
        half = size_m / 2.0
        values = np.arange(-half, half + step_m * 0.5, step_m)
        for x in values:
            points.append([float(x), 0.0, -half])
            points.append([float(x), 0.0, half])
            lines.append([len(points) - 2, len(points) - 1])
        for z in values:
            points.append([-half, 0.0, float(z)])
            points.append([half, 0.0, float(z)])
            lines.append([len(points) - 2, len(points) - 1])
        grid = self.o3d.geometry.LineSet()
        grid.points = self.o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        grid.lines = self.o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        grid.colors = self.o3d.utility.Vector3dVector(np.tile(np.array([[0.32, 0.32, 0.32]]), (len(lines), 1)))
        return grid


def clean_point_cloud_open3d(
    points_m: np.ndarray,
    colors_bgr: np.ndarray | None = None,
    voxel_size_m: float = 0.0015,
    remove_outliers: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Optional Open3D cleanup for saved or displayed point clouds."""
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        raise Open3DUnavailable("Open3D is not installed. Run: python -m pip install open3d") from exc

    if points_m.size == 0:
        return points_m, colors_bgr

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_m.astype(np.float64))
    if colors_bgr is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors_bgr[:, ::-1].astype(np.float64) / 255.0)

    if voxel_size_m > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size_m))
    if remove_outliers and len(pcd.points) >= 50:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.8)
    if len(pcd.points) >= 30:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )

    points = np.asarray(pcd.points, dtype=np.float32)
    colors = None
    if pcd.has_colors():
        colors = (np.asarray(pcd.colors)[:, ::-1] * 255.0).clip(0, 255).astype(np.uint8)
    return points, colors
