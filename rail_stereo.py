"""Metric depth from one industrial camera translated on a rail.

The camera stays a single lens. Depth comes from two photos of a static scene
taken a known distance apart, with that motion perpendicular to the optical
axis. Motion along the optical axis only changes scale and is rejected by the
matcher.

Vertical rail (camera looking horizontally, rail moving up/down): disparity is
along image v, so both frames are rotated 90 degrees counterclockwise, matched
with the usual horizontal stereo matcher, then rotated back.
Z = f_axis * baseline / disparity.

Image A is the first station. Image B is the camera after it moved right
(horizontal) or down in the image (vertical). If almost no depth comes out,
pass --swap.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from io_utils import imwrite_unicode
from pointcloud_viewer import write_ply_ascii


@dataclass
class RailIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class RailResult:
    depth_m: np.ndarray
    disparity: np.ndarray
    points_m: np.ndarray
    colors_bgr: np.ndarray | None
    valid_pixels: int
    median_depth_m: float | None


def imread_unicode(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def load_intrinsics(path: str | Path) -> RailIntrinsics:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    dist = np.array(data.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1)
    return RailIntrinsics(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        dist=dist,
    )


def save_intrinsics(path: str | Path, intr: RailIntrinsics, reprojection_error_px: float) -> None:
    payload = {
        "width": intr.width,
        "height": intr.height,
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
        "dist_coeffs": [float(v) for v in intr.dist.reshape(-1)],
        "reprojection_error_px": reprojection_error_px,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def intrinsics_from_args(args: argparse.Namespace, image_shape: tuple[int, int]) -> RailIntrinsics:
    height, width = image_shape
    if args.intrinsics:
        intr = load_intrinsics(args.intrinsics)
        if (intr.width, intr.height) != (width, height):
            raise ValueError(
                f"Intrinsics are {intr.width}x{intr.height}, images are {width}x{height}."
            )
        return intr
    if args.fx is None or args.fy is None:
        raise ValueError("Pass --intrinsics, or both --fx and --fy. A guessed focal length has no metric scale.")
    dist = np.zeros(5, dtype=np.float64)
    if args.dist:
        dist = np.array([float(v) for v in args.dist], dtype=np.float64)
    return RailIntrinsics(
        width=width,
        height=height,
        fx=float(args.fx),
        fy=float(args.fy),
        cx=float(args.cx if args.cx is not None else (width - 1) / 2.0),
        cy=float(args.cy if args.cy is not None else (height - 1) / 2.0),
        dist=dist,
    )


def _undistort(image: np.ndarray, intr: RailIntrinsics) -> tuple[np.ndarray, RailIntrinsics]:
    if intr.dist.size == 0 or np.allclose(intr.dist, 0):
        return image, intr
    size = (intr.width, intr.height)
    new_k, _ = cv2.getOptimalNewCameraMatrix(intr.matrix, intr.dist, size, 0)
    undistorted = cv2.undistort(image, intr.matrix, intr.dist, None, new_k)
    updated = RailIntrinsics(
        width=intr.width,
        height=intr.height,
        fx=float(new_k[0, 0]),
        fy=float(new_k[1, 1]),
        cx=float(new_k[0, 2]),
        cy=float(new_k[1, 2]),
        dist=np.zeros(5, dtype=np.float64),
    )
    return undistorted, updated


def unrotate_ccw(rotated: np.ndarray, width: int) -> np.ndarray:
    """Inverse of cv2.ROTATE_90_COUNTERCLOCKWISE back to an image of this width."""
    if rotated.shape[0] != width:
        raise ValueError(f"Rotated map height {rotated.shape[0]} does not match original width {width}.")
    return rotated[::-1, :].T


def _align_num_disparities(num_disparities: int, match_width: int) -> int:
    count = max(16, int(num_disparities))
    count = int(np.ceil(count / 16.0) * 16)
    limit = max(16, (match_width // 16) * 16)
    return min(count, limit)


def match_disparity(left: np.ndarray, right: np.ndarray, num_disparities: int, block_size: int) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"Stereo pair sizes differ: {left.shape} vs {right.shape}.")
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    block = max(3, int(block_size) | 1)
    count = _align_num_disparities(num_disparities, gray_left.shape[1])
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=count,
        blockSize=block,
        P1=8 * 3 * block * block,
        P2=32 * 3 * block * block,
        disp12MaxDiff=2,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    raw = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    raw[raw <= 0] = 0
    return raw


def depth_from_disparity(disparity: np.ndarray, focal_px: float, baseline_m: float) -> np.ndarray:
    depth = np.zeros(disparity.shape, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0.5)
    depth[valid] = (float(focal_px) * float(baseline_m) / disparity[valid]).astype(np.float32)
    return depth


def reconstruct_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    intr: RailIntrinsics,
    baseline_m: float,
    axis: str = "vertical",
    swap: bool = False,
    num_disparities: int = 128,
    block_size: int = 5,
    min_depth_m: float = 0.02,
    max_depth_m: float = 0.80,
    stride: int = 2,
) -> RailResult:
    if image_a.shape != image_b.shape:
        raise ValueError(f"Image sizes differ: {image_a.shape} vs {image_b.shape}.")
    if axis not in ("vertical", "horizontal"):
        raise ValueError("axis must be 'vertical' or 'horizontal'.")
    if baseline_m <= 0:
        raise ValueError("Baseline must be positive.")

    if swap:
        image_a, image_b = image_b, image_a
    image_a, intr = _undistort(image_a, intr)
    image_b, intr = _undistort(image_b, intr)

    if axis == "vertical":
        left = cv2.rotate(image_a, cv2.ROTATE_90_COUNTERCLOCKWISE)
        right = cv2.rotate(image_b, cv2.ROTATE_90_COUNTERCLOCKWISE)
        focal = intr.fy
    else:
        left, right = image_a, image_b
        focal = intr.fx

    disparity = match_disparity(left, right, num_disparities, block_size)
    if axis == "vertical":
        disparity = unrotate_ccw(disparity, intr.width)

    depth = depth_from_disparity(disparity, focal, baseline_m)
    depth[(depth < min_depth_m) | (depth > max_depth_m)] = 0
    points, colors = _depth_to_points(depth, image_a, intr, stride)
    valid = depth > 0
    valid_pixels = int(np.count_nonzero(valid))
    median = float(np.median(depth[valid])) if valid_pixels else None
    return RailResult(depth, disparity, points, colors, valid_pixels, median)


def _depth_to_points(
    depth_m: np.ndarray,
    color_bgr: np.ndarray,
    intr: RailIntrinsics,
    stride: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    step = max(1, int(stride))
    ys = np.arange(0, depth_m.shape[0], step, dtype=np.int32)
    xs = np.arange(0, depth_m.shape[1], step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth_m[grid_y, grid_x]
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), None
    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (u - intr.cx) * z / intr.fx
    y = (v - intr.cy) * z / intr.fy
    points = np.column_stack((x, y, z)).astype(np.float32)
    colors = color_bgr[grid_y[valid], grid_x[valid]].astype(np.uint8)
    return points, colors


def depth_preview(color_bgr: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    valid = depth_m > 0
    preview = np.zeros_like(color_bgr)
    if np.any(valid):
        lo, hi = np.percentile(depth_m[valid], [5, 95])
        span = max(float(hi - lo), 1e-6)
        scaled = np.clip((depth_m - lo) / span, 0, 1)
        gray = (scaled * 255).astype(np.uint8)
        gray[~valid] = 0
        preview = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        preview[~valid] = 0
    return np.hstack((color_bgr, preview))


def save_reconstruction(out_dir: str | Path, result: RailResult, image_a: np.ndarray, meta: dict) -> Path:
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    depth_mm = np.clip(np.round(result.depth_m * 1000.0), 0, 65535).astype(np.uint16)
    imwrite_unicode(str(folder / "depth_mm.png"), depth_mm)
    imwrite_unicode(str(folder / "preview.png"), depth_preview(image_a, result.depth_m))
    np.save(str(folder / "depth_m.npy"), result.depth_m)
    write_ply_ascii(str(folder / "cloud.ply"), result.points_m, result.colors_bgr)
    (folder / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return folder


def _find_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    for candidate in (gray, clahe):
        found, corners = cv2.findChessboardCornersSB(candidate, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found and corners is not None:
            return corners.reshape(-1, 2).astype(np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for candidate in (gray, clahe):
        found, corners = cv2.findChessboardCorners(candidate, pattern, flags=flags)
        if found and corners is not None:
            refined = cv2.cornerSubPix(
                candidate,
                corners.astype(np.float32),
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3),
            )
            return refined.reshape(-1, 2).astype(np.float32)
    return None


def calibrate_intrinsics(
    image_paths: list[Path],
    inner_corners: tuple[int, int],
    square_size_m: float,
) -> tuple[RailIntrinsics, float]:
    pattern = (int(inner_corners[0]), int(inner_corners[1]))
    obj = np.zeros((pattern[0] * pattern[1], 3), dtype=np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * float(square_size_m)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    used: list[str] = []
    for path in image_paths:
        image = imread_unicode(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = size
        elif size != image_size:
            print(f"skip {path.name}: size {size[0]}x{size[1]} != {image_size[0]}x{image_size[1]}")
            continue
        corners = _find_corners(gray, pattern)
        if corners is None:
            print(f"skip {path.name}: checkerboard not found")
            continue
        object_points.append(obj.copy())
        image_points.append(corners)
        used.append(path.name)
    if image_size is None or len(object_points) < 3:
        raise RuntimeError(f"Need at least 3 checkerboard views, found {len(object_points)}.")
    rms, camera_matrix, dist, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    intr = RailIntrinsics(
        width=image_size[0],
        height=image_size[1],
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        dist=dist.reshape(-1).astype(np.float64),
    )
    print(f"used {len(used)} views, reprojection RMS {float(rms):.3f} px")
    return intr, float(rms)


def _collect_images(source: str) -> list[Path]:
    path = Path(source)
    if path.is_dir():
        images = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            images.extend(path.glob(ext))
        return sorted(images)
    if any(ch in source for ch in "*?["):
        return sorted(Path().glob(source))
    if path.is_file():
        return [path]
    raise FileNotFoundError(source)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rail translation stereo for one industrial camera.")
    sub = parser.add_subparsers(dest="command", required=True)

    calibrate = sub.add_parser("calibrate", help="Calibrate intrinsics from checkerboard photos.")
    calibrate.add_argument("--images", required=True, help="Directory or glob of checkerboard photos.")
    calibrate.add_argument("--inner-corners", type=int, nargs=2, default=(6, 6), metavar=("COLS", "ROWS"))
    calibrate.add_argument("--square-mm", type=float, default=20.0)
    calibrate.add_argument("--out", default="captures/rail_intrinsics.json")

    reconstruct = sub.add_parser("reconstruct", help="Build depth and a point cloud from two rail photos.")
    reconstruct.add_argument("--image-a", required=True, help="Photo at the first rail station.")
    reconstruct.add_argument("--image-b", required=True, help="Photo after moving right, or down for a vertical rail.")
    reconstruct.add_argument("--baseline-mm", type=float, required=True, help="Measured rail travel between the two photos.")
    reconstruct.add_argument("--axis", choices=("vertical", "horizontal"), default="vertical")
    reconstruct.add_argument("--intrinsics", default=None, help="JSON from the calibrate command.")
    reconstruct.add_argument("--fx", type=float, default=None)
    reconstruct.add_argument("--fy", type=float, default=None)
    reconstruct.add_argument("--cx", type=float, default=None)
    reconstruct.add_argument("--cy", type=float, default=None)
    reconstruct.add_argument("--dist", type=float, nargs="*", default=None, help="Distortion coefficients, usually 5 numbers.")
    reconstruct.add_argument("--swap", action="store_true", help="Swap the two photos if the camera moved the other way.")
    reconstruct.add_argument("--num-disparities", type=int, default=128)
    reconstruct.add_argument("--block-size", type=int, default=5)
    reconstruct.add_argument("--min-depth-mm", type=float, default=20.0)
    reconstruct.add_argument("--max-depth-mm", type=float, default=800.0)
    reconstruct.add_argument("--stride", type=int, default=2)
    reconstruct.add_argument("--out-dir", default="captures/rail_recon")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "calibrate":
        images = _collect_images(args.images)
        intr, rms = calibrate_intrinsics(images, tuple(args.inner_corners), args.square_mm / 1000.0)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_intrinsics(out, intr, rms)
        print(f"wrote {out}")
        return

    image_a = imread_unicode(args.image_a)
    image_b = imread_unicode(args.image_b)
    intr = intrinsics_from_args(args, image_a.shape[:2])
    result = reconstruct_pair(
        image_a,
        image_b,
        intr,
        baseline_m=args.baseline_mm / 1000.0,
        axis=args.axis,
        swap=args.swap,
        num_disparities=args.num_disparities,
        block_size=args.block_size,
        min_depth_m=args.min_depth_mm / 1000.0,
        max_depth_m=args.max_depth_mm / 1000.0,
        stride=args.stride,
    )
    meta = {
        "image_a": str(args.image_a),
        "image_b": str(args.image_b),
        "baseline_mm": args.baseline_mm,
        "axis": args.axis,
        "swapped": bool(args.swap),
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.cx,
        "cy": intr.cy,
        "valid_pixels": result.valid_pixels,
        "median_depth_mm": None if result.median_depth_m is None else result.median_depth_m * 1000.0,
        "points": int(result.points_m.shape[0]),
    }
    folder = save_reconstruction(args.out_dir, result, image_a, meta)
    median = "none" if result.median_depth_m is None else f"{result.median_depth_m * 1000.0:.1f} mm"
    print(f"valid pixels {result.valid_pixels}, median depth {median}, points {result.points_m.shape[0]}")
    print(f"wrote {folder}")


if __name__ == "__main__":
    main()
