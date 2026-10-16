"""Run the trained organ detector on the USB endoscope, a video, or one image."""

from __future__ import annotations

import argparse
import ctypes
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from io_utils import imwrite_unicode

DEFAULT_WEIGHTS = str(Path(__file__).resolve().parent / "weights" / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO organ detection on the USB endoscope.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Trained detect weights, best.pt.")
    parser.add_argument("--conf", type=float, default=0.25, help="Box confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--index", type=int, default=1, help="OpenCV camera index. This endoscope was index 1.")
    parser.add_argument("--width", type=int, default=800, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=600, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested capture FPS.")
    parser.add_argument("--backend", choices=["any", "dshow", "msmf"], default="any")
    parser.add_argument("--video", default="", help="Run on this video instead of the camera.")
    parser.add_argument("--image", default="", help="Run on one image and save the overlay.")
    parser.add_argument("--sample", type=int, default=0, help="With --video, save this many evenly spaced frames instead of a full overlay video.")
    parser.add_argument("--record-dir", default="captures/yolo_preview", help="Directory for overlays and snapshots.")
    parser.add_argument("--device", default="cpu", help="Inference device. This machine has a CPU torch build.")
    return parser.parse_args()


def backend_value(name: str) -> int:
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def poll_key(keys: tuple[int, ...]) -> int:
    """Read a key even when a Chinese IME would swallow cv2.waitKey."""
    key_code = cv2.waitKey(1)
    key = (key_code & 0xFF) if key_code >= 0 else -1
    if os.name != "nt":
        return key
    user32 = ctypes.windll.user32
    for virtual_key in keys:
        if user32.GetAsyncKeyState(virtual_key) & 1:
            return virtual_key
    return key


def load_model(weights: str, device: str) -> YOLO:
    path = Path(weights)
    if not path.is_file():
        raise SystemExit(f"Weights not found: {path}")
    model = YOLO(str(path))
    if model.task != "detect":
        raise SystemExit(f"Expected a detect model, got task={model.task}")
    print(f"Weights: {path}")
    print("Classes: " + ", ".join(f"{idx}:{name}" for idx, name in model.names.items()))
    print(f"Device: {device}")
    return model


def detect(model: YOLO, frame: np.ndarray, conf: float, imgsz: int, device: str):
    return model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]


def best_lesion_box(result) -> tuple[float, tuple[int, int, int, int]] | None:
    """Highest-confidence lesion box. Other classes stay as plain boxes."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for box in result.boxes:
        if result.names[int(box.cls.item())] != "lesion":
            continue
        score = float(box.conf.item())
        xyxy = tuple(int(round(value)) for value in box.xyxy[0].tolist())
        if best is None or score > best[0]:
            best = (score, xyxy)
    return best


def lesion_circle(frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int] | None:
    """Center of the round red dome inside the YOLO lesion box.

    The glare sits uphill of the middle, so the brightest pixel is not the
    center. The grooves are concentric in hue/saturation/value, and the search
    picks the point where that color stays even around each ring.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = ((hue <= 12) | (hue >= 170)) & (sat >= 150) & (val >= 40) & (val <= 140)
    value = val.astype(np.float32)
    step = 2
    margin = 8
    xs = np.arange(x1 + margin, x2 - margin, step)
    ys = np.arange(y1 + margin, y2 - margin, step)
    if xs.size == 0 or ys.size == 0:
        return None
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat_x = grid_x.ravel()
    flat_y = grid_y.ravel()
    on_red = red[flat_y, flat_x]
    if not np.any(on_red):
        return None
    flat_x = flat_x[on_red]
    flat_y = flat_y[on_red]
    radii = np.arange(10, int(min(x2 - x1, y2 - y1) * 0.46), 3)
    if radii.size < 3:
        return None
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    sample_x = flat_x[:, None, None] + radii[None, :, None] * np.cos(angles)[None, None, :]
    sample_y = flat_y[:, None, None] + radii[None, :, None] * np.sin(angles)[None, None, :]
    ix = np.clip(np.rint(sample_x).astype(np.int32), 0, width - 1)
    iy = np.clip(np.rint(sample_y).astype(np.int32), 0, height - 1)
    samples = value[iy, ix]
    angular_std = samples.std(axis=2).mean(axis=1)
    radial_contrast = samples.mean(axis=2).std(axis=1)
    score = radial_contrast / (angular_std + 1.0)
    winner = int(np.argmax(score))
    if float(score[winner]) < 0.35:
        return None
    cx = float(flat_x[winner])
    cy = float(flat_y[winner])
    radius = _dome_radius(value, cx, cy, (x1, y1, x2, y2))
    return int(round(cx)), int(round(cy)), radius


def _dome_radius(value: np.ndarray, cx: float, cy: float, box: tuple[int, int, int, int]) -> int:
    """Outer radius that is still a round ring, before the square corners."""
    x1, y1, x2, y2 = box
    height, width = value.shape
    limit = int(min(cx - x1, x2 - cx, cy - y1, y2 - cy, 70))
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    radius = 16
    for candidate in range(14, max(16, limit) + 1, 2):
        ix = np.clip(np.rint(cx + candidate * np.cos(angles)).astype(np.int32), 0, width - 1)
        iy = np.clip(np.rint(cy + candidate * np.sin(angles)).astype(np.int32), 0, height - 1)
        if float(value[iy, ix].std()) > 16.0:
            if candidate > 24:
                break
            continue
        radius = candidate
    return radius


def draw_lesion_center(shown: np.ndarray, center: tuple[int, int, int] | None) -> None:
    if center is None:
        return
    cx, cy, radius = center
    color = (0, 255, 255)
    cv2.circle(shown, (cx, cy), radius, color, 2, cv2.LINE_AA)
    cv2.drawMarker(shown, (cx, cy), color, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)


def overlay_frame(frame: np.ndarray, result) -> tuple[np.ndarray, str]:
    shown = result.plot()
    lesion = best_lesion_box(result)
    center = lesion_circle(frame, lesion[1]) if lesion is not None else None
    draw_lesion_center(shown, center)
    parts: list[str] = []
    if result.boxes is not None and len(result.boxes):
        order = np.argsort(-result.boxes.conf.cpu().numpy())
        for index in order:
            cls_id = int(result.boxes.cls[index].item())
            score = float(result.boxes.conf[index].item())
            name = result.names[cls_id]
            if name == "lesion" and center is not None and lesion is not None and abs(score - lesion[0]) < 1e-3:
                cx, cy, radius = center
                parts.append(f"lesion {score:.2f} center=({cx},{cy}) r={radius}")
            else:
                parts.append(f"{name} {score:.2f}")
    summary = ", ".join(parts) if parts else "none"
    return shown, summary


def run_image(model: YOLO, image_path: Path, record_dir: Path, conf: float, imgsz: int, device: str) -> None:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"Failed to read image: {image_path}")
    shown, summary = overlay_frame(frame, detect(model, frame, conf, imgsz, device))
    out_path = record_dir / f"{image_path.stem}_yolo.jpg"
    imwrite_unicode(str(out_path), shown)
    print(f"{image_path.name}: {summary}")
    print(f"Saved: {out_path}")


def sample_indices(frame_count: int, sample: int) -> list[int]:
    if frame_count <= 0 or sample <= 0:
        return []
    if sample >= frame_count:
        return list(range(frame_count))
    return [int(round(i * (frame_count - 1) / (sample - 1))) for i in range(sample)]


def run_video(model: YOLO, video_path: Path, record_dir: Path, conf: float, imgsz: int, device: str, sample: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Reading {video_path}")
    print(f"Size {width}x{height}, fps={fps:.1f}, frames={frame_count}")

    if sample > 0:
        wanted = set(sample_indices(frame_count, sample))
        print(f"Saving {len(wanted)} frames to {record_dir}")
        index = 0
        saved = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if index in wanted:
                    shown, summary = overlay_frame(frame, detect(model, frame, conf, imgsz, device))
                    out_path = record_dir / f"{video_path.stem}_f{index:06d}_yolo.jpg"
                    imwrite_unicode(str(out_path), shown)
                    print(f"  f{index:06d}: {summary}")
                    print(f"  saved {out_path.name}")
                    saved += 1
                index += 1
        finally:
            cap.release()
        print(f"Done. {saved} overlay images in {record_dir}")
        return

    out_path = record_dir / f"{video_path.stem}_yolo.avi"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to create overlay video: {out_path}")
    print(f"Writing {out_path}")
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            shown, summary = overlay_frame(frame, detect(model, frame, conf, imgsz, device))
            writer.write(shown)
            written += 1
            if written % 30 == 0:
                print(f"  {written}/{frame_count} {summary}")
    finally:
        writer.release()
        cap.release()
    print(f"Done. Overlay video: {out_path} ({written} frames)")


def run_camera(model: YOLO, args: argparse.Namespace, record_dir: Path) -> None:
    cap = cv2.VideoCapture(args.index, backend_value(args.backend))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera index {args.index}. Try --index 0 or --backend dshow/msmf.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Opened camera index={args.index}, size={actual_w}x{actual_h}")
    print("Keys: q/ESC quit | s snapshot. Click the image window first.")

    win = f"YOLO endoscope index {args.index}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame")
                break
            t0 = time.perf_counter()
            shown, summary = overlay_frame(frame, detect(model, frame, args.conf, args.imgsz, args.device))
            elapsed = time.perf_counter() - t0
            fps_text = f"{1.0 / elapsed:.1f} fps" if elapsed > 0 else ""
            cv2.putText(shown, fps_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(shown, fps_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, shown)
            key = poll_key((0x1B, ord("q"), ord("s")))
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                raw_path = record_dir / f"endoscope_{stamp}.png"
                overlay_path = record_dir / f"endoscope_{stamp}_yolo.jpg"
                imwrite_unicode(str(raw_path), frame)
                imwrite_unicode(str(overlay_path), shown)
                print(f"Saved {raw_path}")
                print(f"Saved {overlay_path}: {summary}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    record_dir = Path(__file__).resolve().parent / args.record_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights, args.device)
    if args.image:
        run_image(model, Path(args.image), record_dir, args.conf, args.imgsz, args.device)
        return
    if args.video:
        run_video(model, Path(args.video), record_dir, args.conf, args.imgsz, args.device, args.sample)
        return
    run_camera(model, args, record_dir)


if __name__ == "__main__":
    main()
