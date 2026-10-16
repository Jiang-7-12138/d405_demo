from __future__ import annotations

import argparse
import ctypes
import os
import time
from pathlib import Path

import cv2
import numpy as np

from io_utils import imwrite_unicode
from marker_frame import MarkerFrameHold, detect_marker_frame, draw_marker_frame
from segmenter import PhantomMaterialSegmenter
from visualization import draw_named_masks, lesion_area_radius_px, mask_centroid, mask_interior_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple OpenCV viewer for a USB endoscope.")
    parser.add_argument("--index", type=int, default=1, help="OpenCV camera index. Bellysim endoscope was detected as index 1 on this PC.")
    parser.add_argument("--width", type=int, default=800, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=600, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested capture FPS.")
    parser.add_argument("--record-dir", default="captures", help="Directory for snapshots and videos.")
    parser.add_argument("--backend", choices=["any", "dshow", "msmf"], default="any")
    parser.add_argument("--video", default="", help="Run recognition on this video file and save an overlay video. Skips the camera.")
    parser.add_argument(
        "--profile",
        default="endoscope-bright",
        choices=["endoscope", "endoscope-bright", "endoscope-dark", "endoscope-deep"],
        help="endoscope is the pale pink dome. endoscope-bright is the bright red dome. endoscope-dark is the dark dome with a lighter cap. endoscope-deep is the final deep-red ringed dome.",
    )
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


class MaskHold:
    """Keep a displayed mask from flickering by taking the recent-frame majority."""

    def __init__(self, history: int = 7, hold_frames: int = 8) -> None:
        self.history = max(1, int(history))
        self.hold_frames = max(0, int(hold_frames))
        self._frames: list[np.ndarray] = []
        self._missing = 0
        self._held: np.ndarray | None = None

    def update(self, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None or not np.any(mask):
            self._missing += 1
            if self._missing > self.hold_frames:
                self._frames.clear()
                self._held = None
            return self._held
        self._missing = 0
        self._frames.append(mask.astype(bool))
        self._frames = self._frames[-self.history :]
        votes = np.sum(np.stack(self._frames, axis=0), axis=0)
        self._held = votes >= (len(self._frames) + 1) // 2
        return self._held


class LesionCenterHold:
    """Keep the displayed lesion center from twitching by a few pixels.

    A new measurement inside the deadband leaves the held center unchanged.
    A real move has to stay in the recent median before the held center steps.
    A short gap keeps the last center instead of blinking off.
    """

    def __init__(self, deadband_px: float = 4.0, history: int = 9, hold_frames: int = 8) -> None:
        self.deadband_px = float(deadband_px)
        self.history = max(1, int(history))
        self.hold_frames = max(0, int(hold_frames))
        self._samples: list[tuple[float, float]] = []
        self._held: tuple[int, int] | None = None
        self._missing = 0

    def update(self, center: tuple[int, int] | None) -> tuple[int, int] | None:
        if center is None:
            self._missing += 1
            if self._missing > self.hold_frames:
                self._samples.clear()
                self._held = None
            return self._held
        self._missing = 0
        self._samples.append((float(center[0]), float(center[1])))
        self._samples = self._samples[-self.history :]
        xs = sorted(sample[0] for sample in self._samples)
        ys = sorted(sample[1] for sample in self._samples)
        mid = len(xs) // 2
        median = (xs[mid], ys[mid])
        if self._held is None:
            self._held = (int(round(median[0])), int(round(median[1])))
            return self._held
        shift = ((median[0] - self._held[0]) ** 2 + (median[1] - self._held[1]) ** 2) ** 0.5
        if shift >= self.deadband_px:
            self._held = (int(round(median[0])), int(round(median[1])))
        return self._held


def recognize_frame(
    frame: np.ndarray,
    segmenter: PhantomMaterialSegmenter,
    center_hold: LesionCenterHold | None = None,
    bowel_hold: MaskHold | None = None,
    bowel_label_hold: LesionCenterHold | None = None,
) -> tuple[np.ndarray, list[str], tuple[int, int] | None]:
    """Draw the painted classes and a lesion center that stays put under small jitter."""
    masks = segmenter.predict_all(frame)
    masks = dict(masks)
    if bowel_hold is not None:
        stable_bowel = bowel_hold.update(masks.get("bowel"))
        if stable_bowel is not None:
            masks["bowel"] = stable_bowel
    found = [name for name, mask in masks.items() if mask is not None and np.any(mask)]
    raw_center = mask_centroid(masks.get("lesion"))
    held = center_hold.update(raw_center) if center_hold is not None else raw_center
    if held is not None and masks.get("lesion") is not None and np.any(masks["lesion"]):
        radius = lesion_area_radius_px(masks["lesion"])
        if radius is not None:
            disk = np.zeros(masks["lesion"].shape, dtype=np.uint8)
            cv2.circle(disk, held, radius, 255, thickness=cv2.FILLED)
            masks["lesion"] = disk.astype(bool)
    bowel_label = mask_interior_point(masks.get("bowel"))
    if bowel_label_hold is not None:
        bowel_label = bowel_label_hold.update(bowel_label)
    shown = draw_named_masks(
        frame,
        masks,
        centers={"lesion": held, "bowel": bowel_label},
        center_names={"lesion"},
    )
    return shown, found, held


def replay_video(video_path: Path, record_dir: Path, profile: str) -> None:
    """Write one overlay video. Open that file to see the recognition result."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = record_dir / f"endoscope_overlay_{video_path.stem}.avi"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to create overlay video: {out_path}")

    segmenter = PhantomMaterialSegmenter(profile=profile)
    marker_hold = MarkerFrameHold()
    center_hold = LesionCenterHold(deadband_px=12.0 if profile in {"endoscope", "endoscope-dark", "endoscope-deep"} else 4.0)
    bowel_hold = MaskHold()
    bowel_label_hold = LesionCenterHold(deadband_px=8.0)
    print(f"Recognition profile: {profile}")
    print(f"Reading {video_path}")
    print(f"Size {width}x{height}, fps={fps:.1f}, frames={frame_count}")
    print(f"Writing {out_path}")
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            shown, found, lesion_center = recognize_frame(frame, segmenter, center_hold, bowel_hold, bowel_label_hold)
            marker = marker_hold.update(detect_marker_frame(frame))
            shown = draw_marker_frame(shown, marker)
            classes = ",".join(found) if found else "none"
            center_text = f" lesion=({lesion_center[0]},{lesion_center[1]})" if lesion_center is not None else ""
            marker_text = f" aruco={marker.marker_id}" if marker.valid else ""
            label = f"FILE {frame.shape[1]}x{frame.shape[0]} {classes}{center_text}{marker_text}"
            cv2.putText(shown, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(shown, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(shown)
            written += 1
            if written % 100 == 0:
                print(f"  {written}/{frame_count}")
    finally:
        writer.release()
        cap.release()
    print(f"Done. Overlay video: {out_path} ({written} frames)")
    print("Open that avi in the player to watch the recognition. It is not the live camera window.")


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    record_dir = base / args.record_dir
    record_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        replay_video(Path(args.video), record_dir, args.profile)
        return

    cap = cv2.VideoCapture(args.index, backend_value(args.backend))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera index {args.index}. Try --index 0 or --backend dshow/msmf.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Opened camera index={args.index}, size={actual_w}x{actual_h}, fps={actual_fps:.1f}")
    print("Keys: q/ESC quit | s snapshot | r start/stop video | m toggle painted recognition")
    print("Letter keys do not need Fn. Click the image window first. Chinese input can be left on.")
    print(f"Recognition profile: {args.profile}")
    if args.profile == "endoscope-bright":
        print("Lesion gate: bright red dome (H 0-8, S 80-165, V 165-240).")
    elif args.profile == "endoscope-dark":
        print("Lesion gate: dark saturated dome (S>=165, V 70-120) with a lighter cap.")
    elif args.profile == "endoscope-deep":
        print("Lesion gate: final deep-red ringed dome (S>=185, V 55-110). Center is the square, not the bright corner.")
    else:
        print("Lesion gate: pale pink dome inside the darker red square.")

    segmenter = PhantomMaterialSegmenter(profile=args.profile)
    marker_hold = MarkerFrameHold()
    center_hold = LesionCenterHold(deadband_px=12.0 if args.profile in {"endoscope", "endoscope-dark", "endoscope-deep"} else 4.0)
    bowel_hold = MaskHold()
    bowel_label_hold = LesionCenterHold(deadband_px=8.0)
    show_masks = True
    writer = None
    video_path = None
    win = f"Endoscope index {args.index}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Failed to read frame")
                break

            found: list[str] = []
            lesion_center = None
            if show_masks:
                shown, found, lesion_center = recognize_frame(frame, segmenter, center_hold, bowel_hold, bowel_label_hold)
            else:
                shown = frame.copy()
            marker = marker_hold.update(detect_marker_frame(frame))
            shown = draw_marker_frame(shown, marker)
            status = "REC" if writer is not None else "LIVE"
            mode = "painted" if show_masks else "raw"
            classes = ",".join(found) if found else "none"
            center_text = f" lesion=({lesion_center[0]},{lesion_center[1]})" if lesion_center is not None else ""
            marker_text = f" aruco={marker.marker_id}" if marker.valid else ""
            label = f"{status} {mode} {frame.shape[1]}x{frame.shape[0]} {classes}{center_text}{marker_text}"
            cv2.putText(shown, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(shown, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(shown)

            cv2.imshow(win, shown)
            key = poll_key((0x1B, ord("q"), ord("s"), ord("r"), ord("m")))
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = record_dir / f"endoscope_{stamp}.png"
                imwrite_unicode(str(path), frame)
                overlay_path = record_dir / f"endoscope_{stamp}_overlay.png"
                overlay, found_now, lesion_center_now = recognize_frame(frame, segmenter, center_hold, bowel_hold, bowel_label_hold)
                overlay = draw_marker_frame(overlay, detect_marker_frame(frame))
                imwrite_unicode(str(overlay_path), overlay)
                center_now = f" lesion=({lesion_center_now[0]},{lesion_center_now[1]})" if lesion_center_now is not None else ""
                print(f"Saved snapshot: {path}")
                print(f"Saved overlay: {overlay_path} classes={','.join(found_now) if found_now else 'none'}{center_now}")
            if key == ord("m"):
                show_masks = not show_masks
                print(f"Recognition {'on' if show_masks else 'off'}")
            if key == ord("r"):
                if writer is None:
                    video_path = record_dir / f"endoscope_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (frame.shape[1], frame.shape[0]))
                    if not writer.isOpened():
                        writer = None
                        print("Failed to start video recording")
                    else:
                        print(f"Recording: {video_path}")
                else:
                    writer.release()
                    writer = None
                    print(f"Stopped recording: {video_path}")
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
