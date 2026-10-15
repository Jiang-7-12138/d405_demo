from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from io_utils import imwrite_unicode
from segmenter import PhantomMaterialSegmenter
from visualization import draw_named_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple OpenCV viewer for a USB endoscope.")
    parser.add_argument("--index", type=int, default=1, help="OpenCV camera index. Bellysim endoscope was detected as index 1 on this PC.")
    parser.add_argument("--width", type=int, default=800, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=600, help="Requested capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Requested capture FPS.")
    parser.add_argument("--record-dir", default="captures", help="Directory for snapshots and videos.")
    parser.add_argument("--backend", choices=["any", "dshow", "msmf"], default="any")
    return parser.parse_args()


def backend_value(name: str) -> int:
    if name == "dshow":
        return cv2.CAP_DSHOW
    if name == "msmf":
        return cv2.CAP_MSMF
    return cv2.CAP_ANY


def recognize_frame(frame: np.ndarray, segmenter: PhantomMaterialSegmenter) -> tuple[np.ndarray, list[str]]:
    """Draw the current painted-phantom classes. No depth, so the lesion circle uses the mask centroid."""
    masks = segmenter.predict_all(frame)
    found = [name for name, mask in masks.items() if mask is not None and np.any(mask)]
    return draw_named_masks(frame, masks), found


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parent
    record_dir = base / args.record_dir
    record_dir.mkdir(parents=True, exist_ok=True)

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
    print("Recognition profile: painted (liver, bowel, omentum, gallbladder, vessel, lesion)")

    segmenter = PhantomMaterialSegmenter(profile="painted")
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
            if show_masks:
                shown, found = recognize_frame(frame, segmenter)
            else:
                shown = frame.copy()
            status = "REC" if writer is not None else "LIVE"
            mode = "painted" if show_masks else "raw"
            classes = ",".join(found) if found else "none"
            label = f"{status} {mode} {frame.shape[1]}x{frame.shape[0]} {classes}"
            cv2.putText(shown, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(shown, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(shown)

            cv2.imshow(win, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = record_dir / f"endoscope_{stamp}.png"
                imwrite_unicode(str(path), frame)
                overlay_path = record_dir / f"endoscope_{stamp}_overlay.png"
                overlay, found_now = recognize_frame(frame, segmenter)
                imwrite_unicode(str(overlay_path), overlay)
                print(f"Saved snapshot: {path}")
                print(f"Saved overlay: {overlay_path} classes={','.join(found_now) if found_now else 'none'}")
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
