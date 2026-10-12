from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from io_utils import imwrite_unicode


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
    print("Keys: q/ESC quit | s snapshot | r start/stop video recording")

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

            shown = frame.copy()
            status = "REC" if writer is not None else "LIVE"
            cv2.putText(shown, f"{status} index={args.index} {frame.shape[1]}x{frame.shape[0]}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(shown, f"{status} index={args.index} {frame.shape[1]}x{frame.shape[0]}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            if writer is not None:
                writer.write(frame)

            cv2.imshow(win, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                path = record_dir / f"endoscope_{time.strftime('%Y%m%d_%H%M%S')}.png"
                imwrite_unicode(str(path), frame)
                print(f"Saved snapshot: {path}")
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
