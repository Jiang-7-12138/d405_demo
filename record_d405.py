"""Record a D405 color video. Press r to start or stop, q or Esc to quit."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from camera_d405 import RealSenseD405


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a color video from the Intel RealSense D405.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--record-dir", default="captures", help="Directory for the saved video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_dir = Path(__file__).resolve().parent / args.record_dir
    record_dir.mkdir(parents=True, exist_ok=True)

    devices = RealSenseD405.list_devices()
    if not devices:
        raise SystemExit("No RealSense device found. Check the D405 cable and close other camera programs.")
    for name, serial in devices:
        print(f"RealSense: {name} serial={serial}")

    camera = RealSenseD405(args.width, args.height, args.fps)
    writer = None
    video_path = None
    win = "D405 record"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("Keys: r start/stop recording | q or Esc quit")

    camera.start()
    try:
        while True:
            frame = camera.read()
            color = frame.color_bgr
            shown = color.copy()
            if writer is not None:
                writer.write(color)
                cv2.putText(shown, "REC", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, shown)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                if writer is None:
                    video_path = record_dir / f"d405_{time.strftime('%Y%m%d_%H%M%S')}.avi"
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"MJPG"),
                        float(args.fps),
                        (color.shape[1], color.shape[0]),
                    )
                    if not writer.isOpened():
                        writer = None
                        print("Failed to start recording")
                    else:
                        print(f"Recording: {video_path}")
                else:
                    writer.release()
                    writer = None
                    print(f"Saved: {video_path}")
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved: {video_path}")
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
