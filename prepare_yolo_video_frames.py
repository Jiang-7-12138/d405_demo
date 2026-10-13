from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract temporally distributed, sharp candidate frames and contact sheets from a video."
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--out-dir", required=True, help="Output directory for candidate images and review sheets")
    parser.add_argument("--count", type=int, default=180, help="Number of candidate frames")
    parser.add_argument("--prefix", default="clip", help="Filename prefix")
    parser.add_argument("--sheet-size", type=int, default=60, help="Frames per contact sheet")
    parser.add_argument("--sheet-cols", type=int, default=6, help="Contact sheet columns")
    parser.add_argument("--thumb-width", type=int, default=160, help="Contact sheet thumbnail width")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="Extracted image JPEG quality")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if args.count < 1 or args.sheet_size < 1 or args.sheet_cols < 1:
        raise ValueError("count, sheet-size, and sheet-cols must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_count < 1 or not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError("Video has invalid frame count or FPS metadata")

    count = min(args.count, frame_count)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Decode once in order and keep the sharpest JPEG per temporal bin.
    # Keeping compressed candidates avoids retaining hundreds of full frames.
    selected: list[tuple[int, bytes, float, int, int] | None] = [None] * count
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        bin_index = min(count - 1, frame_index * count // frame_count)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        current = selected[bin_index]
        if current is None or sharpness > current[2]:
            ok_jpeg, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            )
            if ok_jpeg:
                selected[bin_index] = (frame_index, encoded.tobytes(), sharpness, frame.shape[1], frame.shape[0])
        frame_index += 1
    capture.release()

    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    saved: list[tuple[str, int, float, float, bytes, int, int]] = []
    for order, item in enumerate(selected, start=1):
        if item is None:
            continue
        frame_index, encoded, sharpness, width, height = item
        name = f"{args.prefix}_{order:04d}_f{frame_index:06d}.jpg"
        path = image_dir / name
        path.write_bytes(encoded)
        saved.append((name, frame_index, frame_index / fps, sharpness, encoded, width, height))

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "frame_index", "timestamp_s", "laplacian_variance", "width", "height"])
        for name, frame_index, timestamp, sharpness, _, width, height in saved:
            writer.writerow([name, frame_index, f"{timestamp:.3f}", f"{sharpness:.3f}", width, height])

    sheet_dir = out_dir / "contact_sheets"
    sheet_dir.mkdir(exist_ok=True)
    for sheet_index, start in enumerate(range(0, len(saved), args.sheet_size), start=1):
        batch = saved[start:start + args.sheet_size]
        thumbs = []
        thumb_h = 0
        for name, frame_index, timestamp, _, encoded, _, _ in batch:
            frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            h, w = frame.shape[:2]
            thumb_w = args.thumb_width
            thumb_h = int(round(h * thumb_w / w))
            thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            cv2.putText(
                thumb, f"{start + len(thumbs) + 1:03d} {timestamp:.1f}s f{frame_index}",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA,
            )
            thumbs.append(thumb)
        rows = math.ceil(len(thumbs) / args.sheet_cols)
        blank = np.zeros_like(thumbs[0])
        thumbs.extend([blank.copy() for _ in range(rows * args.sheet_cols - len(thumbs))])
        sheet = cv2.vconcat([
            cv2.hconcat(thumbs[row * args.sheet_cols:(row + 1) * args.sheet_cols])
            for row in range(rows)
        ])
        sheet_path = sheet_dir / f"{args.prefix}_review_{sheet_index:02d}.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"Video: {video_path}")
    print(f"Frames: {frame_count} at {fps:.3f} fps ({frame_count / fps:.2f} s)")
    print(f"Candidates saved: {len(saved)} to {image_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Contact sheets: {sheet_dir}")
    print("These are unlabeled review candidates; inspect and remove duplicates, blur, and frames without the target before annotation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
