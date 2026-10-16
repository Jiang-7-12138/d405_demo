"""Sample endoscope clips and write a CVAT YOLO 1.1 detection archive.

Labels are automatic boxes from PhantomMaterialSegmenter. They are a starting
point for editing in CVAT, not ground truth.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import cv2
import numpy as np

from io_utils import imwrite_unicode
from segmenter import PhantomMaterialSegmenter


CLASS_NAMES = ("liver", "omentum", "gallbladder", "vessel", "bowel", "lesion")
MIN_AREA = {
    "liver": 1500,
    "omentum": 400,
    "gallbladder": 200,
    "vessel": 80,
    "bowel": 800,
    "lesion": 200,
}
MAX_BOXES = {
    "liver": 1,
    "omentum": 1,
    "gallbladder": 1,
    "vessel": 2,
    "bowel": 2,
    "lesion": 1,
}
COLORS = {
    "liver": (0, 140, 255),
    "omentum": (0, 255, 255),
    "gallbladder": (0, 200, 0),
    "vessel": (255, 80, 0),
    "bowel": (255, 160, 80),
    "lesion": (0, 0, 255),
}


def boxes_from_mask(mask: np.ndarray, class_name: str, width: int, height: int) -> list[tuple[int, int, int, int]]:
    if mask is None or int(np.count_nonzero(mask)) < MIN_AREA[class_name]:
        return []
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    found: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < MIN_AREA[class_name]:
            continue
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        if w < 4 or h < 4:
            continue
        if w > width * 0.98 and h > height * 0.98:
            continue
        # A pale patch on the upper liver is not bowel. Real bowel stays low in these views.
        if class_name == "bowel" and (y + h / 2.0) < 0.25 * height:
            continue
        found.append((area, x, y, w, h))
    found.sort(reverse=True)
    return [(x, y, w, h) for _, x, y, w, h in found[: MAX_BOXES[class_name]]]


def yolo_line(class_id: int, box: tuple[int, int, int, int], width: int, height: int) -> str:
    x, y, w, h = box
    cx = min(1.0, max(0.0, (x + w / 2.0) / width))
    cy = min(1.0, max(0.0, (y + h / 2.0) / height))
    nw = min(1.0, max(0.0, w / width))
    nh = min(1.0, max(0.0, h / height))
    return f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def sharpest_frames(video_path: Path, count: int) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    selected: list[tuple[int, float, np.ndarray] | None] = [None] * count
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        bin_index = min(count - 1, frame_index * count // max(frame_count, 1))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        current = selected[bin_index]
        if current is None or sharpness > current[1]:
            selected[bin_index] = (frame_index, sharpness, frame.copy())
        frame_index += 1
    capture.release()
    return [(item[0], item[2]) for item in selected if item is not None]


def detect(frame: np.ndarray, dark: PhantomMaterialSegmenter) -> dict[str, np.ndarray]:
    masks = dark.predict_all(frame)
    lesion = masks.get("lesion")
    if lesion is None or int(np.count_nonzero(lesion)) < MIN_AREA["lesion"]:
        liver = masks.get("liver")
        if liver is not None:
            liver_for_lesion = liver if lesion is None else (liver | lesion)
            masks["lesion"] = PhantomMaterialSegmenter._endoscope_lesion_circle_deep(frame, liver_for_lesion)
    return masks


def draw_boxes(frame: np.ndarray, labeled: list[tuple[str, tuple[int, int, int, int]]]) -> np.ndarray:
    out = frame.copy()
    for name, (x, y, w, h) in labeled:
        color = COLORS[name]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, name, (x, max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, name, (x, max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main() -> int:
    root = Path(__file__).resolve().parent
    captures = root / "captures"
    videos = [
        captures / "endoscope_20261016_180145.avi",
        captures / "endoscope_20261016_180212.avi",
        captures / "endoscope_20261016_180235.avi",
        captures / "endoscope_20261016_180255.avi",
        captures / "endoscope_20261016_180319.avi",
        captures / "endoscope_20261016_180424.avi",
        captures / "endoscope_20261016_180454.avi",
        captures / "endoscope_20261016_180518.avi",
        captures / "endoscope_20261016_180544.avi",
    ]
    out_dir = root / "datasets" / "organ_yolo" / "cvat_yolo_20261016"
    image_dir = out_dir / "obj_train_data"
    preview_dir = root / "datasets" / "organ_yolo" / "preview_20261016"
    image_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    dark = PhantomMaterialSegmenter(profile="endoscope-dark")
    names_text = "\n".join(CLASS_NAMES) + "\n"
    (out_dir / "obj.names").write_text(names_text, encoding="utf-8")
    (out_dir / "obj.data").write_text(
        f"classes = {len(CLASS_NAMES)}\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n",
        encoding="utf-8",
    )
    (out_dir / "data.yaml").write_text(
        "path: .\ntrain: obj_train_data\nval: obj_train_data\nnames:\n"
        + "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
        + "\n",
        encoding="utf-8",
    )

    train_lines: list[str] = []
    manifest_rows: list[tuple] = []
    class_hits = {name: 0 for name in CLASS_NAMES}
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.release()
        duration = frame_count / fps if fps else 0.0
        count = int(np.clip(round(duration / 0.30), 10, 28))
        count = min(count, frame_count)
        saved = 0
        for order, (frame_index, frame) in enumerate(sharpest_frames(video, count), start=1):
            height, width = frame.shape[:2]
            masks = detect(frame, dark)
            lines: list[str] = []
            labeled: list[tuple[str, tuple[int, int, int, int]]] = []
            present: list[str] = []
            for class_id, name in enumerate(CLASS_NAMES):
                boxes = boxes_from_mask(masks.get(name), name, width, height)
                if boxes:
                    present.append(name)
                    class_hits[name] += 1
                for box in boxes:
                    lines.append(yolo_line(class_id, box, width, height))
                    labeled.append((name, box))
            stem = f"{video.stem}_{order:02d}_f{frame_index:06d}"
            imwrite_unicode(str(image_dir / f"{stem}.jpg"), frame)
            (image_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            if order == 1 or order == count // 2 or order == count:
                imwrite_unicode(str(preview_dir / f"{stem}.jpg"), draw_boxes(frame, labeled))
            train_lines.append(f"obj_train_data/{stem}.jpg")
            manifest_rows.append((stem + ".jpg", video.name, frame_index, f"{frame_index / fps:.3f}", ",".join(present), len(lines)))
            saved += 1
        print(f"{video.name}: {saved} frames from {frame_count} ({duration:.1f}s)")

    (out_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "video", "frame_index", "timestamp_s", "classes", "box_count"))
        writer.writerows(manifest_rows)

    zip_path = root / "datasets" / "organ_yolo" / "cvat_yolo_20261016.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ("obj.names", "obj.data", "train.txt"):
            archive.write(out_dir / name, name)
        for path in sorted(image_dir.iterdir()):
            if path.suffix.lower() in {".jpg", ".txt"}:
                archive.write(path, f"obj_train_data/{path.name}")
    print(f"images={len(manifest_rows)}")
    print("hits " + " ".join(f"{name}={class_hits[name]}" for name in CLASS_NAMES))
    print(f"zip={zip_path} bytes={zip_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
