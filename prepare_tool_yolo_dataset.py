from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2

from segmenter import PhantomMaterialSegmenter


CLASS_NAMES = ("liver", "gallbladder", "vessel", "bowel", "lesion")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
MIN_PIXELS = {
    "liver": 1000,
    "gallbladder": 120,
    "vessel": 40,
    "bowel": 500,
    "lesion": 80,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a normal YOLO detection dataset from extracted frames.")
    parser.add_argument("--images", required=True, help="Directory containing extracted JPG/PNG frames")
    parser.add_argument("--out-dir", required=True, help="Output dataset directory")
    parser.add_argument("--profile", default="painted", choices=("legacy", "painted"))
    return parser.parse_args()


def bbox_line(class_name: str, mask, width: int, height: int) -> str | None:
    pixels = int(mask.sum())
    if pixels < MIN_PIXELS[class_name]:
        return None
    points = cv2.findNonZero(mask.astype("uint8"))
    if points is None:
        return None
    x, y, w, h = cv2.boundingRect(points)
    if w < 3 or h < 3:
        return None
    cx = (x + w / 2.0) / width
    cy = (y + h / 2.0) / height
    nw = w / width
    nh = h / height
    return f"{CLASS_IDS[class_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def draw_preview(image, masks: dict[str, object]):
    colors = {
        "liver": (0, 165, 255),
        "gallbladder": (0, 220, 0),
        "vessel": (255, 0, 0),
        "bowel": (255, 130, 80),
        "lesion": (0, 0, 255),
    }
    out = image.copy()
    for name in CLASS_NAMES:
        mask = masks.get(name)
        if mask is None:
            continue
        line = bbox_line(name, mask, image.shape[1], image.shape[0])
        if line is None:
            continue
        points = cv2.findNonZero(mask.astype("uint8"))
        x, y, w, h = cv2.boundingRect(points)
        color = colors[name]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, name, (x, max(18, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def main() -> int:
    args = parse_args()
    image_dir = Path(args.images)
    out_dir = Path(args.out_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    overlays_out = out_dir / "overlays"
    for directory in (images_out, labels_out, overlays_out):
        directory.mkdir(parents=True, exist_ok=True)

    segmenter = PhantomMaterialSegmenter(profile=args.profile)
    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg")) + sorted(image_dir.glob("*.png"))
    manifest_rows = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        masks = segmenter.predict_all(image)
        lines = []
        emitted = []
        for name in CLASS_NAMES:
            line = bbox_line(name, masks.get(name), image.shape[1], image.shape[0])
            if line is not None:
                lines.append(line)
                emitted.append(name)
        shutil.copy2(image_path, images_out / image_path.name)
        (labels_out / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        cv2.imwrite(str(overlays_out / image_path.name), draw_preview(image, masks))
        manifest_rows.append((image_path.name, ",".join(emitted), len(lines)))

    (out_dir / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    (out_dir / "README.txt").write_text(
        "These are AUTO-PREANNOTATED normal YOLO detection boxes, not verified ground truth.\n"
        "Edit every image before training. Format: class center_x center_y width height.\n"
        "Classes: 0 liver, 1 gallbladder, 2 vessel, 3 bowel, 4 lesion.\n"
        "Empty label files are intentional for frames with no accepted auto-detection.\n",
        encoding="utf-8",
    )
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(("image", "auto_classes", "box_count"))
        writer.writerows(manifest_rows)
    print(f"Images: {len(manifest_rows)}")
    print(f"Dataset: {out_dir}")
    print(f"Classes: {', '.join(f'{i}={n}' for i, n in enumerate(CLASS_NAMES))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
