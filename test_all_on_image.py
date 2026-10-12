from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from io_utils import imwrite_unicode
from segmenter import PhantomMaterialSegmenter
from run_demo import mask_geometry
from visualization import draw_named_masks, mask_centroid


def imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect all visible phantom classes on a still image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out-dir", default="captures", help="Output directory.")
    parser.add_argument("--material-profile", choices=["legacy", "painted"], default="painted", help="Threshold set for the phantom paint scheme. Painted matches the current brown-red lesion phantom.")
    parser.add_argument(
        "--coordinate-classes",
        default="lesion",
        help="Comma-separated classes written as coordinate targets. Use 'all' to write every detected class.",
    )
    return parser.parse_args()


def coordinate_class_set(value: str, masks: dict[str, np.ndarray]) -> set[str]:
    value = value.strip().lower()
    if value == "all":
        return set(masks.keys())
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    args = parse_args()
    image = imread_unicode(args.image)
    masks = PhantomMaterialSegmenter(profile=args.material_profile).predict_all(image)
    coordinate_classes = coordinate_class_set(args.coordinate_classes, masks)
    geometries = {name: mask_geometry(mask) for name, mask in masks.items()}
    centers = {name: geom["center_px"] for name, geom in geometries.items()}
    overlay = draw_named_masks(image, masks, centers=centers, label_names=coordinate_classes)

    out_dir = Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    overlay_path = out_dir / f"{stem}_all_classes_overlay.png"
    csv_path = out_dir / f"{stem}_all_classes_centers.csv"
    imwrite_unicode(str(overlay_path), overlay)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "pixel_x", "pixel_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "pixels", "confidence", "geometry"])
        for name, mask in masks.items():
            geom = geometries[name]
            centroid = geom["center_px"]
            pixels = int(geom["area_px"])
            bbox = geom["bbox"]
            if name not in coordinate_classes:
                print(f"{name}: mask pixels={pixels} (coordinate not exported)")
                continue
            if centroid is None:
                writer.writerow([name, "", "", bbox[0], bbox[1], bbox[2], bbox[3], pixels, f"{geom['confidence']:.3f}", geom["geometry"]])
                print(f"{name}: not found")
            else:
                writer.writerow([name, centroid[0], centroid[1], bbox[0], bbox[1], bbox[2], bbox[3], pixels, f"{geom['confidence']:.3f}", geom["geometry"]])
                print(f"{name}: center=({centroid[0]},{centroid[1]}) pixels={pixels} confidence={geom['confidence']:.3f}")

    print(f"wrote {overlay_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
