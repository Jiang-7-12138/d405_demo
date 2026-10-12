from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from io_utils import imwrite_unicode
from segmenter import PhantomMaterialSegmenter


COLORS = {
    "liver": (0, 180, 255),
    "bowel": (255, 160, 80),
    "omentum": (0, 220, 255),
    "gallbladder": (0, 220, 0),
    "vessel": (255, 0, 0),
    "lesion": (0, 255, 255),
}


def imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test rule-based phantom material segmentation on a still image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out-dir", default="captures", help="Output directory.")
    parser.add_argument("--material-profile", choices=["legacy", "painted"], default="legacy", help="Threshold set for the phantom paint scheme.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = imread_unicode(args.image)
    masks = PhantomMaterialSegmenter(profile=args.material_profile).predict_all(image)
    overlay = image.copy()

    for name, mask in masks.items():
        color = COLORS.get(name, (255, 255, 255))
        layer = np.zeros_like(image)
        layer[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, layer, 0.35, 0.0)
        print(f"{name}: pixels={int(np.count_nonzero(mask))}")

    out_dir = Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    imwrite_unicode(str(out_dir / f"{stem}_materials_overlay.png"), overlay)
    for name, mask in masks.items():
        imwrite_unicode(str(out_dir / f"{stem}_{name}_mask.png"), mask.astype(np.uint8) * 255)
    print(f"wrote {out_dir / f'{stem}_materials_overlay.png'}")


if __name__ == "__main__":
    main()
