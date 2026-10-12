from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from io_utils import imwrite_unicode
from segmenter import PhantomLesionSegmenter
from visualization import draw_labeled_mask


def imread_unicode(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")
    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test lesion color segmentation on a still image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--out-dir", default="captures", help="Output directory.")
    parser.add_argument("--lesion-hue1", type=int, nargs=2, default=[0, 14], metavar=("LOW", "HIGH"))
    parser.add_argument("--lesion-hue2", type=int, nargs=2, default=[155, 179], metavar=("LOW", "HIGH"))
    parser.add_argument("--lesion-s-min", type=int, default=35)
    parser.add_argument("--lesion-v-min", type=int, default=55)
    parser.add_argument("--lesion-v-max", type=int, default=255)
    parser.add_argument("--lesion-min-area", type=float, default=120.0)
    parser.add_argument("--lesion-max-area", type=float, default=90000.0)
    parser.add_argument("--lesion-min-circularity", type=float, default=0.12)
    parser.add_argument("--lesion-profile", choices=["auto", "official", "pink", "pink-real"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = imread_unicode(args.image)
    segmenter = PhantomLesionSegmenter(profile=args.lesion_profile)
    masks = segmenter.predict(image)
    mask = masks[0] if masks else np.zeros(image.shape[:2], dtype=bool)

    out_dir = Path(__file__).resolve().parent / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    overlay = draw_labeled_mask(image, mask, label="lesion", fill_color=(0, 255, 255), contour_color=(0, 0, 255))
    mask_u8 = mask.astype(np.uint8) * 255
    imwrite_unicode(str(out_dir / f"{stem}_lesion_overlay.png"), overlay)
    imwrite_unicode(str(out_dir / f"{stem}_lesion_mask.png"), mask_u8)
    print(f"masks={len(masks)} selected_pixels={int(np.count_nonzero(mask))}")
    print(f"wrote {out_dir / f'{stem}_lesion_overlay.png'}")
    print(f"wrote {out_dir / f'{stem}_lesion_mask.png'}")


if __name__ == "__main__":
    main()
