from __future__ import annotations

import cv2
import numpy as np


def imwrite_unicode(path: str, image: np.ndarray) -> None:
    ext = ".png"
    dot = path.rfind(".")
    if dot >= 0:
        ext = path[dot:]
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode image as {ext}: {path}")
    encoded.tofile(path)
