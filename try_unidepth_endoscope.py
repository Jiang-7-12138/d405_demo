"""Run UniDepthV2 ViT-S on one endoscope frame and save a depth preview.

This machine has no NVIDIA GPU, so inference uses CPU. Weights download from
Hugging Face on first run: lpiccinelli/unidepth-v2-vits14.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import types

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party" / "UniDepth"))
# UniDepth imports wandb while loading the package. Inference does not log there.
sys.modules.setdefault("wandb", types.ModuleType("wandb"))

from unidepth.models import UniDepthV2  # noqa: E402
from unidepth.utils.camera import Pinhole  # noqa: E402

from io_utils import imwrite_unicode  # noqa: E402


# Endoscope calibration used by the existing 800x600 depth pipeline.
ENDOSCOPE_K = np.array(
    [[675.99, 0.0, 422.69], [0.0, 677.0, 279.07], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
# Median real/predicted ratio on the D405 frame recon_20261016_230825.
DEPTH_SCALE = 0.170


def load_frame(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from {video_path}")
    return frame


def colorize(depth_m: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    values = depth_m[valid]
    low, high = np.percentile(values, [2, 98])
    normalized = np.clip((depth_m - low) / max(high - low, 1e-6), 0, 1)
    color = cv2.applyColorMap((255 * (1 - normalized)).astype(np.uint8), cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color, float(low), float(high)


def main() -> int:
    video = ROOT / "captures" / "endoscope_20261016_221730.avi"
    frame = load_frame(video)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    print(f"frame {width}x{height} from {video.name}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}", flush=True)
    started = time.perf_counter()
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vits14")
    model.resolution_level = 4
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()
    print(f"model ready in {time.perf_counter() - started:.1f}s", flush=True)

    image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
    camera = None
    if (width, height) == (800, 600):
        camera = Pinhole(K=torch.from_numpy(ENDOSCOPE_K).unsqueeze(0))
        print("using 800x600 endoscope intrinsics", flush=True)
    else:
        print("resolution is not 800x600, letting UniDepth estimate intrinsics", flush=True)

    started = time.perf_counter()
    with torch.inference_mode():
        prediction = model.infer(image, camera)
    elapsed = time.perf_counter() - started
    raw_depth = prediction["depth"].detach().float().cpu().numpy().squeeze()
    depth = raw_depth * DEPTH_SCALE
    valid = depth[np.isfinite(depth) & (depth > 0)]
    print(
        f"inference {elapsed:.1f}s  scale {DEPTH_SCALE:.3f}  "
        f"scaled p02/median/p98 {np.percentile(valid, 2)*100:.1f} / {np.median(valid)*100:.1f} / {np.percentile(valid, 98)*100:.1f} cm",
        flush=True,
    )
    lesion_x, lesion_y = 252, 240
    yy, xx = np.ogrid[:height, :width]
    disk = (xx - lesion_x) ** 2 + (yy - lesion_y) ** 2 <= 15 ** 2
    lesion_values = depth[disk]
    lesion_values = lesion_values[np.isfinite(lesion_values) & (lesion_values > 0)]
    print(
        f"lesion neighborhood ({lesion_x},{lesion_y}) r=15 px  "
        f"p10/median/p90 {np.percentile(lesion_values, 10)*100:.1f} / {np.median(lesion_values)*100:.1f} / {np.percentile(lesion_values, 90)*100:.1f} cm",
        flush=True,
    )

    colored, low, high = colorize(depth)
    preview = np.hstack((frame, colored))
    cv2.circle(preview, (lesion_x, lesion_y), 15, (0, 0, 255), 2)
    cv2.circle(preview, (lesion_x + width, lesion_y), 15, (0, 0, 255), 2)
    cv2.putText(preview, "RGB", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        preview,
        f"scaled x{DEPTH_SCALE:.3f}  {low*100:.1f}-{high*100:.1f} cm",
        (width + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    out_dir = ROOT / "captures"
    preview_path = out_dir / "unidepth_v2_vits14_scaled_preview.jpg"
    depth_path = out_dir / "unidepth_v2_vits14_scaled_depth_m.npy"
    imwrite_unicode(str(preview_path), preview)
    np.save(depth_path, depth.astype(np.float32))
    print(f"saved {preview_path}", flush=True)
    print(f"saved {depth_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
