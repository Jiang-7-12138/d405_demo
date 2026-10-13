# YOLO-seg Dataset Preparation

## Candidate extraction

Extract temporally distributed sharp candidates from a video:

```powershell
python prepare_yolo_video_frames.py `
  --video "D:\path\to\video.mp4" `
  --out-dir "datasets\lesion_yoloseg\video1_candidates" `
  --count 180 `
  --prefix video1
```

The script writes full-resolution JPEGs to `images/`, timestamps and sharpness
to `manifest.csv`, and contact sheets to `contact_sheets/`. It chooses the
sharpest frame within each temporal bin. This is candidate sampling, not
deduplication: inspect the sheets and discard near-duplicates, blur, and frames
that do not resemble the target competition scene.

Run it separately for the real-scene video with a different output directory
and prefix. Keep the source-video identity for every image.

## Label definition

Start with one class, `lesion`, unless the team has enough time and examples to
label other semantic classes consistently. A positive image gets a polygon
around the exact visible lesion boundary. A negative image gets an empty label
file only when the lesion is genuinely absent or not visible. Do not label
normal liver, shadows, tools, or unrelated red objects as lesions. Partially
occluded lesions should be labeled only when the boundary is still reasonably
identifiable, with the same policy used across the dataset.

YOLO-seg rows use:

```text
class_id x1 y1 x2 y2 ... xN yN
```

Coordinates are normalized to `[0, 1]`; polygons need at least three points.
Each image has a same-basename `.txt` label file. Review every automatic mask
before treating it as ground truth. Threshold-generated masks can save clicks,
but are not reliable labels by themselves.

## Split policy

Do not randomly split adjacent frames from one video between train and
validation; nearly identical frames can make validation misleading. Keep
validation images grouped by recording session or scenario. Use the training
video for training and reserve a separate real-scene recording (or a held-out
session from it) for validation. The eventual split must still cover the
lighting, distance, angle, occlusion, and target variants expected at the event.

The first supplied clip is about 21.9 seconds. Its first portion repeatedly
shows a largely static board of red-brown samples, while its later portion
shows the phantom from a small number of viewpoints. The 180 extracted files
are review candidates, not 180 independent appearances. Expect to keep a much
smaller diverse subset from that clip and collect varied frames in the real
competition setup.
