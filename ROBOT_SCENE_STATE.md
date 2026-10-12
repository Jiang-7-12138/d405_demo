# Automated vision output

Run `python run_demo.py`. Detection defaults to `--mask-mode all` and writes
snapshots automatically to `captures/vision_detections.jsonl` at up to 5 Hz. No
key press, mouse selection, or ROI is required for detection/output. Use
`--vision-output-jsonl ""` to disable the JSONL file, `--vision-output-hz` to
change the output rate, and `--obstacle-margin-mm` to inflate organ bounds.

Each line is a standalone JSON snapshot. Length units are meters; pixel values are
image coordinates. `lesion.instances` contains one entry per connected lesion mask:

- `center_px`: mask centroid in pixels.
- `bbox_px`: tight pixel bounding rectangle.
- `contours_px`: pixel coordinates tracing the outer mask boundary.
- `mask_rle`: lossless row-major run-length encoding of the exact mask crop. Counts
  alternate zero/one runs and start with a zero run; `size` is `[height, width]`.
- `center_camera_m`: median 3D point from strict valid depth pixels, or `null` when
  fewer than 10 depth samples are available. `valid_3d` indicates availability.

`obstacles` contains only the non-lesion mask classes. Each organ reports its
pixel center, strict-depth 3D median center, and rough camera-frame AABB based on
5th/95th percentiles expanded by `--obstacle-margin-mm`. If a calibrated
`T_base_camera` is supplied, base-frame bounds are included. Missing masks or
insufficient depth remain explicit `valid: false` entries, not evidence that an
organ is absent. Do not use the identity example transform as a calibration.

The JSONL contains visual detections only: no cut angle, tool pose, cut trajectory,
or robot motion command. Segmentation is still rule-based and needs
validation against the actual live camera view; pixel-exact encoding preserves the
predicted mask exactly, but does not make an inaccurate segmentation accurate.
