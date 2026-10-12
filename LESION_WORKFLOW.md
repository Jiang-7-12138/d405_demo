# Lesion detection workflow

This project now has a deterministic lesion mode for the fixed phantom:

```powershell
cd D:\HuaweiMoveData\Users\蒋高韵\Desktop\D405_vision_demo
python run_demo.py --mask-mode lesion --min-depth-mm 30 --max-depth-mm 450
```

Lesion profiles:

- `--lesion-profile official`: red-brown raised lesion on the official-style liver phantom photo.
- `--lesion-profile pink`: bright pink silicone practice module or pasted pink lesion.
- `--lesion-profile auto`: try `official` first, then fall back to `pink`.

Recommended commands before the real phantom arrives:

```powershell
python run_demo.py --mask-mode lesion --lesion-profile official
python run_demo.py --mask-mode lesion --lesion-profile pink
python run_demo.py --mask-mode lesion --lesion-profile auto
```

Still-image tests:

```powershell
python test_lesion_on_image.py --lesion-profile official --image path\to\official_like_phantom.jpg
python test_lesion_on_image.py --lesion-profile pink --image path\to\pink_silicone_module.jpg
```

Keys:

- `l`: toggle lesion color segmentation.
- `h`: toggle generic high-saturation HSV segmentation.
- `m`: use the manually dragged ROI as the mask.
- `n`: disable segmentation.
- Drag mouse: set an ROI. In lesion mode, the ROI restricts the search area.
- `r`: append the current mask 3D result to `captures/measurements.csv`.
- `s`: save RGB, depth visualization, raw `.npz`, and point cloud `.ply`.

Useful threshold knobs:

```powershell
python run_demo.py --mask-mode lesion `
  --lesion-hue1 0 14 `
  --lesion-hue2 155 179 `
  --lesion-s-min 35 `
  --lesion-v-min 55 `
  --lesion-min-area 120 `
  --lesion-min-circularity 0.12
```

Use lower `--lesion-s-min` if the lesion looks pale under strong light. Narrow
the hue ranges if the liver or vessel markings are also selected. Raise
`--lesion-min-area` when small red/blue printed lines are being selected.

Current recommendation:

For a fixed phantom with stable lesion material and color, start with image
processing plus depth filtering. It is easier to debug and can run in real time
on CPU. Add SAM or YOLO only after collecting real frames that show HSV fails
under lighting, specular highlights, blood-like stains, instrument occlusion, or
large camera auto-exposure changes.

SAM notes:

SAM is promptable instance segmentation, not semantic detection. It does not
know "lesion" by default. You must give prompts such as a point, box, or masks
from another detector. A practical SAM pipeline here would be:

1. Use HSV or YOLO to propose a lesion point/box.
2. Run SAM with that point/box to refine the boundary.
3. Pick the best mask by area, shape, color consistency, and depth validity.
4. Feed the bool mask into the existing `robust_mask_stats` path for 3D center.

YOLO/YOLO-seg notes:

YOLO needs annotated data and training. YOLO gives boxes; YOLO-seg gives masks.
If you train later, YOLO-seg is more directly useful than YOLO because the 3D
measurement needs a mask, not just a rectangle.
