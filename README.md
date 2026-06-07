# Basketball Shot Arc Tracker

## Project Overview

This project loads short side-view phone videos, extracts frames, detects the
basketball through HSV color thresholding and contour analysis, links detections
across frames with a simple motion-continuity tracker, and fits a parabolic arc
to the shot trajectory using least squares and RANSAC.

---

## CS131 Concepts Used

| Concept | Where |
|---------|-------|
| Image / frame representation | `video_io.py` — BGR arrays, resize |
| Filtering and smoothing | `preprocess.py` — median / Gaussian blur |
| HSV threshold segmentation | `detect.py` — `cv2.inRange` |
| Morphological operations | `detect.py` — open + close to clean mask |
| Contour extraction | `detect.py` — `cv2.findContours`, circularity |
| Canny / Hough circles | `detect.py` — optional stretch goal |
| Temporal tracking | `track.py` — constant-velocity prediction |
| Least-squares fitting | `fit.py` — parabola via `np.linalg.lstsq` |
| RANSAC | `fit.py` — robust parabola fitting |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Data Setup

Place short side-view phone videos in `data/raw/`. This project currently uses
ten sample clips:

```
data/raw/sample_shot.mp4
data/raw/sample_shot2.mp4
...
data/raw/sample_shot10.mp4
```

---

## Run the Full Pipeline

```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

To regenerate all tuned sample results:

```bash
python scripts/run_all_samples.py
```

The per-video configs are stored in `configs/sample_shot1.yaml` through
`configs/sample_shot10.yaml`.

---

## Extract Frames for Inspection

```bash
python scripts/extract_frames.py \
    --video data/raw/sample_shot.mp4 \
    --out data/frames/sample_shot \
    --stride 10
```

---

## Build Debug Gallery

```bash
python scripts/make_debug_gallery.py \
    --debug-dir outputs/sample_shot/debug \
    --out outputs/sample_shot/debug_gallery.png
```

---

## Evaluate Against Manual Labels

```bash
python scripts/evaluate_labels.py \
    --detections outputs/sample_shot/detections.csv \
    --labels data/labels/sample_labels.csv
```

---

## Outputs

After running the pipeline:

```
outputs/sample_shot/
├── annotated_video.mp4       # Ball center + trajectory trail overlay
├── detections.csv            # Per-frame x, y, radius, score, status
├── trajectory_plot.png       # Raw centers + RANSAC inliers/outliers + fitted arc
├── debug/
│   ├── frame_0000_debug.png  # 2×2 panel: original | mask | candidates | final
│   ├── frame_0010_debug.png
│   └── ...
├── debug_gallery.png         # Contact sheet of all debug panels
└── summary.json              # Quantitative metrics
```

---

## Tuning HSV Thresholds

Edit `configs/default.yaml`:

```yaml
hsv_threshold:
  lower: [5, 80, 80]   # Hue, Sat, Val lower bound
  upper: [25, 255, 255] # Hue, Sat, Val upper bound
```

Then re-run and inspect `outputs/sample_shot/debug/` to see how the mask changes.

---

## Milestone Status

### What currently works
- Frame loading, resizing, and preprocessing
- HSV mask + morphological cleaning
- Contour-based candidate extraction (area + circularity filter)
- Simple constant-velocity temporal tracker
- Least-squares + RANSAC parabola fitting
- Debug panel generation and gallery
- Annotated video output
- Trajectory plot (inverted y-axis, inlier/outlier colour coding)
- Quantitative summary JSON
- Per-video tuned configs for ten sample shots

### Current failure cases
- Motion blur during fast release causes spread or missing detections
- Ball near player's hands (skin tone / jersey) may fail HSV filter
- Orange-coloured backgrounds or rims can cause false positives
- Large jumps (fast pan or fast ball) can exceed `max_jump_px` and break track
- Heavy rim/net occlusion can merge the ball with similarly colored structures

### Next steps
- Add radius consistency check in tracker
- Manually label the seven additional sample shots for center-error evaluation
- Explore optional Canny/Hough circle comparison
