# Quantitative Evaluation Results

Manual labels were created for representative frames in each sample video using
resized frames from the same coordinate system as the tracker outputs. The
labels include clear frames, near-rim frames, and difficult occlusion cases.
Frames where the ball is not visible are left blank in the label CSV and are
not included in center-error calculations.

The labeled-frame success rate uses a 30 px center-error threshold.

| Video | Pipeline tracking rate | Visible labeled frames | Mean center error (px) | Median center error (px) | Labeled-frame success rate |
|---|---:|---:|---:|---:|---:|
| sample_shot1 | 67.6% | 7 | 0.34 | 0.33 | 100.0% |
| sample_shot2 | 53.5% | 6 | 6.42 | 0.47 | 83.3% |
| sample_shot3 | 50.0% | 5 | 234.96 | 0.25 | 80.0% |

## Label Files

- `data/labels/sample_shot1_labels.csv`
- `data/labels/sample_shot2_labels.csv`
- `data/labels/sample_shot3_labels.csv`

## Evaluation Commands

```bash
python scripts/evaluate_labels.py \
  --detections outputs/sample_shot/detections.csv \
  --labels data/labels/sample_shot1_labels.csv

python scripts/evaluate_labels.py \
  --detections outputs/sample_shot2/detections.csv \
  --labels data/labels/sample_shot2_labels.csv

python scripts/evaluate_labels.py \
  --detections outputs/sample_shot3/detections.csv \
  --labels data/labels/sample_shot3_labels.csv
```

## Interpretation

The tracker is accurate when HSV segmentation finds the correct ball contour:
median center error is below 1 px for all three labeled sets. The larger mean
error on `sample_shot3` is caused by a difficult occlusion/net interaction where
the tracker locks onto a red background/bleacher region instead of the ball.
This should be discussed as a limitation of per-video HSV tuning and
color-based contour tracking.
