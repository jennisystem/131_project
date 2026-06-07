# Quantitative Evaluation Results

The pipeline was regenerated for all ten sample shots. Each clip uses a
per-video YAML file in `configs/` so HSV thresholds, spatial masks, and frame
ranges can be tuned for the visible ball region in that video.

Manual labels are currently available for the first three sample videos. They
were created for representative frames using resized frames from the same
coordinate system as the tracker outputs. The labels include clear frames,
near-rim frames, and difficult occlusion cases. Frames where the ball is not
visible are left blank in the label CSV and are not included in center-error
calculations.

The labeled-frame success rate uses a 30 px center-error threshold.

## Pipeline Summary

| Video | Frames | Tracked | Missing | Tracking % | Mean fit residual (px) | Median fit residual (px) | RANSAC in/out | Fit |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| sample_shot1 | 102 | 69 | 33 | 67.6% | 7.19 | 7.07 | 45/24 | yes |
| sample_shot2 | 99 | 53 | 46 | 53.5% | 7.35 | 7.34 | 26/27 | yes |
| sample_shot3 | 156 | 78 | 78 | 50.0% | 3.56 | 2.05 | 45/33 | yes |
| sample_shot4 | 70 | 27 | 43 | 38.6% | 8.99 | 6.63 | 20/7 | yes |
| sample_shot5 | 27 | 17 | 10 | 63.0% | 9.41 | 8.51 | 17/0 | yes |
| sample_shot6 | 31 | 31 | 0 | 100.0% | 2.29 | 1.15 | 31/0 | yes |
| sample_shot7 | 35 | 35 | 0 | 100.0% | 3.16 | 2.22 | 35/0 | yes |
| sample_shot8 | 61 | 45 | 16 | 73.8% | 6.59 | 4.70 | 42/3 | yes |
| sample_shot9 | 21 | 17 | 4 | 81.0% | 0.28 | 0.22 | 17/0 | yes |
| sample_shot10 | 46 | 44 | 2 | 95.7% | 6.16 | 4.23 | 41/3 | yes |

Several close-up rim clips are intentionally evaluated on shorter clean
segments. After the ball overlaps the rim/net, the HSV mask can merge the ball
with similarly colored or occluding structures. The tuned configs for
`sample_shot4` and `sample_shot6` through `sample_shot10` therefore start at
the first reliable ball contour and stop before prolonged occlusion or exit
frames would cause false tracks.

## Manual Label Evaluation

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
python scripts/run_all_samples.py

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
median center error is below 1 px for the three labeled sets. The larger mean
error on `sample_shot3` is caused by a difficult occlusion/net interaction where
the tracker locks onto a red background/bleacher region instead of the ball.

Across the ten-video set, every tuned run produced a successful RANSAC fit.
The best cases are clips with a clearly separated ball (`sample_shot6`,
`sample_shot7`, `sample_shot9`, and `sample_shot10`). Lower tracking
percentages usually correspond to heavy rim/net occlusion, the ball leaving the
frame, or intentional frame-range limits used to avoid false positives. This
should be discussed as a limitation of per-video HSV tuning and color-based
contour tracking.
