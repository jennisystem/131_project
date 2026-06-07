# Data Directory

## Structure

```
data/
├── raw/          # Place input videos here (not committed to git)
├── frames/       # Extracted frames for inspection and labeling (not committed)
└── labels/       # Manual label CSV files
```

## Adding Videos

Place short side-view phone videos in `data/raw/`. The tuned sample set expects:

```
data/raw/sample_shot.mp4
data/raw/sample_shot2.mp4
data/raw/sample_shot3.mp4
data/raw/sample_shot4.mp4
data/raw/sample_shot5.mp4
data/raw/sample_shot6.mp4
data/raw/sample_shot7.mp4
data/raw/sample_shot8.mp4
data/raw/sample_shot9.mp4
data/raw/sample_shot10.mp4
```

Each sample has a matching config in `configs/`. Run all tuned samples with:

```bash
python scripts/run_all_samples.py
```

## Labels

Use `labels/labels_template.csv` as a starting point for manually labeling ball
centers. Copy it and fill in `x`, `y`, `radius` for each frame you annotate.
Manual labels currently exist for `sample_shot1`, `sample_shot2`, and
`sample_shot3`; the seven added clips have pipeline summaries but not manual
center-error labels yet.
