# Data Directory

## Structure

```
data/
├── raw/          # Place input videos here (not committed to git)
├── frames/       # Extracted frames for inspection and labeling (not committed)
└── labels/       # Manual label CSV files
```

## Adding Videos

Place short side-view phone videos in `data/raw/`. The default config expects:

```
data/raw/sample_shot.mp4
```

## Labels

Use `labels/labels_template.csv` as a starting point for manually labeling ball
centers. Copy it and fill in `x`, `y`, `radius` for each frame you annotate.
