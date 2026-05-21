"""Combine debug frame panels into a single contact-sheet gallery image.

Usage:
    python scripts/make_debug_gallery.py \\
        --debug-dir outputs/sample_shot/debug \\
        --out outputs/sample_shot/debug_gallery.png \\
        --cols 3
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_gallery(debug_dir: str, out_path: str, cols: int = 3) -> None:
    paths = sorted(Path(debug_dir).glob("*_debug.png"))
    if not paths:
        print(f"No debug PNGs found in {debug_dir}")
        return

    panels = [cv2.imread(str(p)) for p in paths]
    panels = [p for p in panels if p is not None]
    if not panels:
        print("Could not read any debug panels.")
        return

    # Uniform size
    h, w = panels[0].shape[:2]
    panels = [cv2.resize(p, (w, h)) for p in panels]

    # Pad to fill the last row
    rows_needed = (len(panels) + cols - 1) // cols
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    while len(panels) < rows_needed * cols:
        panels.append(blank)

    rows = []
    for r in range(rows_needed):
        row_imgs = panels[r * cols : (r + 1) * cols]
        rows.append(np.hstack(row_imgs))
    gallery = np.vstack(rows)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, gallery)
    print(f"Gallery saved to {out_path}  ({len(paths)} panels, {cols} cols)")


def main():
    parser = argparse.ArgumentParser(description="Build a debug frame gallery.")
    parser.add_argument("--debug-dir", required=True, help="Directory containing *_debug.png files.")
    parser.add_argument("--out", required=True, help="Output gallery image path.")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns in the grid.")
    args = parser.parse_args()
    make_gallery(args.debug_dir, args.out, args.cols)


if __name__ == "__main__":
    main()
