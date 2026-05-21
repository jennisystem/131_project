"""Extract and save individual frames from a video for inspection and labeling.

Usage:
    python scripts/extract_frames.py \\
        --video data/raw/sample_shot.mp4 \\
        --out data/frames/sample_shot \\
        --stride 10
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--out", required=True, help="Output directory for frames.")
    parser.add_argument("--stride", type=int, default=1, help="Save every Nth frame.")
    parser.add_argument("--width", type=int, default=None, help="Resize width (optional).")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0

    for frame_idx in tqdm(range(total), unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.stride != 0:
            continue
        if args.width is not None:
            h, w = frame.shape[:2]
            new_h = int(h * args.width / w)
            frame = cv2.resize(frame, (args.width, new_h))
        path = out_dir / f"frame_{frame_idx:04d}.jpg"
        cv2.imwrite(str(path), frame)
        saved += 1

    cap.release()
    print(f"Saved {saved} frames to {out_dir}/")


if __name__ == "__main__":
    main()
