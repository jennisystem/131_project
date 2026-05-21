"""Video loading, frame saving, and CSV output utilities."""

import csv
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def load_video_frames(
    video_path: str,
    resize_width: Optional[int] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    frame_stride: int = 1,
) -> tuple[list[np.ndarray], dict]:
    """Load frames from a video file.

    Returns a list of BGR frames and a metadata dict with keys:
        fps, orig_size, resized_size, total_frames, frame_indices
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    actual_end = total if end_frame is None else min(end_frame, total)
    frame_indices = list(range(start_frame, actual_end, frame_stride))

    # Compute output size
    if resize_width is not None and resize_width != orig_w:
        scale = resize_width / orig_w
        out_w = resize_width
        out_h = int(orig_h * scale)
    else:
        out_w, out_h = orig_w, orig_h

    frames: list[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current = start_frame
    for idx in frame_indices:
        if idx != current:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            current = idx
        ret, frame = cap.read()
        if not ret:
            break
        if (out_w, out_h) != (orig_w, orig_h):
            frame = cv2.resize(frame, (out_w, out_h))
        frames.append(frame)
        current += 1

    cap.release()

    metadata = {
        "fps": fps,
        "orig_size": (orig_w, orig_h),
        "resized_size": (out_w, out_h),
        "total_frames": len(frames),
        "frame_indices": frame_indices[: len(frames)],
    }
    return frames, metadata


def save_video(frames: list[np.ndarray], output_path: str, fps: float) -> None:
    """Write a list of BGR frames to an .mp4 file."""
    if not frames:
        return
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()


def save_detections_csv(detections: list[dict], output_path: str) -> None:
    """Save per-frame detection results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["frame_idx", "x", "y", "radius", "score", "status"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for det in detections:
            writer.writerow({k: det.get(k, "") for k in fieldnames})
