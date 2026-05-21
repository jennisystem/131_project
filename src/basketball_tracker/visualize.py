"""Visualization helpers: frame overlays, debug panels, and trajectory plots."""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------------------------------- #
# Frame-level drawing
# --------------------------------------------------------------------------- #

def draw_candidates(frame: np.ndarray, candidates: list[dict]) -> np.ndarray:
    """Draw all HSV candidates as thin yellow circles."""
    out = frame.copy()
    for c in candidates:
        cx, cy, r = int(c["x"]), int(c["y"]), max(int(c["radius"]), 3)
        cv2.circle(out, (cx, cy), r, (0, 255, 255), 1)
        cv2.putText(
            out,
            f"{c['score']:.2f}",
            (cx + r + 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
        )
    return out


def draw_selected_detection(frame: np.ndarray, detection: dict) -> np.ndarray:
    """Draw the chosen detection as a filled green circle + crosshair."""
    out = frame.copy()
    if detection.get("x") is None:
        return out
    cx, cy = int(detection["x"]), int(detection["y"])
    r = max(int(detection.get("radius") or 12), 5)
    cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
    cv2.drawMarker(out, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
    return out


def draw_trajectory_trail(
    frame: np.ndarray,
    detections_so_far: list[dict],
    trail_length: int = 60,
) -> np.ndarray:
    """Draw a fading dot trail of recent tracked positions."""
    out = frame.copy()
    recent = [d for d in detections_so_far[-trail_length:] if d.get("x") is not None]
    n = len(recent)
    for i, det in enumerate(recent):
        alpha = (i + 1) / max(n, 1)
        radius = max(int(3 * alpha), 1)
        color = (0, int(200 * alpha), int(255 * alpha))
        cv2.circle(out, (int(det["x"]), int(det["y"])), radius, color, -1)
    return out


def make_debug_panel(
    original: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
    candidates_overlay: np.ndarray,
    final_overlay: np.ndarray,
) -> np.ndarray:
    """Assemble a 2×2 grid debug panel (original | mask | candidates | final)."""
    h, w = original.shape[:2]

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    top = np.hstack([original, mask_bgr])
    bottom = np.hstack([candidates_overlay, final_overlay])
    panel = np.vstack([top, bottom])

    labels = ["Original", "HSV Mask", "Candidates", "Final Detection"]
    positions = [(5, 20), (w + 5, 20), (5, h + 20), (w + 5, h + 20)]
    for label, pos in zip(labels, positions):
        cv2.putText(panel, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return panel


def save_debug_frame(
    frame_idx: int,
    original: np.ndarray,
    processed: np.ndarray,
    mask: np.ndarray,
    candidates: list[dict],
    detection: dict,
    detections_so_far: list[dict],
    output_dir: str,
    trail_length: int = 60,
) -> None:
    """Build and save a debug panel for one frame."""
    cands_overlay = draw_candidates(original.copy(), candidates)
    final = draw_selected_detection(original.copy(), detection)
    final = draw_trajectory_trail(final, detections_so_far, trail_length)

    # Status text
    status = detection.get("status", "?")
    color = (0, 255, 0) if status == "tracked" else (0, 0, 255)
    cv2.putText(final, f"#{frame_idx} {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    panel = make_debug_panel(original, processed, mask, cands_overlay, final)

    out_path = Path(output_dir) / f"frame_{frame_idx:04d}_debug.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


# --------------------------------------------------------------------------- #
# Annotated video frame
# --------------------------------------------------------------------------- #

def annotate_frame(
    frame: np.ndarray,
    frame_idx: int,
    detection: dict,
    detections_so_far: list[dict],
    trail_length: int = 60,
) -> np.ndarray:
    """Return a single annotated frame for the output video."""
    out = draw_trajectory_trail(frame.copy(), detections_so_far, trail_length)
    out = draw_selected_detection(out, detection)
    status = detection.get("status", "?")
    color = (0, 255, 0) if status == "tracked" else (0, 0, 200)
    cv2.putText(out, f"Frame {frame_idx}  [{status}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


# --------------------------------------------------------------------------- #
# Trajectory plot
# --------------------------------------------------------------------------- #

def plot_trajectory(
    detections: list[dict],
    fit_result: dict,
    output_path: str,
) -> None:
    """Plot raw detections, RANSAC inliers/outliers, and fitted parabola."""
    from .fit import get_valid_points

    ts, xs, ys = get_valid_points(detections)

    fig, ax = plt.subplots(figsize=(10, 6))

    inlier_mask = fit_result.get("inlier_mask")
    outlier_mask = fit_result.get("outlier_mask")
    coeffs = fit_result.get("coeffs")

    if inlier_mask is not None and inlier_mask.any():
        ax.scatter(xs[inlier_mask], ys[inlier_mask], c="green", s=25, label="RANSAC inliers", zorder=3)
    if outlier_mask is not None and outlier_mask.any():
        ax.scatter(xs[outlier_mask], ys[outlier_mask], c="red", s=15, marker="x", label="RANSAC outliers", zorder=3)
    if inlier_mask is None:
        ax.scatter(xs, ys, c="blue", s=20, label="Raw detections", zorder=3)

    if coeffs is not None and len(xs) > 0:
        x_range = np.linspace(xs.min(), xs.max(), 300)
        a, b, c = coeffs
        y_range = a * x_range ** 2 + b * x_range + c
        ax.plot(x_range, y_range, "orange", linewidth=2, label="Fitted parabola", zorder=2)

    ax.invert_yaxis()
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels, inverted)")
    ax.set_title("Basketball Trajectory — 2D Projection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
