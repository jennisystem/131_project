"""Tracking and fit quality metrics, plus optional manual-label evaluation."""

import csv
from pathlib import Path

import numpy as np


def tracking_summary(detections: list[dict]) -> dict:
    """Compute basic tracking statistics."""
    total = len(detections)
    tracked = sum(1 for d in detections if d.get("status") == "tracked")
    missing = total - tracked
    pct = 100.0 * tracked / total if total > 0 else 0.0
    return {
        "total_frames": total,
        "tracked_frames": tracked,
        "missing_frames": missing,
        "tracking_percentage": round(pct, 1),
    }


def fit_summary(fit_result: dict) -> dict:
    """Compute residual statistics from a RANSAC/LS fit result."""
    residuals = fit_result.get("residuals", np.array([]))
    inlier_mask = fit_result.get("inlier_mask", np.array([], dtype=bool))
    outlier_mask = fit_result.get("outlier_mask", np.array([], dtype=bool))

    valid = residuals[np.isfinite(residuals)]
    inlier_residuals = residuals[inlier_mask & np.isfinite(residuals)]

    return {
        "mean_fit_residual_px": round(float(np.mean(inlier_residuals)), 2) if len(inlier_residuals) else None,
        "median_fit_residual_px": round(float(np.median(inlier_residuals)), 2) if len(inlier_residuals) else None,
        "ransac_inliers": int(inlier_mask.sum()),
        "ransac_outliers": int(outlier_mask.sum()),
        "fit_success": bool(fit_result.get("success", False)),
    }


def load_labels(label_csv_path: str) -> list[dict]:
    """Load manual labels from a CSV file."""
    labels = []
    with open(label_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                labels.append(
                    {
                        "video": row.get("video", ""),
                        "frame_idx": int(row["frame_idx"]),
                        "x": float(row["x"]) if row.get("x") else None,
                        "y": float(row["y"]) if row.get("y") else None,
                        "radius": float(row["radius"]) if row.get("radius") else None,
                    }
                )
            except (KeyError, ValueError):
                continue
    return labels


def evaluate_against_labels(
    detections: list[dict], labels: list[dict], success_threshold_px: float = 30.0
) -> dict:
    """Compare predicted centers to manually labeled centers.

    Returns center-error statistics for tracked labeled frames and a success
    rate over all visible-ball labeled frames.
    """
    det_by_frame = {d["frame_idx"]: d for d in detections}
    errors = []
    visible_labels = 0
    successful = 0
    for lbl in labels:
        if lbl.get("x") is None or lbl.get("y") is None:
            continue
        visible_labels += 1
        frame_idx = lbl["frame_idx"]
        det = det_by_frame.get(frame_idx)
        if det is None or det.get("x") is None:
            continue
        err = np.hypot(det["x"] - lbl["x"], det["y"] - lbl["y"])
        errors.append(err)
        if err <= success_threshold_px:
            successful += 1

    return {
        "visible_labeled_frames": visible_labels,
        "labeled_frames_evaluated": len(errors),
        "mean_center_error_px": round(float(np.mean(errors)), 2) if errors else None,
        "median_center_error_px": round(float(np.median(errors)), 2) if errors else None,
        "label_success_threshold_px": success_threshold_px,
        "label_successful_frames": successful,
        "label_success_rate": round(100.0 * successful / visible_labels, 1) if visible_labels else None,
    }
