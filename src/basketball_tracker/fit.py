"""Parabolic trajectory fitting with least squares and RANSAC."""

import random
from typing import Optional

import numpy as np


def get_valid_points(detections: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (frame_indices, xs, ys) arrays for all tracked detections."""
    ts, xs, ys = [], [], []
    for d in detections:
        if d.get("x") is not None:
            ts.append(d["frame_idx"])
            xs.append(d["x"])
            ys.append(d["y"])
    return np.array(ts, float), np.array(xs, float), np.array(ys, float)


def fit_parabola_xy(xs: np.ndarray, ys: np.ndarray) -> Optional[np.ndarray]:
    """Fit y = ax^2 + bx + c using least squares.

    Returns coefficients [a, b, c] or None if underdetermined.
    """
    if len(xs) < 3:
        return None
    A = np.column_stack([xs ** 2, xs, np.ones_like(xs)])
    coeffs, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    return coeffs


def fit_time_model(
    ts: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> Optional[dict]:
    """Fit x(t) and y(t) as polynomials in frame index t.

    x(t) = linear,  y(t) = quadratic (parabolic flight).
    Returns dict with 'x_coeffs' and 'y_coeffs'.
    """
    if len(ts) < 3:
        return None
    Ax = np.column_stack([ts, np.ones_like(ts)])
    xc, _, _, _ = np.linalg.lstsq(Ax, xs, rcond=None)
    Ay = np.column_stack([ts ** 2, ts, np.ones_like(ts)])
    yc, _, _, _ = np.linalg.lstsq(Ay, ys, rcond=None)
    return {"x_coeffs": xc, "y_coeffs": yc}


def compute_fit_residuals(
    coeffs: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """Return vertical residuals |y_pred - y_actual| for y = ax^2+bx+c."""
    a, b, c = coeffs
    y_pred = a * xs ** 2 + b * xs + c
    return np.abs(y_pred - ys)


def ransac_parabola(
    xs: np.ndarray,
    ys: np.ndarray,
    iterations: int = 300,
    threshold_px: float = 20.0,
    min_inliers: int = 8,
) -> dict:
    """RANSAC parabola fit (y = ax^2 + bx + c).

    Returns a result dict with keys:
        coeffs, inlier_mask, outlier_mask, residuals, success
    """
    n = len(xs)
    best_inlier_mask = np.zeros(n, dtype=bool)
    best_inlier_count = 0
    best_coeffs = None

    if n < 3:
        return {
            "coeffs": None,
            "inlier_mask": best_inlier_mask,
            "outlier_mask": ~best_inlier_mask,
            "residuals": np.zeros(n),
            "success": False,
        }

    for _ in range(iterations):
        sample_idx = random.sample(range(n), 3)
        xs_s = xs[sample_idx]
        ys_s = ys[sample_idx]
        coeffs = fit_parabola_xy(xs_s, ys_s)
        if coeffs is None:
            continue
        residuals = compute_fit_residuals(coeffs, xs, ys)
        inlier_mask = residuals < threshold_px
        count = int(inlier_mask.sum())
        if count > best_inlier_count:
            best_inlier_count = count
            best_inlier_mask = inlier_mask.copy()
            best_coeffs = coeffs

    # Refit using all inliers
    if best_inlier_count >= min_inliers and best_coeffs is not None:
        best_coeffs = fit_parabola_xy(xs[best_inlier_mask], ys[best_inlier_mask])
        success = best_coeffs is not None
    else:
        success = False

    residuals = (
        compute_fit_residuals(best_coeffs, xs, ys)
        if best_coeffs is not None
        else np.full(n, float("nan"))
    )

    return {
        "coeffs": best_coeffs,
        "inlier_mask": best_inlier_mask,
        "outlier_mask": ~best_inlier_mask,
        "residuals": residuals,
        "success": success,
    }
