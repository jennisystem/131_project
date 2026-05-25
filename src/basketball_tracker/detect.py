"""HSV-based basketball detection via color thresholding and contour analysis."""

import math

import cv2
import numpy as np


def make_hsv_mask(
    frame_bgr: np.ndarray, lower: list[int], upper: list[int]
) -> np.ndarray:
    """Return a binary mask isolating pixels within the HSV range."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Red hues wrap around OpenCV's HSV scale, so allow multiple ranges:
    # lower: [[0, 25, 35], [165, 25, 35]]
    # upper: [[15, 255, 255], [179, 255, 255]]
    if lower and isinstance(lower[0], list):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo_vals, hi_vals in zip(lower, upper):
            lo = np.array(lo_vals, dtype=np.uint8)
            hi = np.array(hi_vals, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        return mask

    lo = np.array(lower, dtype=np.uint8)
    hi = np.array(upper, dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def clean_mask(
    mask: np.ndarray, open_kernel: int = 3, close_kernel: int = 7
) -> np.ndarray:
    """Remove small noise (open) and fill gaps (close) in the binary mask."""
    k_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
    )
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def find_ball_candidates(mask: np.ndarray) -> list[dict]:
    """Extract contour-based candidates from a cleaned binary mask."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-3:
            continue

        circularity = (4 * math.pi * area) / (perimeter ** 2)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        (ex, ey), radius = cv2.minEnclosingCircle(cnt)
        x_bbox, y_bbox, w_bbox, h_bbox = cv2.boundingRect(cnt)

        candidates.append(
            {
                "x": cx,
                "y": cy,
                "radius": float(radius),
                "area": float(area),
                "circularity": float(circularity),
                "score": 0.0,
                "bbox": [x_bbox, y_bbox, w_bbox, h_bbox],
            }
        )
    return candidates


def filter_candidates(
    candidates: list[dict],
    min_area: float,
    max_area: float,
    min_circularity: float,
    min_x: float | None = None,
    max_x: float | None = None,
    min_y: float | None = None,
    max_y: float | None = None,
    min_radius: float | None = None,
    max_radius: float | None = None,
) -> list[dict]:
    """Keep only candidates meeting area and circularity thresholds."""
    filtered = []
    for c in candidates:
        if not (min_area <= c["area"] <= max_area):
            continue
        if c["circularity"] < min_circularity:
            continue
        if min_x is not None and c["x"] < min_x:
            continue
        if max_x is not None and c["x"] > max_x:
            continue
        if min_y is not None and c["y"] < min_y:
            continue
        if max_y is not None and c["y"] > max_y:
            continue
        if min_radius is not None and c["radius"] < min_radius:
            continue
        if max_radius is not None and c["radius"] > max_radius:
            continue
        filtered.append(c)
    return filtered


def _score_candidate(candidate: dict, ideal_area: float = 500.0) -> float:
    """Higher is better: rewards circularity and penalises extreme sizes."""
    circ_score = candidate["circularity"]
    size_score = 1.0 - abs(math.log(candidate["area"] + 1) - math.log(ideal_area + 1)) / 10.0
    size_score = max(0.0, size_score)
    return circ_score * 0.7 + size_score * 0.3


def detect_ball_hsv(frame_bgr: np.ndarray, config: dict) -> list[dict]:
    """Full HSV detection pipeline for one frame.

    Returns candidates sorted by score descending.
    """
    hsv_cfg = config.get("hsv_threshold", {})
    morph_cfg = config.get("morphology", {})
    filt_cfg = config.get("contour_filter", {})

    mask = make_hsv_mask(
        frame_bgr,
        lower=hsv_cfg.get("lower", [5, 80, 80]),
        upper=hsv_cfg.get("upper", [25, 255, 255]),
    )
    mask = clean_mask(
        mask,
        open_kernel=morph_cfg.get("open_kernel", 3),
        close_kernel=morph_cfg.get("close_kernel", 7),
    )
    candidates = find_ball_candidates(mask)
    candidates = filter_candidates(
        candidates,
        min_area=filt_cfg.get("min_area", 40),
        max_area=filt_cfg.get("max_area", 5000),
        min_circularity=filt_cfg.get("min_circularity", 0.35),
        min_x=filt_cfg.get("min_x"),
        max_x=filt_cfg.get("max_x"),
        min_y=filt_cfg.get("min_y"),
        max_y=filt_cfg.get("max_y"),
        min_radius=filt_cfg.get("min_radius"),
        max_radius=filt_cfg.get("max_radius"),
    )
    ideal_area = filt_cfg.get("ideal_area", 500.0)
    for c in candidates:
        c["score"] = _score_candidate(c, ideal_area=ideal_area)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates
