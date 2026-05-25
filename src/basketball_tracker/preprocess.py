"""Frame preprocessing: ROI cropping, masking, and blur smoothing."""

import cv2
import numpy as np


def crop_roi(frame: np.ndarray, roi_config: dict) -> np.ndarray:
    """Crop frame to region of interest if enabled."""
    if not roi_config.get("use_roi", False):
        return frame
    x = roi_config["x"]
    y = roi_config["y"]
    w = roi_config["w"]
    h = roi_config["h"]
    return frame[y : y + h, x : x + w]


def smooth_frame(
    frame: np.ndarray, blur_kernel: int = 5, use_median_blur: bool = True
) -> np.ndarray:
    """Apply median or Gaussian blur to reduce noise."""
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    if use_median_blur:
        return cv2.medianBlur(frame, k)
    return cv2.GaussianBlur(frame, (k, k), 0)


def apply_detection_mask(frame: np.ndarray, mask_config: dict) -> np.ndarray:
    """Suppress background regions before color/contour detection.

    This keeps the original frame size, so detection coordinates still line up
    with the unmodified video frames used for annotation.
    """
    if not mask_config.get("use_mask", False):
        return frame

    h, w = frame.shape[:2]
    keep_mask = np.zeros((h, w), dtype=np.uint8)
    regions = mask_config.get("keep_regions", [])

    for region in regions:
        x = max(0, int(region.get("x", 0)))
        y = max(0, int(region.get("y", 0)))
        rw = max(0, int(region.get("w", w)))
        rh = max(0, int(region.get("h", h)))
        x2 = min(w, x + rw)
        y2 = min(h, y + rh)
        keep_mask[y:y2, x:x2] = 255

    if not regions:
        return frame

    outside_mode = mask_config.get("outside_mode", "black")
    if outside_mode == "blur":
        blur_kernel = int(mask_config.get("blur_kernel", 51))
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        background = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
    else:
        background = np.zeros_like(frame)

    return np.where(keep_mask[:, :, None] == 255, frame, background)


def suppress_hsv_background(frame: np.ndarray, suppress_config: dict) -> np.ndarray:
    """Replace configured HSV background colors with black or blur."""
    if not suppress_config.get("use_suppression", False):
        return frame

    lower = suppress_config.get("lower", [])
    upper = suppress_config.get("upper", [])
    if not lower or not upper:
        return frame

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    suppress_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    if lower and isinstance(lower[0], list):
        ranges = zip(lower, upper)
    else:
        ranges = [(lower, upper)]

    for lo_vals, hi_vals in ranges:
        lo = np.array(lo_vals, dtype=np.uint8)
        hi = np.array(hi_vals, dtype=np.uint8)
        suppress_mask = cv2.bitwise_or(suppress_mask, cv2.inRange(hsv, lo, hi))

    regions = suppress_config.get("regions", [])
    if regions:
        h, w = frame.shape[:2]
        region_mask = np.zeros((h, w), dtype=np.uint8)
        for region in regions:
            x = max(0, int(region.get("x", 0)))
            y = max(0, int(region.get("y", 0)))
            rw = max(0, int(region.get("w", w)))
            rh = max(0, int(region.get("h", h)))
            x2 = min(w, x + rw)
            y2 = min(h, y + rh)
            region_mask[y:y2, x:x2] = 255
        suppress_mask = cv2.bitwise_and(suppress_mask, region_mask)

    mode = suppress_config.get("mode", "black")
    if mode == "blur":
        blur_kernel = int(suppress_config.get("blur_kernel", 51))
        blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        replacement = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
    else:
        replacement = np.zeros_like(frame)

    return np.where(suppress_mask[:, :, None] == 255, replacement, frame)


def prepare_frame(frame: np.ndarray, config: dict) -> np.ndarray:
    """Apply ROI crop, smoothing, and optional detection masking."""
    frame = crop_roi(frame, config.get("roi", {}))
    pre = config.get("preprocess", {})
    frame = smooth_frame(
        frame,
        blur_kernel=pre.get("blur_kernel", 5),
        use_median_blur=pre.get("use_median_blur", True),
    )
    frame = apply_detection_mask(frame, config.get("detection_mask", {}))
    frame = suppress_hsv_background(frame, config.get("background_suppression", {}))
    return frame
