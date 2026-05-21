"""Frame preprocessing: ROI cropping and blur smoothing."""

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


def prepare_frame(frame: np.ndarray, config: dict) -> np.ndarray:
    """Apply ROI crop and smoothing in sequence."""
    frame = crop_roi(frame, config.get("roi", {}))
    pre = config.get("preprocess", {})
    frame = smooth_frame(
        frame,
        blur_kernel=pre.get("blur_kernel", 5),
        use_median_blur=pre.get("use_median_blur", True),
    )
    return frame
