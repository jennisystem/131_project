"""Simple motion-continuity tracker linking per-frame candidates."""

import math
from typing import Optional


def predict_next_position(
    track_history: list[dict],
) -> Optional[tuple[float, float]]:
    """Predict next ball position from track history.

    Uses constant-velocity model when two or more detections are available,
    otherwise returns the last known position.
    """
    valid = [d for d in track_history if d.get("x") is not None]
    if len(valid) == 0:
        return None
    if len(valid) == 1:
        return valid[-1]["x"], valid[-1]["y"]
    last = valid[-1]
    prev = valid[-2]
    px = last["x"] + (last["x"] - prev["x"])
    py = last["y"] + (last["y"] - prev["y"])
    return px, py


def select_candidate(
    candidates: list[dict],
    predicted_position: Optional[tuple[float, float]],
    max_jump_px: float,
) -> Optional[dict]:
    """Choose the candidate closest to the predicted position.

    Returns None if no candidate is within max_jump_px.
    """
    if not candidates:
        return None
    if predicted_position is None:
        return candidates[0]

    px, py = predicted_position
    best = None
    best_dist = float("inf")
    for c in candidates:
        dist = math.hypot(c["x"] - px, c["y"] - py)
        if dist < best_dist:
            best_dist = dist
            best = c

    if best_dist > max_jump_px:
        return None
    return best


def track_candidates(
    all_candidates: list[list[dict]],
    config: dict,
) -> list[dict]:
    """Link per-frame candidate lists into a temporal track.

    Parameters
    ----------
    all_candidates:
        List of candidate lists, one per frame (in frame order).
    config:
        Full pipeline config dict.

    Returns
    -------
    List of detection dicts with keys:
        frame_idx, x, y, radius, score, status
    """
    track_cfg = config.get("tracking", {})
    max_jump = track_cfg.get("max_jump_px", 120)
    reacquire_after_missing = track_cfg.get("reacquire_after_missing", 8)

    detections: list[dict] = []
    history: list[dict] = []

    for frame_idx, candidates in enumerate(all_candidates):
        predicted = predict_next_position(history)
        recent_missing = 0
        for det in reversed(history):
            if det.get("x") is not None:
                break
            recent_missing += 1
        if recent_missing >= reacquire_after_missing:
            predicted = None

        # On the very first frame with no history, pick highest-scoring candidate.
        if predicted is None:
            chosen = candidates[0] if candidates else None
        else:
            chosen = select_candidate(candidates, predicted, max_jump)

        if chosen is not None:
            det = {
                "frame_idx": frame_idx,
                "x": chosen["x"],
                "y": chosen["y"],
                "radius": chosen["radius"],
                "score": chosen["score"],
                "status": "tracked",
            }
        else:
            det = {
                "frame_idx": frame_idx,
                "x": None,
                "y": None,
                "radius": None,
                "score": None,
                "status": "missing",
            }

        detections.append(det)
        history.append(det)

    return detections
