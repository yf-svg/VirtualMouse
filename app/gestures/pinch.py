from __future__ import annotations

import math

# MediaPipe landmark indices
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
WRIST = 0


def _dist(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)


def detect_pinch_type(landmarks, pinch_thresh: float = 0.055, margin: float = 0.010):
    """
    Returns one of:
      - "PINCH_INDEX"
      - "PINCH_MIDDLE"
      - None (no confident pinch)

    Uses RAW landmarks recommended.
    pinch_thresh: absolute normalized distance threshold
    margin: winner must be smaller than runner-up by this margin
    """
    lm = landmarks.landmark
    thumb = lm[THUMB_TIP]

    d_index = _dist(thumb, lm[INDEX_TIP])
    d_middle = _dist(thumb, lm[MIDDLE_TIP])

    # optional: can help reject weird cases
    # d_ring = _dist(thumb, lm[RING_TIP])
    # d_pinky = _dist(thumb, lm[PINKY_TIP])

    # Must be under threshold
    best = min(d_index, d_middle)
    if best > pinch_thresh:
        return None

    # Require separation margin to avoid ambiguity
    if abs(d_index - d_middle) < margin:
        return None

    return "PINCH_INDEX" if d_index < d_middle else "PINCH_MIDDLE"