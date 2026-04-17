from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.gestures.features import get_landmarks, palm_center


@dataclass(frozen=True)
class CursorPoint:
    """
    Abstract cursor-space point derived from hand landmarks.
    This is intentionally gesture-agnostic so interaction logic does not depend
    on a final cursor-pose decision.
    """

    x: float
    y: float


def cursor_point_from_landmarks(
    landmarks: Any,
    *,
    anchor_mode: str = "palm_center",
    mirror_x: bool = False,
) -> CursorPoint | None:
    lm = get_landmarks(landmarks)
    if lm is None:
        return None

    if anchor_mode == "palm_center":
        point = palm_center(lm)
    else:
        raise ValueError(f"Unsupported cursor anchor mode: {anchor_mode}")

    x = float(point.x)
    if mirror_x:
        x = 1.0 - x
    return CursorPoint(x=x, y=float(point.y))


def cursor_distance(a: CursorPoint, b: CursorPoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
