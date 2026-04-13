from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.gestures.features import INDEX_TIP, get_landmarks


@dataclass(frozen=True)
class CursorPoint:
    """
    Abstract cursor-space point derived from hand landmarks.
    This is intentionally gesture-agnostic so interaction logic does not depend
    on a final cursor-pose decision.
    """

    x: float
    y: float


def cursor_point_from_landmarks(landmarks: Any) -> CursorPoint | None:
    lm = get_landmarks(landmarks)
    if lm is None or len(lm) <= INDEX_TIP:
        return None

    point = lm[INDEX_TIP]
    return CursorPoint(x=float(point.x), y=float(point.y))


def cursor_distance(a: CursorPoint, b: CursorPoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
