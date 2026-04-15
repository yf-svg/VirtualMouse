from __future__ import annotations

from typing import Set
from app.gestures.registry import GestureSnapshot


def snapshot_to_candidates(s: GestureSnapshot) -> Set[str]:
    """
    Convert raw snapshot signals into a flat candidate label set.
    No AUTH/OPS logic. No special cases. Everything is equal.
    """
    c: Set[str] = set()

    if s.pinch:
        c.add(s.pinch)

    if s.fist:
        c.add("FIST")

    if s.closed_palm:
        c.add("CLOSED_PALM")

    if s.open_palm:
        c.add("OPEN_PALM")

    if s.shaka:
        c.add("SHAKA")

    if s.peace_sign:
        c.add("PEACE_SIGN")

    if s.number:
        c.add(s.number)  # ONE..FIVE

    if s.l_gesture:
        c.add("L")

    if s.bravo:
        c.add("BRAVO")

    if s.thumbs_down:
        c.add("THUMBS_DOWN")

    if s.point_right:
        c.add("POINT_RIGHT")

    if s.point_left:
        c.add("POINT_LEFT")

    return c
