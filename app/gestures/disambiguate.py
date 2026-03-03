from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Iterable, Set

from app.gestures.registry import GestureSnapshot
from app.gestures.rules import snapshot_to_candidates


@dataclass
class Decision:
    active: Optional[str]      # chosen label (or None)
    candidates: Set[str]       # all labels detected in this frame
    reason: str                # "single" | "ambiguous" | "none" | "priority"


def choose_one(
    snapshot: GestureSnapshot,
    allowed: Optional[Iterable[str]] = None,
    priority: Optional[Iterable[str]] = None,
    allow_priority: bool = False,
) -> Decision:
    """
    Validation-first disambiguation:
      - If exactly 1 candidate -> choose it
      - If 0 -> None
      - If >1 -> None (AMBIGUOUS) by default so you can measure overlap
    Optional:
      - If allow_priority=True and priority provided, pick the first match.
    """
    candidates = snapshot_to_candidates(snapshot)
    if allowed is not None:
        allowed_set = set(allowed)
        candidates = {c for c in candidates if c in allowed_set}

    if not candidates:
        return Decision(active=None, candidates=candidates, reason="none")

    if len(candidates) == 1:
        return Decision(active=next(iter(candidates)), candidates=candidates, reason="single")

    if allow_priority and priority is not None:
        for g in priority:
            if g in candidates:
                return Decision(active=g, candidates=candidates, reason="priority")

    return Decision(active=None, candidates=candidates, reason="ambiguous")
