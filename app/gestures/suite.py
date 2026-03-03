# app/gestures/suite.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Set

from app.gestures.engine import GestureEngine
from app.gestures.temporal import TemporalCfg
from app.gestures.sets.ops_set import OPS_ALLOWED, OPS_PRIORITY


@dataclass
class GestureSuiteOut:
    chosen: Optional[str]
    stable: Optional[str]
    candidates: Set[str]
    reason: str
    down: Optional[str]
    up: Optional[str]


class GestureSuite:
    """
    Backwards-compatible wrapper for your runtime.
    Uses the new validation-only GestureEngine internally.
    """

    def __init__(self):
        self.engine = GestureEngine(
            TemporalCfg(window=7, confirm=4, min_hold=2),
            allowed=OPS_ALLOWED,
            priority=OPS_PRIORITY,
            allow_priority=False,
        )

    def reset(self) -> None:
        self.engine.reset()

    def detect(self, hand_landmarks: Any) -> GestureSuiteOut:
        out = self.engine.process(hand_landmarks)
        return GestureSuiteOut(
            chosen=out.decision.active,
            stable=out.temporal.stable,
            candidates=out.candidates,
            reason=out.decision.reason,
            down=out.temporal.down,
            up=out.temporal.up,
        )
