from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Set

from app.gestures.registry import GestureRegistry, GestureSnapshot
from app.gestures.disambiguate import choose_one, Decision
from app.gestures.temporal import TemporalFilter, TemporalCfg, TemporalOut
from app.gestures.rules import snapshot_to_candidates


@dataclass
class EngineOut:
    snapshot: Optional[GestureSnapshot]
    candidates: Set[str]
    decision: Decision
    temporal: TemporalOut


class GestureEngine:
    """
    Validation-only gesture engine:
      - No AUTH/OPS modes
      - No special casing for any gesture
      - Exposes candidates, ambiguity rate, stable output, edges
    """

    def __init__(
        self,
        temporal_cfg: TemporalCfg = TemporalCfg(),
        allowed: Optional[Iterable[str]] = None,
        priority: Optional[Iterable[str]] = None,
        allow_priority: bool = False,
    ):
        self.registry = GestureRegistry()
        self.temporal = TemporalFilter(temporal_cfg)
        self.allowed = set(allowed) if allowed is not None else None
        self.priority = list(priority) if priority is not None else None
        self.allow_priority = allow_priority

    def reset(self) -> None:
        self.registry.reset()
        self.temporal.reset()

    def process(self, hand_landmarks: Any | None) -> EngineOut:
        # No hand -> reset stateful detectors and temporal filter
        if hand_landmarks is None:
            self.registry.reset()
            t = self.temporal.update(None)
            empty_dec = Decision(active=None, candidates=set(), reason="none")
            return EngineOut(snapshot=None, candidates=set(), decision=empty_dec, temporal=t)

        snap = self.registry.detect(hand_landmarks)
        candidates = snapshot_to_candidates(snap)
        if self.allowed is not None:
            candidates = {c for c in candidates if c in self.allowed}

        # Disambiguation is validation-first: ambiguous => active None
        decision = choose_one(
            snap,
            allowed=self.allowed,
            priority=self.priority,
            allow_priority=self.allow_priority,
        )

        # Temporal stabilizer runs on chosen label (can be None)
        t = self.temporal.update(decision.active)

        return EngineOut(snapshot=snap, candidates=candidates, decision=decision, temporal=t)
