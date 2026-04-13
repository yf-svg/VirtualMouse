# app/gestures/suite.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Set

from app.gestures.classifier import ClassifierPrediction, SVMClassifier
from app.gestures.engine import GestureEngine
from app.gestures.temporal import PredictionGate, PredictionGateCfg, TemporalCfg
from app.gestures.sets.ops_set import OPS_ALLOWED, OPS_PRIORITY


@dataclass
class GestureSuiteOut:
    chosen: Optional[str]
    stable: Optional[str]
    eligible: Optional[str]
    candidates: Set[str]
    reason: str
    down: Optional[str]
    up: Optional[str]
    source: str
    confidence: float | None
    rule_chosen: Optional[str]
    ml_chosen: Optional[str]
    ml_reason: str
    feature_reason: str
    hold_frames: int
    gate_reason: str


class GestureSuite:
    """
    Runtime classifier wrapper.
    ML is the primary path when a validated runtime model is available.
    Rules remain the safe fallback when the model is absent or rejected.
    """

    def __init__(
        self,
        *,
        classifier: SVMClassifier | None = None,
        temporal_cfg: TemporalCfg | None = None,
        prediction_gate_cfg: PredictionGateCfg | None = None,
        allowed: Iterable[str] | None = None,
        priority: Iterable[str] | None = None,
    ):
        allowed_labels = set(allowed) if allowed is not None else OPS_ALLOWED
        priority_labels = list(priority) if priority is not None else OPS_PRIORITY
        runtime_temporal_cfg = temporal_cfg or TemporalCfg(window=5, confirm=2, min_hold=1)
        self.engine = GestureEngine(
            runtime_temporal_cfg,
            allowed=allowed_labels,
            priority=priority_labels,
            allow_priority=False,
        )
        self.classifier = classifier or SVMClassifier(allowed=allowed_labels)
        self.prediction_gate = PredictionGate(prediction_gate_cfg)

    def reset(self) -> None:
        self.engine.reset()
        self.prediction_gate.reset()

    def _allow_ml_hysteresis_hold(
        self,
        *,
        ml_prediction: ClassifierPrediction,
        feature_reason: str,
    ) -> bool:
        stable_label = self.prediction_gate.stable_label
        stable_source = self.prediction_gate.stable_source
        if stable_label is None or stable_source != "ml":
            return False
        if feature_reason != "ok":
            return False
        if ml_prediction.label != stable_label:
            return False
        if ml_prediction.confidence is None:
            return False
        min_confidence = float(getattr(self.classifier, "min_confidence", 0.70))
        stay_threshold = max(0.0, min_confidence - self.prediction_gate.cfg.ml_hysteresis_delta)
        return ml_prediction.confidence >= stay_threshold

    def _select_primary_label(
        self,
        *,
        ml_prediction: ClassifierPrediction,
        rule_label: str | None,
        feature_reason: str,
    ) -> tuple[str | None, str, str]:
        if ml_prediction.accepted and feature_reason == "ok":
            return ml_prediction.label, "ml", "ml:accepted"

        if self._allow_ml_hysteresis_hold(
            ml_prediction=ml_prediction,
            feature_reason=feature_reason,
        ):
            return ml_prediction.label, "ml", "ml:hysteresis_hold"

        if rule_label is not None:
            fallback_reason = ml_prediction.reason
            if fallback_reason == "accepted" and feature_reason != "ok":
                fallback_reason = f"feature_{feature_reason}"
            return rule_label, "rules", f"rule_fallback:{fallback_reason}"

        if ml_prediction.accepted:
            return None, "none", f"ml_rejected:feature_{feature_reason}"

        return None, "none", f"none:{ml_prediction.reason}"

    def detect(self, hand_landmarks: Any) -> GestureSuiteOut:
        out = self.engine.process(hand_landmarks)
        feature_vector = out.feature_temporal.smoothed or out.feature_vector
        ml_prediction = self.classifier.predict(feature_vector)
        chosen, source, reason = self._select_primary_label(
            ml_prediction=ml_prediction,
            rule_label=out.decision.active,
            feature_reason=out.feature_temporal.reason,
        )
        temporal = self.prediction_gate.update(chosen, source=source)
        return GestureSuiteOut(
            chosen=chosen,
            stable=temporal.stable,
            eligible=temporal.eligible,
            candidates=out.candidates,
            reason=reason,
            down=temporal.down,
            up=temporal.up,
            source=source,
            confidence=ml_prediction.confidence if source == "ml" else None,
            rule_chosen=out.decision.active,
            ml_chosen=ml_prediction.label,
            ml_reason=ml_prediction.reason,
            feature_reason=out.feature_temporal.reason,
            hold_frames=temporal.hold_frames,
            gate_reason=temporal.reason,
        )
