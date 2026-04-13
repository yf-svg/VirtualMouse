from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gestures.classifier import ClassifierPrediction
from app.gestures.disambiguate import Decision
from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION, FeatureVector
from app.gestures.suite import GestureSuite
from app.gestures.temporal import FeatureTemporalOut, PredictionGate, PredictionGateCfg


def _feature_vector() -> FeatureVector:
    return FeatureVector(
        values=tuple(0.01 * idx for idx in range(FEATURE_DIMENSION)),
        schema_version=FEATURE_SCHEMA_VERSION,
    )


def _feature_temporal(reason: str = "ok", *, feature_vector: FeatureVector | None = None) -> FeatureTemporalOut:
    vector = feature_vector or _feature_vector()
    return FeatureTemporalOut(
        smoothed=vector,
        ready=reason != "warming_up",
        window_size=5 if reason != "warming_up" else 2,
        schema_version=vector.schema_version,
        instability_score=0.01 if reason == "ok" else 0.20,
        pairwise_delta_median=0.01 if reason == "ok" else 0.20,
        latest_deviation=0.01 if reason == "ok" else 0.20,
        passed=reason == "ok",
        reason=reason,
    )


def _engine_out(*, rule_label: str | None, feature_reason: str = "ok"):
    vector = _feature_vector()
    return SimpleNamespace(
        decision=Decision(active=rule_label, candidates={rule_label} if rule_label else set(), reason="single" if rule_label else "none"),
        candidates={rule_label} if rule_label else set(),
        feature_vector=vector,
        feature_temporal=_feature_temporal(feature_reason, feature_vector=vector),
    )


class _SequenceEngine:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self._index = 0

    def process(self, hand_landmarks):
        out = self._outputs[self._index]
        self._index = min(self._index + 1, len(self._outputs) - 1)
        return out

    def reset(self):
        self._index = 0


class _SequenceClassifier:
    def __init__(self, predictions, *, min_confidence: float = 0.70):
        self._predictions = list(predictions)
        self._index = 0
        self.min_confidence = min_confidence

    def predict(self, feature_vector):
        prediction = self._predictions[self._index]
        self._index = min(self._index + 1, len(self._predictions) - 1)
        return prediction


class Phase43RuntimePredictionGateTests(unittest.TestCase):
    def test_prediction_gate_requires_hold_before_eligible(self):
        gate = PredictionGate(PredictionGateCfg(enter_confirm=1, switch_confirm=2, release_confirm=2, eligible_hold=2))

        first = gate.update("FIST", source="rules")
        second = gate.update("FIST", source="rules")

        self.assertEqual(first.stable, "FIST")
        self.assertIsNone(first.eligible)
        self.assertEqual(first.hold_frames, 1)
        self.assertEqual(second.stable, "FIST")
        self.assertEqual(second.eligible, "FIST")
        self.assertEqual(second.down, "FIST")
        self.assertIsNone(second.up)

    def test_prediction_gate_requires_two_frames_to_switch(self):
        gate = PredictionGate(PredictionGateCfg(enter_confirm=1, switch_confirm=2, release_confirm=2, eligible_hold=2))
        gate.update("FIST", source="rules")
        gate.update("FIST", source="rules")

        first_switch = gate.update("BRAVO", source="rules")
        second_switch = gate.update("BRAVO", source="rules")

        self.assertEqual(first_switch.stable, "FIST")
        self.assertEqual(first_switch.eligible, "FIST")
        self.assertEqual(first_switch.pending_label, "BRAVO")
        self.assertEqual(first_switch.pending_frames, 1)
        self.assertEqual(second_switch.stable, "BRAVO")
        self.assertEqual(second_switch.eligible, "BRAVO")
        self.assertEqual(second_switch.up, "FIST")
        self.assertEqual(second_switch.down, "BRAVO")

    def test_prediction_gate_requires_two_frames_to_release(self):
        gate = PredictionGate(PredictionGateCfg(enter_confirm=1, switch_confirm=2, release_confirm=2, eligible_hold=2))
        gate.update("FIST", source="rules")
        gate.update("FIST", source="rules")

        first_none = gate.update(None, source="none")
        second_none = gate.update(None, source="none")

        self.assertEqual(first_none.stable, "FIST")
        self.assertEqual(first_none.eligible, "FIST")
        self.assertEqual(first_none.pending_frames, 1)
        self.assertIsNone(first_none.up)
        self.assertIsNone(second_none.stable)
        self.assertIsNone(second_none.eligible)
        self.assertEqual(second_none.up, "FIST")

    def test_suite_keeps_ml_label_during_small_confidence_dip(self):
        suite = GestureSuite(
            classifier=_SequenceClassifier(
                [
                    ClassifierPrediction(
                        label="BRAVO",
                        confidence=0.82,
                        accepted=True,
                        reason="accepted",
                        model_path="models/gesture_svm.joblib",
                    ),
                    ClassifierPrediction(
                        label="BRAVO",
                        confidence=0.64,
                        accepted=False,
                        reason="low_confidence",
                        model_path="models/gesture_svm.joblib",
                    ),
                ],
                min_confidence=0.70,
            ),
            prediction_gate_cfg=PredictionGateCfg(enter_confirm=1, switch_confirm=2, release_confirm=2, eligible_hold=2),
        )
        suite.engine = _SequenceEngine(
            [
                _engine_out(rule_label="FIST"),
                _engine_out(rule_label="FIST"),
            ]
        )

        first = suite.detect(object())
        second = suite.detect(object())

        self.assertEqual(first.source, "ml")
        self.assertEqual(first.chosen, "BRAVO")
        self.assertEqual(second.source, "ml")
        self.assertEqual(second.chosen, "BRAVO")
        self.assertEqual(second.reason, "ml:hysteresis_hold")
        self.assertEqual(second.stable, "BRAVO")
        self.assertEqual(second.eligible, "BRAVO")


if __name__ == "__main__":
    unittest.main()
