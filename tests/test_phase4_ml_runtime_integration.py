from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gestures.classifier import SVMClassifier
from app.gestures.disambiguate import Decision
from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION, FeatureVector
from app.gestures.suite import GestureSuite
from app.gestures.temporal import FeatureTemporalOut


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


def _engine_out(
    *,
    rule_label: str | None,
    candidates: set[str] | None = None,
    feature_reason: str = "ok",
    feature_vector: FeatureVector | None = None,
):
    vector = feature_vector or _feature_vector()
    return SimpleNamespace(
        decision=Decision(active=rule_label, candidates=candidates or set(), reason="single" if rule_label else "none"),
        candidates=candidates or ({rule_label} if rule_label else set()),
        feature_vector=vector,
        feature_temporal=_feature_temporal(feature_reason, feature_vector=vector),
    )


class _FakeProbModel:
    def __init__(self, *, classes: list[str], probs: list[float]):
        self.classes_ = classes
        self._probs = probs

    def predict_proba(self, rows):
        return [self._probs for _ in rows]


class _FakeEngine:
    def __init__(self, out):
        self.out = out
        self.reset_called = False

    def process(self, hand_landmarks):
        return self.out

    def reset(self):
        self.reset_called = True


class Phase4MlRuntimeIntegrationTests(unittest.TestCase):
    def test_classifier_reports_missing_model_without_crashing(self):
        classifier = SVMClassifier()
        prediction = classifier.predict(_feature_vector())
        self.assertFalse(prediction.accepted)
        self.assertEqual(prediction.reason, "model_unavailable")

    def test_classifier_accepts_probability_backed_prediction(self):
        classifier = SVMClassifier(
            allowed={"BRAVO", "FIST"},
            model_bundle={
                "model": _FakeProbModel(classes=["BRAVO", "FIST"], probs=[0.84, 0.16]),
                "schema_version": FEATURE_SCHEMA_VERSION,
                "feature_dimension": FEATURE_DIMENSION,
                "labels": ["BRAVO", "FIST"],
                "min_confidence": 0.70,
            },
        )
        prediction = classifier.predict(_feature_vector())
        self.assertTrue(prediction.accepted)
        self.assertEqual(prediction.label, "BRAVO")
        self.assertAlmostEqual(prediction.confidence or 0.0, 0.84, places=2)

    def test_suite_falls_back_to_rules_when_model_is_missing(self):
        suite = GestureSuite()
        suite.engine = _FakeEngine(_engine_out(rule_label="FIST"))

        out = suite.detect(object())

        self.assertEqual(out.chosen, "FIST")
        self.assertEqual(out.source, "rules")
        self.assertEqual(out.rule_chosen, "FIST")
        self.assertEqual(out.ml_reason, "model_unavailable")

    def test_suite_prefers_ml_when_prediction_is_accepted(self):
        classifier = SVMClassifier(
            allowed={"BRAVO", "FIST"},
            model_bundle={
                "model": _FakeProbModel(classes=["BRAVO", "FIST"], probs=[0.91, 0.09]),
                "schema_version": FEATURE_SCHEMA_VERSION,
                "feature_dimension": FEATURE_DIMENSION,
                "labels": ["BRAVO", "FIST"],
                "min_confidence": 0.70,
            },
        )
        suite = GestureSuite(classifier=classifier)
        suite.engine = _FakeEngine(_engine_out(rule_label="FIST", candidates={"FIST", "BRAVO"}))

        out = suite.detect(object())

        self.assertEqual(out.chosen, "BRAVO")
        self.assertEqual(out.source, "ml")
        self.assertEqual(out.ml_chosen, "BRAVO")
        self.assertAlmostEqual(out.confidence or 0.0, 0.91, places=2)

    def test_suite_falls_back_when_feature_stream_is_not_ready_for_ml(self):
        classifier = SVMClassifier(
            allowed={"BRAVO", "FIST"},
            model_bundle={
                "model": _FakeProbModel(classes=["BRAVO", "FIST"], probs=[0.93, 0.07]),
                "schema_version": FEATURE_SCHEMA_VERSION,
                "feature_dimension": FEATURE_DIMENSION,
                "labels": ["BRAVO", "FIST"],
                "min_confidence": 0.70,
            },
        )
        suite = GestureSuite(classifier=classifier)
        suite.engine = _FakeEngine(_engine_out(rule_label="FIST", feature_reason="warming_up"))

        out = suite.detect(object())

        self.assertEqual(out.chosen, "FIST")
        self.assertEqual(out.source, "rules")
        self.assertIn("feature_warming_up", out.reason)

    def test_suite_falls_back_when_ml_confidence_is_too_low(self):
        classifier = SVMClassifier(
            allowed={"BRAVO", "FIST"},
            model_bundle={
                "model": _FakeProbModel(classes=["BRAVO", "FIST"], probs=[0.55, 0.45]),
                "schema_version": FEATURE_SCHEMA_VERSION,
                "feature_dimension": FEATURE_DIMENSION,
                "labels": ["BRAVO", "FIST"],
                "min_confidence": 0.70,
            },
        )
        suite = GestureSuite(classifier=classifier)
        suite.engine = _FakeEngine(_engine_out(rule_label="FIST"))

        out = suite.detect(object())

        self.assertEqual(out.chosen, "FIST")
        self.assertEqual(out.source, "rules")
        self.assertEqual(out.ml_reason, "low_confidence")
        self.assertIn("low_confidence", out.reason)


if __name__ == "__main__":
    unittest.main()
