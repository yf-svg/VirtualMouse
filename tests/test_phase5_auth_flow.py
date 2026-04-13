from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gestures.disambiguate import Decision
from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION, FeatureVector
from app.gestures.sets.auth_set import AUTH_ALLOWED
from app.gestures.suite import GestureSuite
from app.gestures.temporal import FeatureTemporalOut
from app.security.auth import GestureAuth, GestureAuthCfg


def _feature_vector() -> FeatureVector:
    return FeatureVector(
        values=tuple(0.01 * idx for idx in range(FEATURE_DIMENSION)),
        schema_version=FEATURE_SCHEMA_VERSION,
    )


def _feature_temporal() -> FeatureTemporalOut:
    vector = _feature_vector()
    return FeatureTemporalOut(
        smoothed=vector,
        ready=True,
        window_size=5,
        schema_version=vector.schema_version,
        instability_score=0.01,
        pairwise_delta_median=0.01,
        latest_deviation=0.01,
        passed=True,
        reason="ok",
    )


class _FakeEngine:
    def __init__(self, label: str | None):
        self.label = label

    def process(self, hand_landmarks):
        return SimpleNamespace(
            decision=Decision(active=self.label, candidates={self.label} if self.label else set(), reason="single" if self.label else "none"),
            candidates={self.label} if self.label else set(),
            feature_vector=_feature_vector(),
            feature_temporal=_feature_temporal(),
        )

    def reset(self):
        pass


class Phase5AuthFlowTests(unittest.TestCase):
    def test_auth_sequence_succeeds_in_order(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"), step_timeout_s=4.0))

        first = auth.update("ONE", now=1.0)
        second = auth.update("TWO", now=2.0)
        third = auth.update("THREE", now=3.0)

        self.assertEqual(first.status, "started")
        self.assertEqual(second.status, "progress")
        self.assertEqual(third.status, "success")
        self.assertTrue(third.authenticated)

    def test_auth_resets_on_wrong_gesture(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"), step_timeout_s=4.0))
        auth.update("ONE", now=1.0)

        out = auth.update("FOUR", now=2.0)

        self.assertEqual(out.status, "reset_wrong")
        self.assertEqual(out.matched_steps, 0)
        self.assertEqual(out.expected_next, "ONE")
        self.assertEqual(out.failed_attempts, 1)

    def test_auth_resets_on_timeout(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), step_timeout_s=2.0))
        auth.update("ONE", now=1.0)

        out = auth.update(None, now=4.5)

        self.assertEqual(out.status, "reset_timeout")
        self.assertEqual(out.matched_steps, 0)
        self.assertEqual(out.expected_next, "ONE")
        self.assertEqual(out.failed_attempts, 1)

    def test_auth_restart_accepts_first_gesture_again(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"), step_timeout_s=4.0))
        auth.update("ONE", now=1.0)

        out = auth.update("ONE", now=2.0)

        self.assertEqual(out.status, "started")
        self.assertEqual(out.matched_steps, 1)
        self.assertEqual(out.expected_next, "TWO")

    def test_auth_suite_can_be_instantiated_for_auth_labels(self):
        suite = GestureSuite(allowed=AUTH_ALLOWED)
        suite.engine = _FakeEngine("ONE")

        out = suite.detect(object())

        self.assertEqual(out.chosen, "ONE")
        self.assertEqual(out.stable, "ONE")

    def test_auth_enters_lockout_after_max_failures(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), max_failures=2, cooldown_s=5.0))

        auth.update("FOUR", now=1.0)
        out = auth.update("FIVE", now=2.0)

        self.assertEqual(out.status, "locked_out")
        self.assertEqual(out.failed_attempts, 2)
        self.assertGreater(out.retry_after_s or 0.0, 0.0)

    def test_auth_recovers_after_lockout_window(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), max_failures=1, cooldown_s=3.0))

        locked = auth.update("FOUR", now=1.0)
        during = auth.update("ONE", now=2.0)
        after = auth.update("ONE", now=4.5)

        self.assertEqual(locked.status, "locked_out")
        self.assertEqual(during.status, "locked_out")
        self.assertEqual(after.status, "started")
        self.assertEqual(after.failed_attempts, 0)


if __name__ == "__main__":
    unittest.main()
