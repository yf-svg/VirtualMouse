from __future__ import annotations

import unittest

from app.gestures.disambiguate import choose_one
from app.gestures.registry import GestureSnapshot
from app.gestures.sets.labels import ALL_ALLOWED_LABELS, AUTH_LABELS, OPS_LABELS, PRESENTATION_LABELS
from app.gestures.sets.ops_set import OPS_ALLOWED, OPS_PRIORITY, ops_runtime_suite_kwargs
from app.gestures.suite import GestureSuite


class GestureSuiteConfigTests(unittest.TestCase):
    def test_canonical_label_registry_matches_current_set_contracts(self):
        self.assertEqual(ALL_ALLOWED_LABELS, frozenset(AUTH_LABELS | OPS_LABELS))
        self.assertTrue(PRESENTATION_LABELS.issubset(OPS_LABELS))

    def test_ops_set_includes_thumbs_down(self):
        self.assertIn("THUMBS_DOWN", OPS_ALLOWED)
        self.assertIn("THUMBS_DOWN", OPS_PRIORITY)
        self.assertIn("SHAKA", OPS_ALLOWED)
        self.assertIn("SHAKA", OPS_PRIORITY)

    def test_runtime_gesture_suite_allows_thumbs_down(self):
        suite = GestureSuite()
        self.assertIsNotNone(suite.engine.allowed)
        self.assertIn("THUMBS_DOWN", suite.engine.allowed)
        self.assertIn("SHAKA", suite.engine.allowed)
        self.assertEqual(suite.engine.allowed, OPS_ALLOWED)

    def test_ops_runtime_suite_enables_priority_without_changing_global_default(self):
        default_suite = GestureSuite()
        runtime_suite = GestureSuite(**ops_runtime_suite_kwargs())

        self.assertFalse(default_suite.engine.allow_priority)
        self.assertTrue(runtime_suite.engine.allow_priority)
        self.assertEqual(runtime_suite.engine.allowed, OPS_ALLOWED)
        self.assertEqual(runtime_suite.engine.priority, OPS_PRIORITY)

    def test_ops_runtime_priority_keeps_primary_pinch_when_competing_with_cursor_shape(self):
        snapshot = GestureSnapshot(
            pinch="PINCH_INDEX",
            fist=False,
            closed_palm=True,
            open_palm=False,
            shaka=False,
            peace_sign=False,
            number=None,
            l_gesture=True,
            bravo=False,
            thumbs_down=False,
            point_right=False,
            point_left=False,
        )

        decision = choose_one(snapshot, **ops_runtime_suite_kwargs())

        self.assertEqual(decision.active, "PINCH_INDEX")
        self.assertEqual(decision.reason, "priority")

    def test_ops_runtime_priority_keeps_secondary_pinch_when_competing_with_cursor_shape(self):
        snapshot = GestureSnapshot(
            pinch="PINCH_MIDDLE",
            fist=False,
            closed_palm=True,
            open_palm=False,
            shaka=False,
            peace_sign=False,
            number=None,
            l_gesture=True,
            bravo=False,
            thumbs_down=False,
            point_right=False,
            point_left=False,
        )

        decision = choose_one(snapshot, **ops_runtime_suite_kwargs())

        self.assertEqual(decision.active, "PINCH_MIDDLE")
        self.assertEqual(decision.reason, "priority")


if __name__ == "__main__":
    unittest.main()
