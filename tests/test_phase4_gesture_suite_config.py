from __future__ import annotations

import unittest

from app.gestures.sets.labels import ALL_ALLOWED_LABELS, AUTH_LABELS, OPS_LABELS, PRESENTATION_LABELS
from app.gestures.sets.ops_set import OPS_ALLOWED, OPS_PRIORITY
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


if __name__ == "__main__":
    unittest.main()
