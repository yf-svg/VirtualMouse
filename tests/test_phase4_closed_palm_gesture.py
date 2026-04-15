from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gestures.hand_gestures import HandGestures, detect_closed_palm, detect_open_palm
from app.gestures.registry import GestureRegistry
from app.gestures.rules import snapshot_to_candidates


def _make_points(coords: list[tuple[float, float, float]]):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in coords]


def _flat_closed_palm_hand():
    return _make_points(
        [
            (0.50, 0.85, 0.00),
            (0.40, 0.75, 0.00), (0.33, 0.68, 0.00), (0.29, 0.63, 0.00), (0.25, 0.60, 0.00),
            (0.44, 0.55, 0.00), (0.45, 0.40, 0.00), (0.46, 0.26, 0.00), (0.47, 0.12, 0.00),
            (0.50, 0.53, 0.00), (0.50, 0.37, 0.00), (0.50, 0.23, 0.00), (0.50, 0.09, 0.00),
            (0.56, 0.55, 0.00), (0.55, 0.39, 0.00), (0.54, 0.25, 0.00), (0.53, 0.12, 0.00),
            (0.62, 0.60, 0.00), (0.60, 0.45, 0.00), (0.59, 0.31, 0.00), (0.58, 0.18, 0.00),
        ]
    )


def _spread_open_palm_hand():
    return _make_points(
        [
            (0.50, 0.85, 0.00),
            (0.28, 0.70, 0.00), (0.20, 0.60, 0.00), (0.12, 0.52, 0.00), (0.04, 0.44, 0.00),
            (0.38, 0.55, 0.00), (0.32, 0.38, 0.00), (0.27, 0.23, 0.00), (0.22, 0.08, 0.00),
            (0.49, 0.52, 0.00), (0.49, 0.34, 0.00), (0.49, 0.19, 0.00), (0.49, 0.05, 0.00),
            (0.62, 0.54, 0.00), (0.66, 0.36, 0.00), (0.70, 0.21, 0.00), (0.74, 0.08, 0.00),
            (0.76, 0.60, 0.00), (0.82, 0.44, 0.00), (0.87, 0.30, 0.00), (0.91, 0.18, 0.00),
        ]
    )


class ClosedPalmGestureTests(unittest.TestCase):
    def test_detects_flat_closed_palm_and_rejects_open_palm(self):
        hand = HandGestures()
        pose = _flat_closed_palm_hand()

        self.assertTrue(hand.detect_closed_palm(pose))
        self.assertTrue(detect_closed_palm(pose))
        self.assertFalse(hand.detect_open_palm(pose))
        self.assertFalse(detect_open_palm(pose))

    def test_open_palm_does_not_collapse_into_closed_palm(self):
        hand = HandGestures()
        pose = _spread_open_palm_hand()

        self.assertTrue(hand.detect_open_palm(pose))
        self.assertFalse(hand.detect_closed_palm(pose))

    def test_registry_and_rule_candidates_surface_closed_palm(self):
        snapshot = GestureRegistry().detect(_flat_closed_palm_hand())

        self.assertTrue(snapshot.closed_palm)
        self.assertFalse(snapshot.open_palm)
        self.assertIn("CLOSED_PALM", snapshot_to_candidates(snapshot))
        self.assertNotIn("OPEN_PALM", snapshot_to_candidates(snapshot))


if __name__ == "__main__":
    unittest.main()
