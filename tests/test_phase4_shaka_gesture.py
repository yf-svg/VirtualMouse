from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gestures.hand_gestures import HandGestures, detect_shaka
from app.gestures.registry import GestureRegistry
from app.gestures.rules import snapshot_to_candidates


def _make_points(coords: list[tuple[float, float, float]]):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in coords]


def _shaka_hand():
    return _make_points(
        [
            (0.00, 0.00, 0.00),
            (-0.05, -0.03, 0.00), (-0.10, -0.05, 0.00), (-0.18, -0.08, 0.00), (-0.28, -0.10, 0.00),
            (0.05, -0.03, 0.00), (0.06, 0.00, 0.00), (0.07, 0.03, 0.00), (0.08, 0.06, 0.00),
            (0.08, -0.03, 0.00), (0.09, 0.01, 0.00), (0.10, 0.04, 0.00), (0.11, 0.07, 0.00),
            (0.12, -0.03, 0.00), (0.13, 0.00, 0.00), (0.14, 0.03, 0.00), (0.15, 0.06, 0.00),
            (0.17, -0.03, 0.00), (0.22, -0.07, 0.00), (0.28, -0.10, 0.00), (0.35, -0.13, 0.00),
        ]
    )


def _shaka_with_index_extended():
    hand = _shaka_hand()
    hand[6].x, hand[6].y = 0.06, -0.07
    hand[7].x, hand[7].y = 0.08, -0.12
    hand[8].x, hand[8].y = 0.10, -0.18
    return hand


def _back_facing_shaka_hand():
    hand = _shaka_hand()
    for idx in (8, 12, 16, 20):
        hand[idx].z = 0.08
    hand[4].z = -0.06
    return hand


class ShakaGestureTests(unittest.TestCase):
    def test_detects_canonical_shaka_and_suppresses_number_one(self):
        hand = HandGestures()
        pose = _shaka_hand()
        self.assertTrue(hand.detect_shaka(pose))
        self.assertTrue(detect_shaka(pose))
        self.assertIsNone(hand.detect_numbers_1_to_5(pose))

    def test_shaka_rejects_when_index_is_extended(self):
        self.assertFalse(detect_shaka(_shaka_with_index_extended()))

    def test_detects_shaka_when_back_of_hand_faces_camera(self):
        self.assertTrue(detect_shaka(_back_facing_shaka_hand()))

    def test_registry_and_rule_candidates_surface_shaka(self):
        snapshot = GestureRegistry().detect(_shaka_hand())
        self.assertTrue(snapshot.shaka)
        self.assertFalse(snapshot.peace_sign)
        self.assertIsNone(snapshot.number)
        self.assertIn("SHAKA", snapshot_to_candidates(snapshot))


if __name__ == "__main__":
    unittest.main()
