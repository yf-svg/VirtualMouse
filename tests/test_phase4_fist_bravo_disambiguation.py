from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gestures.bravo import detect_bravo
from app.gestures.fist import detect_fist
from app.gestures.hand_gestures import detect_bravo as legacy_detect_bravo
from app.gestures.registry import GestureRegistry


def _make_points(coords: list[tuple[float, float, float]]):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in coords]


def _clear_bravo_hand():
    return _make_points(
        [
            (0.50, 0.62, 0.00),
            (0.46, 0.58, 0.00), (0.46, 0.48, 0.00), (0.46, 0.34, 0.00), (0.46, 0.14, 0.00),
            (0.58, 0.50, 0.00), (0.57, 0.54, 0.00), (0.56, 0.57, 0.00), (0.55, 0.61, 0.00),
            (0.54, 0.49, 0.00), (0.53, 0.53, 0.00), (0.52, 0.57, 0.00), (0.51, 0.61, 0.00),
            (0.50, 0.49, 0.00), (0.49, 0.53, 0.00), (0.48, 0.57, 0.00), (0.47, 0.61, 0.00),
            (0.46, 0.50, 0.00), (0.46, 0.53, 0.00), (0.46, 0.56, 0.00), (0.46, 0.59, 0.00),
        ]
    )


def _fist_with_relaxed_thumb():
    return _make_points(
        [
            (0.50, 0.62, 0.00),
            (0.46, 0.56, 0.00), (0.47, 0.52, 0.00), (0.49, 0.48, 0.00), (0.51, 0.44, 0.00),
            (0.58, 0.50, 0.00), (0.57, 0.54, 0.00), (0.56, 0.57, 0.00), (0.55, 0.60, 0.00),
            (0.54, 0.49, 0.00), (0.53, 0.53, 0.00), (0.52, 0.56, 0.00), (0.51, 0.60, 0.00),
            (0.50, 0.49, 0.00), (0.49, 0.53, 0.00), (0.48, 0.56, 0.00), (0.47, 0.60, 0.00),
            (0.46, 0.50, 0.00), (0.46, 0.53, 0.00), (0.46, 0.56, 0.00), (0.46, 0.59, 0.00),
        ]
    )


def _emerging_bravo_hand():
    return _make_points(
        [
            (0.50, 0.62, 0.00),
            (0.50, 0.62, 0.00), (0.46, 0.58, 0.00), (0.46, 0.48, 0.00), (0.46, 0.34, 0.00),
            (0.58, 0.50, 0.00), (0.57, 0.54, 0.00), (0.56, 0.57, 0.00), (0.55, 0.61, 0.00),
            (0.54, 0.49, 0.00), (0.53, 0.53, 0.00), (0.52, 0.57, 0.00), (0.51, 0.61, 0.00),
            (0.50, 0.49, 0.00), (0.49, 0.53, 0.00), (0.48, 0.57, 0.00), (0.47, 0.61, 0.00),
            (0.46, 0.50, 0.00), (0.46, 0.53, 0.00), (0.46, 0.56, 0.00), (0.46, 0.59, 0.00),
        ]
    )


def _side_folded_bravo_hand():
    return _make_points(
        [
            (0.50, 0.62, 0.00),
            (0.48, 0.62, 0.00), (0.47, 0.53, 0.00), (0.46, 0.43, 0.00), (0.46, 0.28, 0.00),
            (0.58, 0.50, 0.00), (0.58, 0.56, 0.00), (0.60, 0.56, 0.00), (0.63, 0.56, 0.00),
            (0.54, 0.49, 0.00), (0.54, 0.55, 0.00), (0.56, 0.55, 0.00), (0.59, 0.55, 0.00),
            (0.50, 0.49, 0.00), (0.50, 0.55, 0.00), (0.52, 0.55, 0.00), (0.55, 0.55, 0.00),
            (0.46, 0.50, 0.00), (0.46, 0.55, 0.00), (0.48, 0.55, 0.00), (0.51, 0.55, 0.00),
        ]
    )


class FistBravoDisambiguationTests(unittest.TestCase):
    def test_clear_bravo_pose_stays_bravo(self):
        hand = _clear_bravo_hand()
        self.assertTrue(detect_bravo(hand))
        self.assertEqual(legacy_detect_bravo(hand), detect_bravo(hand))
        self.assertFalse(detect_fist(hand))

    def test_emerging_bravo_pose_is_detected_early(self):
        hand = _emerging_bravo_hand()
        self.assertTrue(detect_bravo(hand))
        self.assertFalse(detect_fist(hand))

    def test_side_folded_bravo_pose_is_detected(self):
        hand = _side_folded_bravo_hand()
        self.assertTrue(detect_bravo(hand))
        self.assertEqual(legacy_detect_bravo(hand), detect_bravo(hand))
        self.assertFalse(detect_fist(hand))

    def test_fist_with_relaxed_thumb_is_not_bravo(self):
        hand = _fist_with_relaxed_thumb()
        self.assertTrue(detect_fist(hand))
        self.assertFalse(detect_bravo(hand))
        self.assertEqual(legacy_detect_bravo(hand), detect_bravo(hand))

    def test_registry_prefers_fist_for_relaxed_thumb_closure(self):
        snapshot = GestureRegistry().detect(_fist_with_relaxed_thumb())
        self.assertTrue(snapshot.fist)
        self.assertFalse(snapshot.bravo)


if __name__ == "__main__":
    unittest.main()
