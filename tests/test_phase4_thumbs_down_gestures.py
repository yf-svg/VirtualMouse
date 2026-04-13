from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gestures.bravo import detect_bravo
from app.gestures.thumbs_down import detect_thumbs_down


def _make_points(coords: list[tuple[float, float, float]]):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in coords]


def _strict_thumbs_down_hand():
    return _make_points(
        [
            (0.48, 0.54, 0.00),
            (0.56, 0.54, 0.00), (0.61, 0.60, 0.00), (0.66, 0.74, 0.00), (0.71, 0.88, 0.00),
            (0.56, 0.40, 0.00), (0.53, 0.31, 0.00), (0.50, 0.28, 0.00), (0.47, 0.31, 0.00),
            (0.50, 0.39, 0.00), (0.47, 0.31, 0.00), (0.44, 0.28, 0.00), (0.42, 0.31, 0.00),
            (0.45, 0.40, 0.00), (0.42, 0.33, 0.00), (0.39, 0.31, 0.00), (0.38, 0.34, 0.00),
            (0.40, 0.42, 0.00), (0.37, 0.36, 0.00), (0.35, 0.35, 0.00), (0.34, 0.38, 0.00),
        ]
    )


def _variant_thumbs_down_hand():
    hand = _strict_thumbs_down_hand()
    hand[8].x = 0.49
    hand[8].y = 0.28
    hand[12].x = 0.43
    hand[12].y = 0.29
    hand[20].x = 0.33
    hand[20].y = 0.40
    hand[4].x = 0.73
    hand[4].y = 0.86
    return hand


def _thumbs_up_like_hand():
    return _make_points(
        [
            (0.48, 0.54, 0.00),
            (0.56, 0.54, 0.00), (0.61, 0.48, 0.00), (0.66, 0.34, 0.00), (0.71, 0.18, 0.00),
            (0.56, 0.40, 0.00), (0.53, 0.31, 0.00), (0.50, 0.28, 0.00), (0.47, 0.31, 0.00),
            (0.50, 0.39, 0.00), (0.47, 0.31, 0.00), (0.44, 0.28, 0.00), (0.42, 0.31, 0.00),
            (0.45, 0.40, 0.00), (0.42, 0.33, 0.00), (0.39, 0.31, 0.00), (0.38, 0.34, 0.00),
            (0.40, 0.42, 0.00), (0.37, 0.36, 0.00), (0.35, 0.35, 0.00), (0.34, 0.38, 0.00),
        ]
    )


def _thumbs_down_with_index_extended():
    hand = _strict_thumbs_down_hand()
    hand[6].x = 0.54
    hand[6].y = 0.26
    hand[7].x = 0.56
    hand[7].y = 0.18
    hand[8].x = 0.59
    hand[8].y = 0.10
    return hand


class ThumbsDownGestureTests(unittest.TestCase):
    def test_detects_strict_thumbs_down_pose(self):
        hand = _strict_thumbs_down_hand()
        self.assertTrue(detect_thumbs_down(hand))
        self.assertFalse(detect_bravo(hand))

    def test_detects_natural_thumbs_down_variant_with_flatter_finger_stack(self):
        hand = _variant_thumbs_down_hand()
        self.assertTrue(detect_thumbs_down(hand))
        self.assertFalse(detect_bravo(hand))

    def test_rejects_thumbs_up_like_pose(self):
        hand = _thumbs_up_like_hand()
        self.assertFalse(detect_thumbs_down(hand))

    def test_rejects_pose_with_clearly_extended_index(self):
        hand = _thumbs_down_with_index_extended()
        self.assertFalse(detect_thumbs_down(hand))


if __name__ == "__main__":
    unittest.main()
