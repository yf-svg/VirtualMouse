from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.control.cursor_space import cursor_point_from_landmarks


def _point(x: float, y: float, z: float = 0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _hand_with_index_tip(index_tip_x: float, index_tip_y: float):
    lm = [_point(0.5, 0.5) for _ in range(21)]
    lm[0] = _point(0.50, 0.82)   # wrist
    lm[5] = _point(0.36, 0.58)   # index_mcp
    lm[9] = _point(0.46, 0.54)   # middle_mcp
    lm[13] = _point(0.56, 0.58)  # ring_mcp
    lm[17] = _point(0.66, 0.64)  # pinky_mcp
    lm[8] = _point(index_tip_x, index_tip_y)  # index_tip
    return lm


class Phase6CursorSpaceTests(unittest.TestCase):
    def test_palm_center_anchor_is_stable_against_index_tip_articulation(self):
        open_hand = _hand_with_index_tip(0.36, 0.24)
        pinch_like_hand = _hand_with_index_tip(0.47, 0.49)

        open_point = cursor_point_from_landmarks(open_hand, anchor_mode="palm_center", mirror_x=False)
        pinch_point = cursor_point_from_landmarks(pinch_like_hand, anchor_mode="palm_center", mirror_x=False)

        self.assertIsNotNone(open_point)
        self.assertIsNotNone(pinch_point)
        self.assertAlmostEqual(open_point.x, pinch_point.x, places=6)
        self.assertAlmostEqual(open_point.y, pinch_point.y, places=6)

    def test_cursor_space_can_mirror_x_for_natural_selfie_control(self):
        hand = _hand_with_index_tip(0.36, 0.24)

        raw_point = cursor_point_from_landmarks(hand, anchor_mode="palm_center", mirror_x=False)
        mirrored_point = cursor_point_from_landmarks(hand, anchor_mode="palm_center", mirror_x=True)

        self.assertIsNotNone(raw_point)
        self.assertIsNotNone(mirrored_point)
        self.assertAlmostEqual(mirrored_point.x, 1.0 - raw_point.x, places=6)
        self.assertAlmostEqual(mirrored_point.y, raw_point.y, places=6)


if __name__ == "__main__":
    unittest.main()
