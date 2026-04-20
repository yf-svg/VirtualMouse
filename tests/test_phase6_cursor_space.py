from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.config import PresentationToolConfig
from app.control.cursor_space import cursor_point_from_landmarks, presentation_pointer_point_from_landmarks, remap_cursor_point


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

    def test_index_tip_anchor_tracks_fingertip_directly(self):
        hand = _hand_with_index_tip(0.36, 0.24)

        point = cursor_point_from_landmarks(hand, anchor_mode="index_tip", mirror_x=False)

        self.assertIsNotNone(point)
        self.assertAlmostEqual(point.x, 0.36, places=6)
        self.assertAlmostEqual(point.y, 0.24, places=6)

    def test_presentation_pointer_uses_its_dedicated_anchor_mode(self):
        hand = _hand_with_index_tip(0.36, 0.98)

        point = presentation_pointer_point_from_landmarks(
            hand,
            cfg=PresentationToolConfig(
                pointer_anchor_mode="index_tip",
                pointer_input_x_min=0.0,
                pointer_input_x_max=1.0,
                pointer_input_y_min=0.0,
                pointer_input_y_max=1.0,
                pointer_edge_blend_margin=0.0,
                pointer_edge_blend_max_pull=0.0,
            ),
            mirror_x=False,
        )

        self.assertIsNotNone(point)
        self.assertAlmostEqual(point.x, 0.36, places=6)
        self.assertAlmostEqual(point.y, 0.98, places=6)

    def test_presentation_pointer_softly_blends_back_from_edge(self):
        hand = _hand_with_index_tip(0.98, 0.98)

        point = presentation_pointer_point_from_landmarks(
            hand,
            cfg=PresentationToolConfig(
                pointer_anchor_mode="index_tip",
                pointer_edge_blend_margin=0.20,
                pointer_edge_blend_max_pull=0.25,
                pointer_input_x_min=0.0,
                pointer_input_x_max=1.0,
                pointer_input_y_min=0.0,
                pointer_input_y_max=1.0,
            ),
            mirror_x=False,
        )

        self.assertIsNotNone(point)
        self.assertLess(point.x, 0.98)
        self.assertLess(point.y, 0.98)
        self.assertGreater(point.x, 0.80)
        self.assertGreater(point.y, 0.80)

    def test_remap_cursor_point_expands_reachable_bottom_range(self):
        point = remap_cursor_point(
            _point(0.40, 0.75),
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=0.75,
        )

        self.assertAlmostEqual(point.x, 0.40, places=6)
        self.assertAlmostEqual(point.y, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
