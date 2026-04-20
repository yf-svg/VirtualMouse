from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gestures.hand_gestures import HandGestures
from app.gestures.pointing import PointingDetector, PointingCfg, detect_point_left, detect_point_right
from app.gestures.registry import GestureRegistry, GestureSnapshot
from app.gestures.rules import snapshot_to_candidates


def _make_points(coords: list[tuple[float, float, float]]):
    return [SimpleNamespace(x=x, y=y, z=z) for x, y, z in coords]


def _wrap_hand(landmarks, handedness: str):
    return SimpleNamespace(landmarks=landmarks, handedness=handedness)


def _display_point_right_hand():
    # Mirror-view semantics: user sees POINT_RIGHT, raw landmarks point left.
    landmarks = _make_points(
        [
            (0.00, 0.00, 0.02),   # wrist
            (-0.09, -0.01, 0.01), (-0.07, 0.02, 0.00), (-0.04, 0.04, -0.01), (-0.02, 0.05, -0.02),   # thumb compact, left of palm => palm-facing right hand
            (0.05, -0.03, 0.01), (-0.06, -0.02, 0.00), (-0.19, -0.01, -0.01), (-0.34, 0.00, -0.02),  # raw left
            (0.00, -0.03, 0.02), (-0.05, 0.02, 0.02), (-0.04, 0.04, 0.00), (-0.02, 0.05, -0.03),     # visible curled fingers
            (-0.05, -0.03, 0.02), (-0.08, 0.02, 0.02), (-0.07, 0.04, 0.00), (-0.05, 0.05, -0.03),
            (-0.10, -0.02, 0.02), (-0.11, 0.01, 0.02), (-0.10, 0.03, 0.00), (-0.09, 0.04, -0.03),
        ]
    )
    return _wrap_hand(landmarks, "Right")


def _display_point_left_hand():
    # Mirror-view semantics: user sees POINT_LEFT, raw landmarks point right.
    landmarks = _make_points(
        [
            (0.00, 0.00, 0.02),   # wrist
            (0.10, -0.01, 0.02), (0.08, 0.02, 0.03), (0.04, 0.04, 0.04), (0.02, 0.05, 0.05),         # thumb compact, right of palm => back-facing right hand
            (-0.05, -0.03, 0.01), (0.06, -0.02, 0.00), (0.19, -0.01, -0.01), (0.34, 0.00, -0.02),    # raw right
            (0.00, -0.03, 0.02), (0.05, 0.02, 0.01), (0.04, 0.04, 0.03), (0.02, 0.05, 0.06),         # hidden curled fingers
            (0.05, -0.03, 0.02), (0.08, 0.02, 0.01), (0.07, 0.04, 0.03), (0.05, 0.05, 0.06),
            (0.10, -0.02, 0.02), (0.11, 0.01, 0.01), (0.10, 0.03, 0.03), (0.09, 0.04, 0.06),
        ]
    )
    return _wrap_hand(landmarks, "Right")


def _display_point_right_with_l_thumb():
    landmarks = _make_points(
        [
            (0.00, 0.00, 0.02),
            (-0.10, -0.04, 0.01), (-0.14, -0.12, 0.00), (-0.20, -0.22, -0.02), (-0.25, -0.32, -0.03),  # thumb too extended
            (0.05, -0.03, 0.01), (-0.06, -0.02, 0.00), (-0.19, -0.01, -0.01), (-0.34, 0.00, -0.02),
            (0.00, -0.03, 0.02), (-0.05, 0.02, 0.02), (-0.04, 0.04, 0.00), (-0.02, 0.05, -0.03),
            (-0.05, -0.03, 0.02), (-0.08, 0.02, 0.02), (-0.07, 0.04, 0.00), (-0.05, 0.05, -0.03),
            (-0.10, -0.02, 0.02), (-0.11, 0.01, 0.02), (-0.10, 0.03, 0.00), (-0.09, 0.04, -0.03),
        ]
    )
    return _wrap_hand(landmarks, "Right")


def _display_point_right_back_facing():
    hand = _display_point_right_hand()
    for tip_idx in (12, 16, 20):
        pip_idx = tip_idx - 2
        hand.landmarks[tip_idx].z = hand.landmarks[pip_idx].z + 0.08
    return hand


def _clean_l_hand():
    return _make_points(
        [
            (0.50, 0.70, 0.00),
            (0.44, 0.66, 0.00), (0.38, 0.66, 0.00), (0.30, 0.66, 0.00), (0.20, 0.66, 0.00),
            (0.56, 0.56, 0.00), (0.56, 0.45, 0.00), (0.56, 0.33, 0.00), (0.56, 0.20, 0.00),
            (0.60, 0.58, 0.00), (0.59, 0.64, 0.00), (0.58, 0.68, 0.00), (0.57, 0.71, 0.00),
            (0.64, 0.60, 0.00), (0.63, 0.66, 0.00), (0.62, 0.70, 0.00), (0.61, 0.73, 0.00),
            (0.68, 0.62, 0.00), (0.67, 0.67, 0.00), (0.66, 0.71, 0.00), (0.65, 0.74, 0.00),
        ]
    )


def _clean_one_hand():
    return _make_points(
        [
            (0.50, 0.70, 0.00),
            (0.52, 0.69, 0.00), (0.54, 0.70, 0.00), (0.56, 0.71, 0.00), (0.60, 0.68, 0.00),
            (0.56, 0.56, 0.00), (0.56, 0.45, 0.00), (0.56, 0.33, 0.00), (0.56, 0.20, 0.00),
            (0.60, 0.58, 0.00), (0.59, 0.64, 0.00), (0.58, 0.68, 0.00), (0.57, 0.71, 0.00),
            (0.64, 0.60, 0.00), (0.63, 0.66, 0.00), (0.62, 0.70, 0.00), (0.61, 0.73, 0.00),
            (0.68, 0.62, 0.00), (0.67, 0.67, 0.00), (0.66, 0.71, 0.00), (0.65, 0.74, 0.00),
        ]
    )


def _horizontal_one_like_hand():
    return _make_points(
        [
            (0.50, 0.70, 0.00),
            (0.47, 0.68, 0.00), (0.48, 0.66, 0.00), (0.49, 0.64, 0.00), (0.50, 0.62, 0.00),
            (0.56, 0.50, 0.00), (0.64, 0.46, 0.00), (0.70, 0.40, 0.00), (0.76, 0.44, 0.00),
            (0.60, 0.58, 0.00), (0.59, 0.64, 0.00), (0.58, 0.68, 0.00), (0.57, 0.71, 0.00),
            (0.64, 0.60, 0.00), (0.63, 0.66, 0.00), (0.62, 0.70, 0.00), (0.61, 0.73, 0.00),
            (0.68, 0.62, 0.00), (0.67, 0.67, 0.00), (0.66, 0.71, 0.00), (0.65, 0.74, 0.00),
        ]
    )


def _side_profile_one_hand():
    return _make_points(
        [
            (0.50, 0.70, 0.00),
            (0.52, 0.69, 0.00), (0.54, 0.70, 0.00), (0.56, 0.71, 0.00), (0.60, 0.68, 0.00),
            (0.56, 0.56, 0.00), (0.63, 0.50, 0.00), (0.70, 0.45, 0.00), (0.76, 0.40, 0.00),
            (0.60, 0.58, 0.00), (0.59, 0.64, 0.00), (0.58, 0.68, 0.00), (0.57, 0.71, 0.00),
            (0.64, 0.60, 0.00), (0.63, 0.66, 0.00), (0.62, 0.70, 0.00), (0.61, 0.73, 0.00),
            (0.68, 0.62, 0.00), (0.67, 0.67, 0.00), (0.66, 0.71, 0.00), (0.65, 0.74, 0.00),
        ]
    )


def _clean_three_imr_hand():
    return _make_points(
        [
            (0.50, 0.70, 0.00),
            (0.50, 0.68, 0.00), (0.51, 0.67, 0.00), (0.52, 0.66, 0.00), (0.53, 0.65, 0.00),
            (0.56, 0.56, 0.00), (0.56, 0.45, 0.00), (0.56, 0.33, 0.00), (0.56, 0.20, 0.00),
            (0.60, 0.58, 0.00), (0.60, 0.47, 0.00), (0.60, 0.35, 0.00), (0.60, 0.22, 0.00),
            (0.64, 0.60, 0.00), (0.64, 0.49, 0.00), (0.64, 0.37, 0.00), (0.64, 0.25, 0.00),
            (0.68, 0.62, 0.00), (0.67, 0.67, 0.00), (0.66, 0.71, 0.00), (0.65, 0.74, 0.00),
        ]
    )


def _clean_three_imp_hand():
    return _make_points(
        [
            (0.50, 0.70, 0.00),
            (0.50, 0.68, 0.00), (0.51, 0.67, 0.00), (0.52, 0.66, 0.00), (0.53, 0.65, 0.00),
            (0.56, 0.56, 0.00), (0.56, 0.45, 0.00), (0.56, 0.33, 0.00), (0.56, 0.20, 0.00),
            (0.60, 0.58, 0.00), (0.60, 0.47, 0.00), (0.60, 0.35, 0.00), (0.60, 0.22, 0.00),
            (0.64, 0.60, 0.00), (0.63, 0.66, 0.00), (0.62, 0.70, 0.00), (0.61, 0.73, 0.00),
            (0.68, 0.62, 0.00), (0.68, 0.51, 0.00), (0.68, 0.39, 0.00), (0.68, 0.27, 0.00),
        ]
    )


def _three_mrp_hand():
    return _make_points(
        [
            (0.50, 0.70, 0.00),
            (0.50, 0.68, 0.00), (0.51, 0.67, 0.00), (0.52, 0.66, 0.00), (0.53, 0.65, 0.00),
            (0.56, 0.58, 0.00), (0.57, 0.64, 0.00), (0.58, 0.68, 0.00), (0.59, 0.71, 0.00),
            (0.60, 0.58, 0.00), (0.60, 0.47, 0.00), (0.60, 0.35, 0.00), (0.60, 0.22, 0.00),
            (0.64, 0.60, 0.00), (0.64, 0.49, 0.00), (0.64, 0.37, 0.00), (0.64, 0.25, 0.00),
            (0.68, 0.62, 0.00), (0.68, 0.51, 0.00), (0.68, 0.39, 0.00), (0.68, 0.27, 0.00),
        ]
    )


class PointingGestureTests(unittest.TestCase):
    def test_detect_point_right_uses_mirrored_display_semantics(self):
        hand = _display_point_right_hand()
        self.assertTrue(detect_point_right(hand))
        self.assertFalse(detect_point_left(hand))

    def test_detect_point_left_uses_mirrored_display_semantics(self):
        hand = _display_point_left_hand()
        self.assertTrue(detect_point_left(hand))
        self.assertFalse(detect_point_right(hand))

    def test_pointing_pose_does_not_read_as_number_one(self):
        hand = HandGestures()
        self.assertIsNone(hand.detect_numbers_1_to_5(_display_point_right_hand()))
        self.assertIsNone(hand.detect_numbers_1_to_5(_display_point_left_hand()))

    def test_pointing_pose_requires_compact_thumb_to_avoid_l_overlap(self):
        self.assertFalse(detect_point_right(_display_point_right_with_l_thumb()))

    def test_l_pose_does_not_read_as_number_one(self):
        hand = HandGestures()
        pose = _clean_l_hand()
        self.assertTrue(hand.detect_L(pose))
        self.assertIsNone(hand.detect_numbers_1_to_5(pose))

    def test_compact_thumb_one_still_reads_as_one(self):
        hand = HandGestures()
        pose = _clean_one_hand()
        self.assertFalse(hand.detect_L(pose))
        self.assertEqual(hand.detect_numbers_1_to_5(pose), "ONE")

    def test_horizontal_index_pose_does_not_read_as_number_one(self):
        hand = HandGestures()
        pose = _horizontal_one_like_hand()
        self.assertIsNone(hand.detect_numbers_1_to_5(pose))

    def test_side_profile_one_still_reads_as_one(self):
        hand = HandGestures()
        pose = _side_profile_one_hand()
        self.assertFalse(hand.detect_L(pose))
        self.assertEqual(hand.detect_numbers_1_to_5(pose), "ONE")

    def test_three_accepts_index_middle_ring_combo(self):
        hand = HandGestures()
        self.assertEqual(hand.detect_numbers_1_to_5(_clean_three_imr_hand()), "THREE")

    def test_three_accepts_index_middle_pinky_combo(self):
        hand = HandGestures()
        self.assertEqual(hand.detect_numbers_1_to_5(_clean_three_imp_hand()), "THREE")

    def test_three_rejects_middle_ring_pinky_combo(self):
        hand = HandGestures()
        self.assertIsNone(hand.detect_numbers_1_to_5(_three_mrp_hand()))

    def test_registry_surfaces_l_without_one_overlap(self):
        snapshot = GestureRegistry().detect(_clean_l_hand())
        self.assertTrue(snapshot.l_gesture)
        self.assertIsNone(snapshot.number)
        self.assertIn("L", snapshot_to_candidates(snapshot))
        self.assertNotIn("ONE", snapshot_to_candidates(snapshot))

    def test_unclear_orientation_can_pass_when_pointer_shape_is_clean(self):
        hand = _display_point_right_hand()
        for point in hand.landmarks:
            point.z = 0.0
        analysis = PointingDetector().analyze(hand, direction="right")
        self.assertTrue(analysis.matched)
        self.assertEqual(analysis.reason, "ok_shape")

    def test_point_right_allows_opposite_depth_orientation_when_shape_is_clean(self):
        hand = _display_point_right_back_facing()
        analysis = PointingDetector().analyze(hand, direction="right")
        self.assertTrue(analysis.matched)
        self.assertEqual(analysis.orientation, "back")
        self.assertEqual(analysis.reason, "ok_shape")

    def test_candidate_mapping_includes_point_labels(self):
        snapshot = GestureSnapshot(
            pinch=None,
            fist=False,
            closed_palm=False,
            open_palm=False,
            shaka=False,
            peace_sign=False,
            number=None,
            l_gesture=False,
            bravo=False,
            thumbs_down=False,
            point_right=True,
            point_left=False,
        )
        self.assertEqual(snapshot_to_candidates(snapshot), {"POINT_RIGHT"})


if __name__ == "__main__":
    unittest.main()
