from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gestures.features import assess_hand_input_quality, extract_feature_vector
from tests.test_phase3_normalization import _make_landmarks


_PALM_IDS = (0, 5, 9, 13, 17)
_FINGER_GROUPS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _mean_abs_delta(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def _palm_center(landmarks) -> tuple[float, float, float]:
    xs = [landmarks[i].x for i in _PALM_IDS]
    ys = [landmarks[i].y for i in _PALM_IDS]
    zs = [landmarks[i].z for i in _PALM_IDS]
    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(zs) / len(zs),
    )


def _make_user_variant(
    *,
    thumb_scale: float = 1.0,
    index_scale: float = 1.0,
    middle_scale: float = 1.0,
    ring_scale: float = 1.0,
    pinky_scale: float = 1.0,
    spread_scale: float = 1.0,
    base_scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
):
    landmarks = _make_landmarks(scale=base_scale, tx=tx, ty=ty, tz=tz)
    center_x, center_y, center_z = _palm_center(landmarks)
    scale_map = {
        "thumb": thumb_scale,
        "index": index_scale,
        "middle": middle_scale,
        "ring": ring_scale,
        "pinky": pinky_scale,
    }

    out = []
    for idx, point in enumerate(landmarks):
        finger_scale = 1.0
        for finger_name, finger_ids in _FINGER_GROUPS.items():
            if idx in finger_ids:
                finger_scale = scale_map[finger_name]
                break

        dx = (point.x - center_x) * finger_scale
        dy = (point.y - center_y) * finger_scale
        dz = (point.z - center_z) * finger_scale
        if idx in (5, 9, 13, 17):
            dx *= spread_scale

        out.append(
            SimpleNamespace(
                x=center_x + dx,
                y=center_y + dy,
                z=center_z + dz,
            )
        )
    return out


class _WrappedDetection:
    def __init__(self, landmarks, *, background_tag: str, frame_bgr):
        self.landmarks = landmarks
        self.background_tag = background_tag
        self.frame_bgr = frame_bgr


class Phase3ValidationTests(unittest.TestCase):
    def assertTupleNear(
        self,
        left: tuple[float, ...],
        right: tuple[float, ...],
        *,
        delta: float = 1e-5,
    ) -> None:
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertAlmostEqual(a, b, delta=delta)

    def test_feature_vector_is_translation_and_scale_invariant(self):
        base = extract_feature_vector(_make_landmarks())
        shifted_scaled = extract_feature_vector(
            _make_landmarks(scale=2.4, tx=0.6, ty=-0.35, tz=0.15)
        )
        self.assertTupleNear(base.values, shifted_scaled.values)

    def test_feature_extraction_ignores_background_metadata(self):
        plain = _WrappedDetection(
            _make_landmarks(),
            background_tag="plain_wall",
            frame_bgr=[[0, 0], [0, 0]],
        )
        cluttered = {
            "landmarks": _WrappedDetection(
                _make_landmarks(),
                background_tag="busy_room",
                frame_bgr=[[255, 255], [64, 128]],
            )
        }

        plain_vector = extract_feature_vector(plain)
        cluttered_vector = extract_feature_vector(cluttered)
        self.assertTupleNear(plain_vector.values, cluttered_vector.values)

    def test_feature_vector_remains_stable_across_user_like_variants(self):
        base = extract_feature_vector(_make_landmarks()).values
        variants = (
            _make_user_variant(
                thumb_scale=0.98,
                index_scale=1.01,
                middle_scale=1.0,
                ring_scale=0.99,
                pinky_scale=0.97,
                spread_scale=1.01,
                base_scale=1.2,
                tx=0.2,
                ty=-0.1,
            ),
            _make_user_variant(
                thumb_scale=1.03,
                index_scale=0.99,
                middle_scale=1.02,
                ring_scale=1.01,
                pinky_scale=1.04,
                spread_scale=1.03,
                base_scale=0.85,
                tx=-0.5,
                ty=0.3,
            ),
            _make_user_variant(
                thumb_scale=0.96,
                index_scale=1.04,
                middle_scale=1.03,
                ring_scale=0.98,
                pinky_scale=1.01,
                spread_scale=0.98,
                base_scale=1.6,
                tx=0.8,
                ty=0.2,
            ),
        )

        for variant in variants:
            vector = extract_feature_vector(variant)
            self.assertLess(_mean_abs_delta(base, vector.values), 0.06)

    def test_quality_gate_rejects_incomplete_landmarks(self):
        quality = assess_hand_input_quality(_make_landmarks()[:10])
        self.assertFalse(quality.passed)
        self.assertEqual(quality.reason, "incomplete_landmarks")

    def test_quality_gate_rejects_tiny_hands(self):
        quality = assess_hand_input_quality(_make_landmarks(scale=0.03))
        self.assertFalse(quality.passed)
        self.assertEqual(quality.reason, "hand_too_small")


if __name__ == "__main__":
    unittest.main()
