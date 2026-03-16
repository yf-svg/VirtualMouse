from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.gestures.features import (
    extract_normalized_hand_landmarks,
    normalized_landmark_names,
)


def _make_landmarks(scale: float = 1.0, tx: float = 0.0, ty: float = 0.0, tz: float = 0.0):
    """
    Synthetic hand-like landmark pattern with non-zero palm span so the scale
    normalization is meaningful.
    """
    base = [
        (0.00, 0.00, 0.00),   # wrist
        (-0.10, -0.04, 0.00), (-0.12, -0.12, 0.00), (-0.13, -0.21, 0.00), (-0.14, -0.30, 0.00),
        (-0.05, -0.03, 0.00), (-0.04, -0.16, 0.00), (-0.03, -0.27, 0.00), (-0.02, -0.38, 0.00),
        (0.00, -0.03, 0.00),  (0.00, -0.18, 0.00),  (0.00, -0.31, 0.00),  (0.00, -0.44, 0.00),
        (0.05, -0.03, 0.00),  (0.04, -0.16, 0.00),  (0.03, -0.28, 0.00),  (0.02, -0.40, 0.00),
        (0.10, -0.02, 0.00),  (0.10, -0.13, 0.00),  (0.09, -0.23, 0.00),  (0.08, -0.33, 0.00),
    ]
    return [
        SimpleNamespace(
            x=(x * scale) + tx,
            y=(y * scale) + ty,
            z=(z * scale) + tz,
        )
        for x, y, z in base
    ]


class NormalizationTests(unittest.TestCase):
    def assertTupleAlmostEqual(self, left: tuple[float, ...], right: tuple[float, ...], places: int = 9) -> None:
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertAlmostEqual(a, b, places=places)

    def test_normalized_landmark_contract_dimension_is_stable(self):
        normalized = extract_normalized_hand_landmarks(_make_landmarks())
        self.assertEqual(normalized.dimension, 63)
        self.assertEqual(len(normalized_landmark_names()), 63)

    def test_normalization_is_translation_invariant(self):
        base = extract_normalized_hand_landmarks(_make_landmarks())
        shifted = extract_normalized_hand_landmarks(_make_landmarks(tx=0.9, ty=-0.7, tz=0.2))
        self.assertTupleAlmostEqual(base.values, shifted.values)

    def test_normalization_is_scale_invariant(self):
        base = extract_normalized_hand_landmarks(_make_landmarks())
        scaled = extract_normalized_hand_landmarks(_make_landmarks(scale=2.75, tx=0.3, ty=-0.1))
        self.assertTupleAlmostEqual(base.values, scaled.values)


if __name__ == "__main__":
    unittest.main()
