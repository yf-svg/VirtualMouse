from __future__ import annotations

import unittest

from app.gestures.features import (
    extract_geometric_features,
    geometric_angle_names,
    geometric_distance_names,
)
from tests.test_phase3_normalization import _make_landmarks


class GeometricFeatureTests(unittest.TestCase):
    def assertTupleAlmostEqual(self, left: tuple[float, ...], right: tuple[float, ...], places: int = 5) -> None:
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertAlmostEqual(a, b, places=places)

    def test_geometric_feature_contract_dimension_is_stable(self):
        geometry = extract_geometric_features(_make_landmarks())
        self.assertEqual(len(geometry.angle_names), 10)
        self.assertEqual(len(geometry.distance_names), 19)
        self.assertEqual(len(geometry.angle_values), 10)
        self.assertEqual(len(geometry.distance_values), 19)
        self.assertEqual(geometry.angle_names, geometric_angle_names())
        self.assertEqual(geometry.distance_names, geometric_distance_names())

    def test_angle_features_are_translation_and_scale_invariant(self):
        base = extract_geometric_features(_make_landmarks())
        shifted_scaled = extract_geometric_features(_make_landmarks(scale=2.5, tx=0.4, ty=-0.2, tz=0.1))
        self.assertTupleAlmostEqual(base.angle_values, shifted_scaled.angle_values)

    def test_normalized_distance_features_are_translation_and_scale_invariant(self):
        base = extract_geometric_features(_make_landmarks())
        shifted_scaled = extract_geometric_features(_make_landmarks(scale=2.5, tx=0.4, ty=-0.2, tz=0.1))
        self.assertTupleAlmostEqual(base.distance_values, shifted_scaled.distance_values)


if __name__ == "__main__":
    unittest.main()
