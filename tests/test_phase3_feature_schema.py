from __future__ import annotations

import unittest

from app.gestures.features import (
    FEATURE_DIMENSION,
    FEATURE_SCHEMA,
    extract_feature_vector,
    feature_dimension,
    feature_names,
    feature_schema,
)
from tests.test_phase3_normalization import _make_landmarks


class FeatureSchemaTests(unittest.TestCase):
    def test_feature_schema_has_stable_version(self):
        schema = feature_schema()
        self.assertEqual(schema.version, "phase3.v2")

    def test_feature_schema_matches_feature_names(self):
        schema = feature_schema()
        self.assertEqual(schema.names, feature_names())
        self.assertEqual(schema.dimension, len(feature_names()))

    def test_exported_contract_constants_match_schema(self):
        schema = feature_schema()
        self.assertEqual(FEATURE_SCHEMA, schema)
        self.assertEqual(FEATURE_DIMENSION, schema.dimension)
        self.assertEqual(feature_dimension(), schema.dimension)

    def test_extracted_feature_vector_matches_schema(self):
        vector = extract_feature_vector(_make_landmarks())
        schema = feature_schema()
        self.assertEqual(vector.dimension, schema.dimension)
        self.assertEqual(vector.schema_version, schema.version)


if __name__ == "__main__":
    unittest.main()
