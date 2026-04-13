from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib

from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION
from app.gestures.model_bundle import (
    CANDIDATE_ARTIFACT_KIND,
    RUNTIME_ARTIFACT_KIND,
    RUNTIME_BUNDLE_VERSION,
    promote_runtime_model_bundle,
)


class _FakeProbModel:
    def __init__(self):
        self.classes_ = ["BRAVO", "FIST"]

    def predict_proba(self, rows):
        return [[0.9, 0.1] for _ in rows]


def _runtime_bundle() -> dict[str, object]:
    return {
        "artifact_kind": RUNTIME_ARTIFACT_KIND,
        "bundle_version": RUNTIME_BUNDLE_VERSION,
        "trainer_version": "phase4.2.v1",
        "trained_at": "2026-04-13T00:00:00",
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dimension": FEATURE_DIMENSION,
        "labels": ["BRAVO", "FIST"],
        "min_confidence": 0.70,
        "model": _FakeProbModel(),
    }


class Phase44RuntimeModelPromotionTests(unittest.TestCase):
    def test_promotion_rejects_non_runtime_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "candidate.joblib"
            joblib.dump(
                {
                    "artifact_kind": CANDIDATE_ARTIFACT_KIND,
                    "schema_version": FEATURE_SCHEMA_VERSION,
                    "feature_dimension": FEATURE_DIMENSION,
                    "labels": ["BRAVO", "FIST"],
                    "model": _FakeProbModel(),
                },
                source_path,
            )

            with self.assertRaises(ValueError) as ctx:
                promote_runtime_model_bundle(source_path, live_model_path=Path(tmpdir) / "gesture_svm.joblib")

        self.assertIn("runtime_model_bundle", str(ctx.exception))

    def test_promotion_copies_bundle_and_backs_up_previous_live_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "exported_bundle.joblib"
            live_path = root / "gesture_svm.joblib"
            backup_dir = root / "archive"

            joblib.dump(_runtime_bundle(), source_path)
            joblib.dump(_runtime_bundle() | {"trained_at": "2026-04-12T00:00:00"}, live_path)

            result = promote_runtime_model_bundle(
                source_path,
                live_model_path=live_path,
                backup_dir=backup_dir,
            )

            promoted = joblib.load(live_path)
            backups = list(backup_dir.glob("gesture_svm_*.joblib"))

        self.assertTrue(result.replaced_existing)
        self.assertIsNotNone(result.backup_model_path)
        self.assertEqual(promoted["trained_at"], "2026-04-13T00:00:00")
        self.assertEqual(len(backups), 1)

    def test_promotion_is_noop_when_source_is_already_live_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            live_path = Path(tmpdir) / "gesture_svm.joblib"
            joblib.dump(_runtime_bundle(), live_path)

            result = promote_runtime_model_bundle(live_path, live_model_path=live_path)

        self.assertTrue(result.replaced_existing)
        self.assertIsNone(result.backup_model_path)


if __name__ == "__main__":
    unittest.main()
