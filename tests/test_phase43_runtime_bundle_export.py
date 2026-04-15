from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib

from app.gestures.classifier import SVMClassifier
from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION, FeatureVector
from app.gestures.model_bundle import CANDIDATE_ARTIFACT_KIND, RUNTIME_ARTIFACT_KIND, RUNTIME_BUNDLE_VERSION
from app.gestures.training import export_runtime_model_bundle, train_svm_from_validated_dataset


def _feature_vector(seed: float = 0.0) -> list[float]:
    values: list[float] = []
    for idx in range(FEATURE_DIMENSION):
        values.append(round(seed + ((idx % 7) - 3) * 0.03, 6))
    return values


def _validated_sample(
    *,
    gesture_label: str,
    user_id: str,
    split: str,
    sample_index: int,
    seed: float,
) -> dict[str, object]:
    return {
        "sample_id": f"{gesture_label}_{user_id}_{sample_index}",
        "gesture_label": gesture_label,
        "user_id": user_id,
        "session_id": f"sess_{gesture_label}_{user_id}",
        "handedness": "Right",
        "capture_context": {"background": "plain", "round": "phase4_v2"},
        "recording_path": f"data/recordings/{gesture_label}_{user_id}.json",
        "sample_index": sample_index,
        "feature_values": _feature_vector(seed=seed),
        "schema_version": FEATURE_SCHEMA_VERSION,
        "quality_reason": "ok",
        "quality_scale": 0.12,
        "quality_palm_width": 0.08,
        "quality_bbox_width": 0.12,
        "quality_bbox_height": 0.18,
        "split": split,
    }


def _validated_dataset_payload() -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for split, user_id, offset in (
        ("train", "U01", 0.00),
        ("validation", "U02", 0.10),
        ("test", "U03", 0.20),
    ):
        samples.extend(
            [
                _validated_sample(gesture_label="FIST", user_id=user_id, split=split, sample_index=0, seed=0.01 + offset),
                _validated_sample(gesture_label="FIST", user_id=user_id, split=split, sample_index=1, seed=0.02 + offset),
                _validated_sample(gesture_label="BRAVO", user_id=user_id, split=split, sample_index=2, seed=0.70 + offset),
                _validated_sample(gesture_label="BRAVO", user_id=user_id, split=split, sample_index=3, seed=0.72 + offset),
            ]
        )

    return {
        "generated_at": "2026-04-13T00:00:00",
        "validator_version": "phase4.1.v1",
        "policy": {},
        "feature_schema": {
            "version": FEATURE_SCHEMA_VERSION,
            "dimension": FEATURE_DIMENSION,
            "names": [f"feature_{idx}" for idx in range(FEATURE_DIMENSION)],
        },
        "validated_samples": samples,
        "rejected_samples": [],
        "rejected_sessions": [],
        "split_plan": {"status": "ok", "assignments": {}, "issues": [], "cv_strategy": "StratifiedGroupKFold", "cv_n_splits": 3},
        "summary": {
            "validated_sample_count": len(samples),
            "rejected_sample_count": 0,
            "rejected_session_count": 0,
            "distinct_labels": ["BRAVO", "FIST"],
            "distinct_users": ["U01", "U02", "U03"],
            "samples_by_label": {"FIST": 6, "BRAVO": 6},
            "samples_by_user": {"U01": 4, "U02": 4, "U03": 4},
            "samples_by_split": {"train": 4, "validation": 4, "test": 4},
            "split_status": "ok",
        },
    }


class Phase43RuntimeBundleExportTests(unittest.TestCase):
    def test_export_runtime_bundle_writes_runtime_approved_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "validated_dataset.json"
            candidate_path = root / "svm_candidate.joblib"
            report_path = root / "training_report.json"
            runtime_path = root / "gesture_svm.joblib"
            dataset_path.write_text(json.dumps(_validated_dataset_payload()), encoding="utf-8")

            train_svm_from_validated_dataset(
                dataset_path,
                output_model_path=candidate_path,
                output_report_path=report_path,
            )
            export = export_runtime_model_bundle(
                candidate_path,
                output_model_path=runtime_path,
                min_confidence=0.78,
                training_report_path=report_path,
            )

            runtime_payload = joblib.load(runtime_path)

        self.assertEqual(export.bundle_version, RUNTIME_BUNDLE_VERSION)
        self.assertEqual(runtime_payload["artifact_kind"], RUNTIME_ARTIFACT_KIND)
        self.assertEqual(runtime_payload["bundle_version"], RUNTIME_BUNDLE_VERSION)
        self.assertEqual(runtime_payload["schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(runtime_payload["feature_dimension"], FEATURE_DIMENSION)
        self.assertEqual(runtime_payload["min_confidence"], 0.78)
        self.assertEqual(set(runtime_payload["labels"]), {"BRAVO", "FIST"})
        self.assertIn("validation", runtime_payload["metrics"])
        self.assertIn("per_label", runtime_payload["metrics"]["validation"])
        self.assertIn("BRAVO", runtime_payload["metrics"]["validation"]["confusion_matrix"])

    def test_classifier_rejects_training_candidate_as_runtime_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "validated_dataset.json"
            candidate_path = root / "svm_candidate.joblib"
            report_path = root / "training_report.json"
            dataset_path.write_text(json.dumps(_validated_dataset_payload()), encoding="utf-8")

            train_svm_from_validated_dataset(
                dataset_path,
                output_model_path=candidate_path,
                output_report_path=report_path,
            )
            candidate_payload = joblib.load(candidate_path)

            classifier = SVMClassifier(model_path=candidate_path, allowed={"BRAVO", "FIST"})
            prediction = classifier.predict(
                FeatureVector(values=tuple(_feature_vector(seed=0.05)), schema_version=FEATURE_SCHEMA_VERSION)
            )

        self.assertEqual(candidate_payload["artifact_kind"], CANDIDATE_ARTIFACT_KIND)
        self.assertFalse(classifier.available)
        self.assertEqual(classifier.load_reason, "artifact_not_runtime_approved")
        self.assertFalse(prediction.accepted)
        self.assertEqual(prediction.reason, "artifact_not_runtime_approved")

    def test_classifier_accepts_exported_runtime_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "validated_dataset.json"
            candidate_path = root / "svm_candidate.joblib"
            report_path = root / "training_report.json"
            runtime_path = root / "gesture_svm.joblib"
            dataset_path.write_text(json.dumps(_validated_dataset_payload()), encoding="utf-8")

            train_svm_from_validated_dataset(
                dataset_path,
                output_model_path=candidate_path,
                output_report_path=report_path,
            )
            export_runtime_model_bundle(
                candidate_path,
                output_model_path=runtime_path,
                min_confidence=0.60,
                training_report_path=report_path,
            )

            classifier = SVMClassifier(model_path=runtime_path, allowed={"BRAVO", "FIST"})
            prediction = classifier.predict(
                FeatureVector(values=tuple(_feature_vector(seed=0.73)), schema_version=FEATURE_SCHEMA_VERSION)
            )

        self.assertTrue(classifier.available)
        self.assertEqual(classifier.load_reason, "ready")
        self.assertTrue(prediction.accepted)
        self.assertIn(prediction.label, {"BRAVO", "FIST"})
        self.assertIsNotNone(prediction.confidence)


if __name__ == "__main__":
    unittest.main()
