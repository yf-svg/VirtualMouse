from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib

from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION, feature_schema
from app.gestures.training import TRAINER_VERSION, TrainingPolicy, train_svm_from_validated_dataset


def _feature_vector(seed: float = 0.0) -> list[float]:
    values: list[float] = []
    for idx, name in enumerate(feature_schema().names):
        if name.endswith(("_x_norm", "_y_norm", "_z_norm")):
            values.append(round(seed + ((idx % 7) - 3) * 0.03, 6))
        elif name.endswith("_angle"):
            values.append(110.0 + (idx % 5))
        elif name in {"extended_count", "curled_count", "near_palm_count"}:
            values.append(float((idx % 3) + 1))
        else:
            values.append(0.75 + (idx % 4) * 0.08 + seed)
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
            "names": list(feature_schema().names),
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


def _validated_dataset_payload_with_grouped_train_users() -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for split, user_id, offset in (
        ("train", "U01", 0.00),
        ("train", "U02", 0.05),
        ("train", "U03", 0.10),
        ("validation", "U04", 0.15),
        ("test", "U05", 0.20),
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
            "names": list(feature_schema().names),
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
            "distinct_users": ["U01", "U02", "U03", "U04", "U05"],
            "samples_by_label": {"FIST": 10, "BRAVO": 10},
            "samples_by_user": {"U01": 4, "U02": 4, "U03": 4, "U04": 4, "U05": 4},
            "samples_by_split": {"train": 12, "validation": 4, "test": 4},
            "split_status": "ok",
        },
    }


def _validated_dataset_payload_missing_train_label() -> dict[str, object]:
    payload = _validated_dataset_payload_with_grouped_train_users()
    payload["validated_samples"].append(
        _validated_sample(
            gesture_label="OPEN_PALM",
            user_id="U04",
            split="validation",
            sample_index=10,
            seed=0.33,
        )
    )
    payload["validated_samples"].append(
        _validated_sample(
            gesture_label="OPEN_PALM",
            user_id="U05",
            split="test",
            sample_index=11,
            seed=0.36,
        )
    )
    payload["summary"]["validated_sample_count"] = len(payload["validated_samples"])
    payload["summary"]["distinct_labels"] = ["BRAVO", "FIST", "OPEN_PALM"]
    payload["summary"]["samples_by_label"] = {"FIST": 10, "BRAVO": 10, "OPEN_PALM": 2}
    return payload


def _validated_dataset_payload_with_train_cv_unavailable() -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for split, user_id, offset, include_bravo in (
        ("train", "U01", 0.00, True),
        ("train", "U02", 0.05, False),
        ("validation", "U03", 0.10, True),
        ("test", "U04", 0.15, True),
    ):
        samples.extend(
            [
                _validated_sample(gesture_label="FIST", user_id=user_id, split=split, sample_index=0, seed=0.01 + offset),
                _validated_sample(gesture_label="FIST", user_id=user_id, split=split, sample_index=1, seed=0.02 + offset),
            ]
        )
        if include_bravo:
            samples.extend(
                [
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
            "names": list(feature_schema().names),
        },
        "validated_samples": samples,
        "rejected_samples": [],
        "rejected_sessions": [],
        "split_plan": {"status": "ok", "assignments": {}, "issues": [], "cv_strategy": None, "cv_n_splits": None},
        "summary": {
            "validated_sample_count": len(samples),
            "rejected_sample_count": 0,
            "rejected_session_count": 0,
            "distinct_labels": ["BRAVO", "FIST"],
            "distinct_users": ["U01", "U02", "U03", "U04"],
            "samples_by_label": {"FIST": 8, "BRAVO": 6},
            "samples_by_user": {"U01": 4, "U02": 2, "U03": 4, "U04": 4},
            "samples_by_split": {"train": 6, "validation": 4, "test": 4},
            "split_status": "ok",
        },
    }


class Phase42TrainingPipelineTests(unittest.TestCase):
    def test_training_pipeline_fits_candidate_from_validated_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "validated_dataset.json"
            model_path = Path(tmpdir) / "svm_candidate.joblib"
            report_path = Path(tmpdir) / "training_report.json"
            dataset_path.write_text(json.dumps(_validated_dataset_payload()), encoding="utf-8")

            result = train_svm_from_validated_dataset(
                dataset_path,
                output_model_path=model_path,
                output_report_path=report_path,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            model_payload = joblib.load(model_path)

        self.assertEqual(result.trainer_version, TRAINER_VERSION)
        self.assertEqual(report["trainer_version"], TRAINER_VERSION)
        self.assertEqual(report["split_status"], "ok")
        self.assertEqual(set(report["labels"]), {"BRAVO", "FIST"})
        self.assertEqual(model_payload["artifact_kind"], "training_candidate")
        self.assertEqual(model_payload["schema_version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(model_payload["feature_dimension"], FEATURE_DIMENSION)
        self.assertEqual(result.model_selection_status, "search_skipped_insufficient_train_users")
        self.assertEqual(result.model_selection_reason, "train_split_has_fewer_than_two_distinct_users")
        self.assertEqual(report["model_selection_status"], "search_skipped_insufficient_train_users")
        self.assertEqual(report["model_selection_reason"], "train_split_has_fewer_than_two_distinct_users")
        self.assertIn("search_param_grid", report)
        self.assertIn("training_policy", report)
        self.assertEqual(report["metrics"]["validation"]["labels"], ["BRAVO", "FIST"])
        self.assertEqual(report["metrics"]["validation"]["per_label"]["BRAVO"]["support"], 2)
        self.assertIn("FIST", report["metrics"]["validation"]["confusion_matrix"]["BRAVO"])

    def test_training_pipeline_rejects_dataset_without_ok_split_status(self):
        payload = _validated_dataset_payload()
        payload["summary"]["split_status"] = "insufficient_users_for_disjoint_split"
        payload["split_plan"]["status"] = "insufficient_users_for_disjoint_split"

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "validated_dataset.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                train_svm_from_validated_dataset(
                    dataset_path,
                    output_model_path=Path(tmpdir) / "candidate.joblib",
                    output_report_path=Path(tmpdir) / "report.json",
                )

        self.assertIn("split_status", str(ctx.exception))

    def test_training_pipeline_rejects_dataset_with_missing_train_label_coverage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "validated_dataset.json"
            dataset_path.write_text(json.dumps(_validated_dataset_payload_missing_train_label()), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                train_svm_from_validated_dataset(
                    dataset_path,
                    output_model_path=Path(tmpdir) / "candidate.joblib",
                    output_report_path=Path(tmpdir) / "report.json",
                )

        self.assertIn("missing", str(ctx.exception).lower())
        self.assertIn("OPEN_PALM", str(ctx.exception))

    def test_training_pipeline_runs_grouped_search_when_train_has_multiple_users(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "validated_dataset.json"
            model_path = Path(tmpdir) / "svm_candidate.joblib"
            report_path = Path(tmpdir) / "training_report.json"
            dataset_path.write_text(json.dumps(_validated_dataset_payload_with_grouped_train_users()), encoding="utf-8")

            result = train_svm_from_validated_dataset(
                dataset_path,
                output_model_path=model_path,
                output_report_path=report_path,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))
            model_payload = joblib.load(model_path)

        self.assertEqual(result.model_selection_status, "search_ok")
        self.assertEqual(report["model_selection_status"], "search_ok")
        self.assertEqual(result.model_selection_reason, "grid_search_completed")
        self.assertEqual(result.cv_strategy, "StratifiedGroupKFold")
        self.assertEqual(report["cv_n_splits"], 3)
        self.assertIn("C", result.best_params)
        self.assertIn("gamma", result.best_params)
        self.assertGreaterEqual(float(result.cv_best_score or 0.0), 0.0)
        self.assertEqual(model_payload["model_selection_status"], "search_ok")
        self.assertEqual(report["metrics"]["train"]["per_label"]["BRAVO"]["support"], 6)
        self.assertIn("BRAVO", report["metrics"]["test"]["confusion_matrix"])

    def test_training_pipeline_skips_search_when_group_cv_cannot_preserve_train_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "validated_dataset.json"
            model_path = Path(tmpdir) / "svm_candidate.joblib"
            report_path = Path(tmpdir) / "training_report.json"
            dataset_path.write_text(json.dumps(_validated_dataset_payload_with_train_cv_unavailable()), encoding="utf-8")

            result = train_svm_from_validated_dataset(
                dataset_path,
                output_model_path=model_path,
                output_report_path=report_path,
            )

            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.model_selection_status, "search_skipped_cv_unavailable")
        self.assertIn("train_label_coverage", result.model_selection_reason or "")
        self.assertEqual(report["model_selection_status"], "search_skipped_cv_unavailable")

    def test_training_pipeline_rejects_invalid_search_refit_metric(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "validated_dataset.json"
            dataset_path.write_text(json.dumps(_validated_dataset_payload_with_grouped_train_users()), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                train_svm_from_validated_dataset(
                    dataset_path,
                    output_model_path=Path(tmpdir) / "candidate.joblib",
                    output_report_path=Path(tmpdir) / "report.json",
                    policy=TrainingPolicy(search_refit_metric="roc_auc"),
                )

        self.assertIn("search_refit_metric", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
