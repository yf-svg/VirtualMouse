from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION, feature_schema
from app.gestures.validation import (
    ValidationPolicy,
    build_group_cv_strategy,
    instantiate_group_cv,
    save_validated_dataset,
    validate_recording_files,
)


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


def _sample_payload(
    *,
    gesture_label: str,
    user_id: str,
    session_id: str,
    sample_index: int,
    feature_values: list[float] | None = None,
    quality_reason: str = "ok",
    schema_version: str = FEATURE_SCHEMA_VERSION,
) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "frame_seq": sample_index + 1,
        "captured_at": 1000.0 + sample_index,
        "gesture_label": gesture_label,
        "user_id": user_id,
        "session_id": session_id,
        "handedness": "Right",
        "schema_version": schema_version,
        "quality_reason": quality_reason,
        "quality_scale": 0.12,
        "quality_palm_width": 0.08,
        "quality_bbox_width": 0.12,
        "quality_bbox_height": 0.18,
        "feature_values": feature_values or _feature_vector(seed=sample_index * 0.001),
    }


def _session_payload(
    *,
    gesture_label: str,
    user_id: str,
    session_id: str,
    samples: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "gesture_label": gesture_label,
        "user_id": user_id,
        "session_id": session_id,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_dimension": FEATURE_DIMENSION,
        "capture_context": {"background": "plain", "round": "phase4_v1"},
        "created_at": "2026-04-10T22:00:00",
        "sample_count": len(samples),
        "samples": samples,
    }


class Phase41ValidationEngineTests(unittest.TestCase):
    def test_validation_accepts_schema_consistent_recordings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fist_u01.json"
            payload = _session_payload(
                gesture_label="FIST",
                user_id="U01",
                session_id="sess_u01_fist",
                samples=[_sample_payload(gesture_label="FIST", user_id="U01", session_id="sess_u01_fist", sample_index=0)],
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            dataset = validate_recording_files([path], assign_splits=False, apply_outlier_filter=False)

        self.assertEqual(dataset.summary["validated_sample_count"], 1)
        self.assertEqual(dataset.summary["rejected_sample_count"], 0)
        self.assertEqual(dataset.validated_samples[0].gesture_label, "FIST")

    def test_validation_rejects_unknown_labels_and_bad_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            payload = _session_payload(
                gesture_label="ALIEN_LABEL",
                user_id="U01",
                session_id="sess_bad",
                samples=[
                    _sample_payload(
                        gesture_label="ALIEN_LABEL",
                        user_id="U01",
                        session_id="sess_bad",
                        sample_index=0,
                        schema_version="phase3.v1",
                    )
                ],
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            dataset = validate_recording_files([path], assign_splits=False, apply_outlier_filter=False)

        self.assertEqual(dataset.summary["validated_sample_count"], 0)
        self.assertEqual(dataset.summary["rejected_session_count"], 1)
        self.assertIn("unknown_label", dataset.rejected_sessions[0].issues[0].code)

    def test_duplicate_feature_vectors_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_values = _feature_vector(seed=0.1)
            path_a = Path(tmpdir) / "a.json"
            path_b = Path(tmpdir) / "b.json"
            payload_a = _session_payload(
                gesture_label="FIST",
                user_id="U01",
                session_id="sess_a",
                samples=[_sample_payload(gesture_label="FIST", user_id="U01", session_id="sess_a", sample_index=0, feature_values=feature_values)],
            )
            payload_b = _session_payload(
                gesture_label="FIST",
                user_id="U02",
                session_id="sess_b",
                samples=[_sample_payload(gesture_label="FIST", user_id="U02", session_id="sess_b", sample_index=0, feature_values=feature_values)],
            )
            path_a.write_text(json.dumps(payload_a), encoding="utf-8")
            path_b.write_text(json.dumps(payload_b), encoding="utf-8")
            dataset = validate_recording_files([path_a, path_b], assign_splits=False, apply_outlier_filter=False)

        self.assertEqual(dataset.summary["validated_sample_count"], 1)
        self.assertEqual(dataset.summary["rejected_sample_count"], 1)
        self.assertEqual(dataset.rejected_samples[0].issues[0].code, "duplicate_feature_vector")

    def test_outlier_filter_flags_robust_feature_outlier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "open_palm.json"
            samples = [
                _sample_payload(
                    gesture_label="OPEN_PALM",
                    user_id="U01",
                    session_id="sess_open",
                    sample_index=idx,
                    feature_values=_feature_vector(seed=idx * 0.001),
                )
                for idx in range(11)
            ]
            outlier_values = _feature_vector(seed=0.0)
            for idx, name in enumerate(feature_schema().names):
                if name.endswith(("_x_norm", "_y_norm", "_z_norm")):
                    outlier_values[idx] = 4.5
                elif name.endswith("_angle"):
                    outlier_values[idx] = 175.0
                elif name not in {"extended_count", "curled_count", "near_palm_count"}:
                    outlier_values[idx] = 4.8
            samples.append(
                _sample_payload(
                    gesture_label="OPEN_PALM",
                    user_id="U01",
                    session_id="sess_open",
                    sample_index=11,
                    feature_values=outlier_values,
                )
            )
            payload = _session_payload(
                gesture_label="OPEN_PALM",
                user_id="U01",
                session_id="sess_open",
                samples=samples,
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            dataset = validate_recording_files([path], assign_splits=False, apply_outlier_filter=True)

        self.assertEqual(dataset.summary["validated_sample_count"], 11)
        self.assertEqual(dataset.summary["rejected_sample_count"], 1)
        self.assertEqual(dataset.rejected_samples[0].issues[0].code, "feature_outlier")

    def test_split_assignment_keeps_users_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths: list[Path] = []
            for user_offset, user_id in enumerate(("U01", "U02", "U03")):
                for gesture_idx, gesture_label in enumerate(("FIST", "BRAVO", "OPEN_PALM")):
                    path = Path(tmpdir) / f"{gesture_label}_{user_id}.json"
                    payload = _session_payload(
                        gesture_label=gesture_label,
                        user_id=user_id,
                        session_id=f"sess_{gesture_label}_{user_id}",
                        samples=[
                            _sample_payload(
                                gesture_label=gesture_label,
                                user_id=user_id,
                                session_id=f"sess_{gesture_label}_{user_id}",
                                sample_index=idx,
                                feature_values=_feature_vector(
                                    seed=(user_offset * 0.10) + (gesture_idx * 0.03) + (idx + 1) * 0.005
                                ),
                            )
                            for idx in range(2)
                        ],
                    )
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    paths.append(path)

            dataset = validate_recording_files(paths, apply_outlier_filter=False)

        self.assertEqual(dataset.split_plan.status, "ok")
        self.assertEqual(dataset.split_plan.planner, "exhaustive")
        self.assertIsNotNone(dataset.split_plan.assignment_score)
        splits_by_user = {}
        for sample in dataset.validated_samples:
            splits_by_user.setdefault(sample.user_id, set()).add(sample.split)
        self.assertTrue(all(len(splits) == 1 for splits in splits_by_user.values()))
        self.assertEqual({next(iter(s)) for s in splits_by_user.values()}, {"train", "validation", "test"})
        train_labels = {
            sample.gesture_label
            for sample in dataset.validated_samples
            if sample.split == "train"
        }
        self.assertEqual(train_labels, {"FIST", "BRAVO", "OPEN_PALM"})

    def test_split_assignment_uses_beam_search_for_larger_user_sets_and_records_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths: list[Path] = []
            for user_offset, user_id in enumerate(("U01", "U02", "U03", "U04")):
                for gesture_idx, gesture_label in enumerate(("FIST", "BRAVO", "OPEN_PALM")):
                    path = Path(tmpdir) / f"{gesture_label}_{user_id}.json"
                    payload = _session_payload(
                        gesture_label=gesture_label,
                        user_id=user_id,
                        session_id=f"sess_{gesture_label}_{user_id}",
                        samples=[
                            _sample_payload(
                                gesture_label=gesture_label,
                                user_id=user_id,
                                session_id=f"sess_{gesture_label}_{user_id}",
                                sample_index=idx,
                                feature_values=_feature_vector(
                                    seed=(user_offset * 0.08) + (gesture_idx * 0.02) + (idx + 1) * 0.004
                                ),
                            )
                            for idx in range(2)
                        ],
                    )
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    paths.append(path)

            dataset = validate_recording_files(
                paths,
                policy=ValidationPolicy(split_planner_exhaustive_max_users=2, split_planner_beam_width=10),
                apply_outlier_filter=False,
            )

        self.assertEqual(dataset.split_plan.status, "ok")
        self.assertEqual(dataset.split_plan.planner, "beam_search")
        self.assertIsNotNone(dataset.split_plan.assignment_score)
        self.assertEqual(dataset.policy["split_planner_exhaustive_max_users"], 2)
        self.assertEqual(dataset.policy["split_planner_beam_width"], 10)
        self.assertEqual({sample.split for sample in dataset.validated_samples}, {"train", "validation", "test"})

    def test_split_assignment_reports_insufficient_train_label_coverage_when_labels_are_user_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths: list[Path] = []
            for user_id, gesture_label in (("U01", "FIST"), ("U02", "BRAVO"), ("U03", "OPEN_PALM")):
                path = Path(tmpdir) / f"{gesture_label}_{user_id}.json"
                payload = _session_payload(
                    gesture_label=gesture_label,
                    user_id=user_id,
                    session_id=f"sess_{user_id}",
                    samples=[_sample_payload(gesture_label=gesture_label, user_id=user_id, session_id=f"sess_{user_id}", sample_index=0)],
                )
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)

            dataset = validate_recording_files(paths, apply_outlier_filter=False)

        self.assertEqual(dataset.split_plan.status, "insufficient_train_label_coverage")
        self.assertEqual(dataset.summary["split_status"], "insufficient_train_label_coverage")

    def test_split_assignment_reports_insufficient_users_for_three_way_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths: list[Path] = []
            for user_id, gesture_label in (("U01", "FIST"), ("U02", "BRAVO")):
                path = Path(tmpdir) / f"{gesture_label}_{user_id}.json"
                payload = _session_payload(
                    gesture_label=gesture_label,
                    user_id=user_id,
                    session_id=f"sess_{user_id}",
                    samples=[_sample_payload(gesture_label=gesture_label, user_id=user_id, session_id=f"sess_{user_id}", sample_index=0)],
                )
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)
            dataset = validate_recording_files(paths, apply_outlier_filter=False)

        self.assertEqual(dataset.split_plan.status, "insufficient_users_for_disjoint_split")
        self.assertEqual(dataset.summary["split_status"], "insufficient_users_for_disjoint_split")

    def test_group_cv_helpers_return_group_aware_strategy(self):
        strategy_name, n_splits = build_group_cv_strategy(num_users=4)
        self.assertEqual(strategy_name, "StratifiedGroupKFold")
        self.assertEqual(n_splits, 4)
        cv = instantiate_group_cv(
            labels=["FIST", "BRAVO", "OPEN_PALM", "FIST", "BRAVO", "OPEN_PALM"],
            groups=["U01", "U01", "U01", "U02", "U02", "U02"],
        )
        self.assertEqual(cv.__class__.__name__, "GroupKFold")
        cv = instantiate_group_cv(
            labels=["FIST", "BRAVO", "OPEN_PALM", "FIST", "BRAVO", "OPEN_PALM", "FIST", "BRAVO", "OPEN_PALM"],
            groups=["U01", "U01", "U01", "U02", "U02", "U02", "U03", "U03", "U03"],
        )
        self.assertEqual(cv.__class__.__name__, "StratifiedGroupKFold")
        with self.assertRaises(ValueError):
            instantiate_group_cv(
                labels=["FIST", "FIST", "BRAVO", "BRAVO"],
                groups=["U01", "U02", "U01", "U01"],
            )

    def test_save_validated_dataset_writes_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "fist_u01.json"
            output = Path(tmpdir) / "validated.json"
            payload = _session_payload(
                gesture_label="FIST",
                user_id="U01",
                session_id="sess_u01",
                samples=[_sample_payload(gesture_label="FIST", user_id="U01", session_id="sess_u01", sample_index=0)],
            )
            source.write_text(json.dumps(payload), encoding="utf-8")
            dataset = validate_recording_files([source], assign_splits=False, apply_outlier_filter=False)
            save_validated_dataset(dataset, output)
            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(written["summary"]["validated_sample_count"], 1)
        self.assertEqual(written["validated_samples"][0]["gesture_label"], "FIST")


if __name__ == "__main__":
    unittest.main()
