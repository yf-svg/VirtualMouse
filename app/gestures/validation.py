from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from app.gestures.features import (
    DEFAULT_FEATURE_CFG,
    FEATURE_DIMENSION,
    FEATURE_SCHEMA_VERSION,
    feature_schema,
)
from app.gestures.sets.labels import ALL_ALLOWED_LABELS


VALIDATION_ENGINE_VERSION = "phase4.1.v1"
DEFAULT_ALLOWED_LABELS = ALL_ALLOWED_LABELS


@dataclass(frozen=True)
class ValidationPolicy:
    allowed_labels: frozenset[str] = DEFAULT_ALLOWED_LABELS
    expected_schema_version: str = FEATURE_SCHEMA_VERSION
    expected_feature_dimension: int = FEATURE_DIMENSION
    min_quality_scale: float = DEFAULT_FEATURE_CFG.min_quality_scale
    min_quality_palm_width: float = DEFAULT_FEATURE_CFG.min_quality_palm_width
    min_quality_bbox_width: float = DEFAULT_FEATURE_CFG.min_bbox_width
    min_quality_bbox_height: float = DEFAULT_FEATURE_CFG.min_bbox_height
    duplicate_round_decimals: int = 8
    outlier_min_samples_per_label: int = 12
    outlier_modified_z_threshold: float = 7.5
    outlier_feature_fraction_threshold: float = 0.18
    min_users_for_disjoint_split: int = 3
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    split_names: tuple[str, str, str] = ("train", "validation", "test")
    split_planner_exhaustive_max_users: int = 8
    split_planner_beam_width: int = 64
    random_state: int = 42


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    level: str = "error"


@dataclass(frozen=True)
class ValidatedSample:
    sample_id: str
    gesture_label: str
    user_id: str
    session_id: str
    handedness: str | None
    capture_context: dict[str, str]
    recording_path: str
    sample_index: int
    feature_values: tuple[float, ...]
    schema_version: str
    quality_reason: str
    quality_scale: float
    quality_palm_width: float
    quality_bbox_width: float
    quality_bbox_height: float
    split: str | None = None


@dataclass(frozen=True)
class RejectedSample:
    sample_id: str
    gesture_label: str
    user_id: str
    session_id: str
    recording_path: str
    sample_index: int
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class RejectedSession:
    recording_path: str
    session_id: str
    issues: tuple[ValidationIssue, ...]


@dataclass(frozen=True)
class DatasetSplitPlan:
    status: str
    assignments: dict[str, str]
    issues: tuple[ValidationIssue, ...] = ()
    cv_strategy: str | None = None
    cv_n_splits: int | None = None
    planner: str | None = None
    assignment_score: float | None = None


@dataclass(frozen=True)
class GroupCvPlan:
    status: str
    strategy: str | None
    n_splits: int | None
    reason: str


@dataclass(frozen=True)
class _SplitSearchState:
    assignments: dict[str, str]
    score: float


@dataclass(frozen=True)
class ValidatedDataset:
    generated_at: str
    validator_version: str
    policy: dict[str, Any]
    feature_schema: dict[str, Any]
    validated_samples: tuple[ValidatedSample, ...]
    rejected_samples: tuple[RejectedSample, ...]
    rejected_sessions: tuple[RejectedSession, ...]
    split_plan: DatasetSplitPlan
    summary: dict[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "validator_version": self.validator_version,
            "policy": dict(self.policy),
            "feature_schema": dict(self.feature_schema),
            "validated_samples": [asdict(sample) for sample in self.validated_samples],
            "rejected_samples": [asdict(sample) for sample in self.rejected_samples],
            "rejected_sessions": [asdict(session) for session in self.rejected_sessions],
            "split_plan": {
                "status": self.split_plan.status,
                "assignments": dict(self.split_plan.assignments),
                "issues": [asdict(issue) for issue in self.split_plan.issues],
                "cv_strategy": self.split_plan.cv_strategy,
                "cv_n_splits": self.split_plan.cv_n_splits,
                "planner": self.split_plan.planner,
                "assignment_score": self.split_plan.assignment_score,
            },
            "summary": dict(self.summary),
        }


def load_recording_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Recording payload must be a JSON object.")
    return payload


def save_validated_dataset(dataset: ValidatedDataset, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset.to_payload(), indent=2), encoding="utf-8")
    return output


def build_group_cv_strategy(*, num_users: int) -> tuple[str, int]:
    if num_users < 2:
        raise ValueError("Grouped cross-validation requires at least two distinct users.")
    n_splits = min(5, num_users)
    if n_splits < 2:
        raise ValueError("Grouped cross-validation requires at least two splits.")
    if n_splits >= 3:
        return ("StratifiedGroupKFold", n_splits)
    return ("GroupKFold", n_splits)


def instantiate_group_cv(*, labels: list[str], groups: list[str], random_state: int = 42):
    plan = resolve_group_cv_plan(labels=labels, groups=groups, random_state=random_state)
    if plan.status != "ok" or plan.strategy is None or plan.n_splits is None:
        raise ValueError(plan.reason)
    return _build_group_cv(plan.strategy, plan.n_splits, random_state=random_state)


def resolve_group_cv_plan(*, labels: list[str], groups: list[str], random_state: int = 42) -> GroupCvPlan:
    labels_arr = np.asarray([str(label) for label in labels], dtype=object)
    groups_arr = np.asarray([str(group) for group in groups], dtype=object)

    if labels_arr.size == 0 or groups_arr.size == 0:
        return GroupCvPlan(
            status="unavailable",
            strategy=None,
            n_splits=None,
            reason="group_cv_requires_non_empty_labels_and_groups",
        )
    if labels_arr.size != groups_arr.size:
        return GroupCvPlan(
            status="unavailable",
            strategy=None,
            n_splits=None,
            reason="group_cv_requires_labels_and_groups_with_equal_length",
        )

    unique_groups = sorted({str(group) for group in groups_arr.tolist()})
    if len(unique_groups) < 2:
        return GroupCvPlan(
            status="unavailable",
            strategy=None,
            n_splits=None,
            reason="group_cv_requires_at_least_two_distinct_users",
        )

    unique_labels = sorted({str(label) for label in labels_arr.tolist()})
    x_dummy = np.zeros((labels_arr.size, 1), dtype=np.uint8)

    for strategy_name, n_splits in _candidate_group_cv_specs(num_groups=len(unique_groups)):
        cv = _build_group_cv(strategy_name, n_splits, random_state=random_state)
        try:
            splits = list(cv.split(x_dummy, labels_arr, groups_arr))
        except ValueError:
            continue
        if _cv_preserves_train_label_coverage(splits, labels_arr, required_labels=unique_labels):
            return GroupCvPlan(
                status="ok",
                strategy=strategy_name,
                n_splits=n_splits,
                reason="validated_fold_train_label_coverage",
            )

    label_group_counts = {
        label: len({group for current_label, group in zip(labels_arr.tolist(), groups_arr.tolist()) if current_label == label})
        for label in unique_labels
    }
    min_label_groups = min(label_group_counts.values()) if label_group_counts else 0
    return GroupCvPlan(
        status="unavailable",
        strategy=None,
        n_splits=None,
        reason=(
            "no_group_cv_strategy_preserves_train_label_coverage:"
            f"users={len(unique_groups)}:min_label_groups={min_label_groups}"
        ),
    )


def _candidate_group_cv_specs(*, num_groups: int) -> list[tuple[str, int]]:
    max_splits = min(5, int(num_groups))
    specs: list[tuple[str, int]] = []
    for n_splits in range(max_splits, 2, -1):
        specs.append(("StratifiedGroupKFold", n_splits))
    for n_splits in range(max_splits, 1, -1):
        specs.append(("GroupKFold", n_splits))
    return specs


def _build_group_cv(strategy_name: str, n_splits: int, *, random_state: int):
    if strategy_name == "StratifiedGroupKFold":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if strategy_name == "GroupKFold":
        return GroupKFold(n_splits=n_splits)
    raise ValueError(f"Unsupported group CV strategy: {strategy_name}")


def _cv_preserves_train_label_coverage(
    splits: list[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    *,
    required_labels: list[str],
) -> bool:
    required = set(required_labels)
    if not required:
        return False
    for train_idx, _test_idx in splits:
        train_labels = {str(label) for label in labels[train_idx].tolist()}
        if train_labels != required:
            return False
    return True


def validate_recording_files(
    paths: list[str | Path],
    *,
    policy: ValidationPolicy | None = None,
    assign_splits: bool = True,
    apply_outlier_filter: bool = True,
) -> ValidatedDataset:
    policy = policy or ValidationPolicy()
    valid_samples: list[ValidatedSample] = []
    rejected_samples: list[RejectedSample] = []
    rejected_sessions: list[RejectedSession] = []

    for path in [Path(p) for p in paths]:
        session_valid, session_rejected, session_issue = _validate_recording_file(path, policy)
        valid_samples.extend(session_valid)
        rejected_samples.extend(session_rejected)
        if session_issue is not None:
            rejected_sessions.append(session_issue)

    deduped_samples, duplicate_rejections = _reject_duplicate_samples(valid_samples, policy)
    rejected_samples.extend(duplicate_rejections)

    if apply_outlier_filter:
        filtered_samples, outlier_rejections = _reject_outliers(deduped_samples, policy)
        valid_samples = filtered_samples
        rejected_samples.extend(outlier_rejections)
    else:
        valid_samples = deduped_samples

    split_plan = (
        _plan_group_splits(valid_samples, policy) if assign_splits else DatasetSplitPlan(status="not_requested", assignments={})
    )
    if split_plan.assignments:
        valid_samples = [
            _with_split(sample, split_plan.assignments.get(sample.sample_id))
            for sample in valid_samples
        ]

    dataset = ValidatedDataset(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        validator_version=VALIDATION_ENGINE_VERSION,
        policy=_policy_payload(policy),
        feature_schema=feature_schema().as_dict(),
        validated_samples=tuple(valid_samples),
        rejected_samples=tuple(rejected_samples),
        rejected_sessions=tuple(rejected_sessions),
        split_plan=split_plan,
        summary=_build_summary(valid_samples, rejected_samples, rejected_sessions, split_plan),
    )
    return dataset


def _with_split(sample: ValidatedSample, split: str | None) -> ValidatedSample:
    return ValidatedSample(
        sample_id=sample.sample_id,
        gesture_label=sample.gesture_label,
        user_id=sample.user_id,
        session_id=sample.session_id,
        handedness=sample.handedness,
        capture_context=dict(sample.capture_context),
        recording_path=sample.recording_path,
        sample_index=sample.sample_index,
        feature_values=sample.feature_values,
        schema_version=sample.schema_version,
        quality_reason=sample.quality_reason,
        quality_scale=sample.quality_scale,
        quality_palm_width=sample.quality_palm_width,
        quality_bbox_width=sample.quality_bbox_width,
        quality_bbox_height=sample.quality_bbox_height,
        split=split,
    )


def _validate_recording_file(path: Path, policy: ValidationPolicy) -> tuple[list[ValidatedSample], list[RejectedSample], RejectedSession | None]:
    payload = load_recording_payload(path)
    session_issues = _validate_session_payload(payload, policy)
    session_id = str(payload.get("session_id", "unknown"))
    if session_issues:
        return [], [], RejectedSession(recording_path=str(path), session_id=session_id, issues=tuple(session_issues))

    capture_context = dict(payload.get("capture_context", {}))
    validated: list[ValidatedSample] = []
    rejected: list[RejectedSample] = []
    for sample_payload in payload.get("samples", []):
        sample_issues = _validate_sample_payload(payload, sample_payload, policy)
        sample_id = _sample_id(path, sample_payload)
        if sample_issues:
            rejected.append(
                RejectedSample(
                    sample_id=sample_id,
                    gesture_label=str(sample_payload.get("gesture_label", payload["gesture_label"])),
                    user_id=str(sample_payload.get("user_id", payload["user_id"])),
                    session_id=str(sample_payload.get("session_id", payload["session_id"])),
                    recording_path=str(path),
                    sample_index=int(sample_payload.get("sample_index", -1)),
                    issues=tuple(sample_issues),
                )
            )
            continue

        validated.append(
            ValidatedSample(
                sample_id=sample_id,
                gesture_label=str(sample_payload["gesture_label"]),
                user_id=str(sample_payload["user_id"]),
                session_id=str(sample_payload["session_id"]),
                handedness=sample_payload.get("handedness"),
                capture_context=capture_context,
                recording_path=str(path),
                sample_index=int(sample_payload["sample_index"]),
                feature_values=tuple(float(v) for v in sample_payload["feature_values"]),
                schema_version=str(sample_payload["schema_version"]),
                quality_reason=str(sample_payload["quality_reason"]),
                quality_scale=float(sample_payload["quality_scale"]),
                quality_palm_width=float(sample_payload["quality_palm_width"]),
                quality_bbox_width=float(sample_payload["quality_bbox_width"]),
                quality_bbox_height=float(sample_payload["quality_bbox_height"]),
            )
        )
    return validated, rejected, None


def _validate_session_payload(payload: dict[str, Any], policy: ValidationPolicy) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required_fields = (
        "gesture_label",
        "user_id",
        "session_id",
        "schema_version",
        "feature_dimension",
        "samples",
    )
    for field_name in required_fields:
        if field_name not in payload:
            issues.append(ValidationIssue("missing_session_field", f"Missing session field: {field_name}"))

    gesture_label = payload.get("gesture_label")
    if gesture_label not in policy.allowed_labels:
        issues.append(ValidationIssue("unknown_label", f"Unknown gesture label: {gesture_label!r}"))

    if payload.get("schema_version") != policy.expected_schema_version:
        issues.append(
            ValidationIssue(
                "schema_version_mismatch",
                f"Expected schema_version={policy.expected_schema_version}, got {payload.get('schema_version')!r}",
            )
        )
    if int(payload.get("feature_dimension", -1)) != policy.expected_feature_dimension:
        issues.append(
            ValidationIssue(
                "feature_dimension_mismatch",
                f"Expected feature_dimension={policy.expected_feature_dimension}, got {payload.get('feature_dimension')!r}",
            )
        )
    if not isinstance(payload.get("samples"), list):
        issues.append(ValidationIssue("invalid_samples_field", "Session samples must be a list."))
    capture_context = payload.get("capture_context", {})
    if capture_context is not None and not isinstance(capture_context, dict):
        issues.append(ValidationIssue("invalid_capture_context", "Session capture_context must be an object."))
    return issues


def _validate_sample_payload(
    session_payload: dict[str, Any],
    sample_payload: dict[str, Any],
    policy: ValidationPolicy,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    required_fields = (
        "sample_index",
        "gesture_label",
        "user_id",
        "session_id",
        "schema_version",
        "quality_reason",
        "quality_scale",
        "quality_palm_width",
        "quality_bbox_width",
        "quality_bbox_height",
        "feature_values",
    )
    for field_name in required_fields:
        if field_name not in sample_payload:
            issues.append(ValidationIssue("missing_sample_field", f"Missing sample field: {field_name}"))
    if issues:
        return issues

    if sample_payload["gesture_label"] != session_payload["gesture_label"]:
        issues.append(ValidationIssue("session_label_mismatch", "Sample gesture label does not match session gesture label."))
    if sample_payload["user_id"] != session_payload["user_id"]:
        issues.append(ValidationIssue("session_user_mismatch", "Sample user id does not match session user id."))
    if sample_payload["session_id"] != session_payload["session_id"]:
        issues.append(ValidationIssue("session_id_mismatch", "Sample session id does not match session session id."))
    if sample_payload["schema_version"] != policy.expected_schema_version:
        issues.append(
            ValidationIssue(
                "sample_schema_mismatch",
                f"Expected schema_version={policy.expected_schema_version}, got {sample_payload['schema_version']!r}",
            )
        )
    if sample_payload["quality_reason"] != "ok":
        issues.append(ValidationIssue("rejected_quality_reason", f"Sample quality_reason must be 'ok', got {sample_payload['quality_reason']!r}"))

    feature_values = sample_payload["feature_values"]
    if not isinstance(feature_values, list):
        issues.append(ValidationIssue("invalid_feature_values", "Sample feature_values must be a list."))
        return issues
    if len(feature_values) != policy.expected_feature_dimension:
        issues.append(
            ValidationIssue(
                "sample_dimension_mismatch",
                f"Expected {policy.expected_feature_dimension} features, got {len(feature_values)}",
            )
        )
        return issues

    numeric_values = [float(v) for v in feature_values]
    if not all(math.isfinite(v) for v in numeric_values):
        issues.append(ValidationIssue("non_finite_features", "Sample feature vector contains NaN or infinite values."))
        return issues

    issues.extend(_validate_feature_ranges(numeric_values))
    issues.extend(_validate_quality_metrics(sample_payload, policy))
    return issues


def _validate_feature_ranges(values: list[float]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    names = feature_schema().names
    for name, value in zip(names, values):
        if name.endswith("_angle"):
            if value < 0.0 or value > 180.0:
                issues.append(ValidationIssue("angle_out_of_range", f"{name}={value:.5f} is outside [0, 180]."))
                break
        elif name in {"extended_count", "curled_count", "near_palm_count"}:
            if value < 0.0 or value > 5.0:
                issues.append(ValidationIssue("count_out_of_range", f"{name}={value:.5f} is outside [0, 5]."))
                break
        elif name.endswith(("_x_norm", "_y_norm", "_z_norm")):
            if abs(value) > 5.0:
                issues.append(ValidationIssue("normalized_coordinate_outlier", f"{name}={value:.5f} exceeds absolute bound 5.0."))
                break
        else:
            if value < 0.0 or value > 6.0:
                issues.append(ValidationIssue("distance_ratio_out_of_range", f"{name}={value:.5f} is outside [0, 6]."))
                break
    return issues


def _validate_quality_metrics(sample_payload: dict[str, Any], policy: ValidationPolicy) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    metrics = {
        "quality_scale": float(sample_payload["quality_scale"]),
        "quality_palm_width": float(sample_payload["quality_palm_width"]),
        "quality_bbox_width": float(sample_payload["quality_bbox_width"]),
        "quality_bbox_height": float(sample_payload["quality_bbox_height"]),
    }
    if not all(math.isfinite(v) for v in metrics.values()):
        issues.append(ValidationIssue("non_finite_quality_metrics", "Quality metrics contain NaN or infinite values."))
        return issues
    if metrics["quality_scale"] < policy.min_quality_scale:
        issues.append(ValidationIssue("quality_scale_below_threshold", "quality_scale is below Phase 3 validation threshold."))
    if metrics["quality_palm_width"] < policy.min_quality_palm_width:
        issues.append(ValidationIssue("quality_palm_width_below_threshold", "quality_palm_width is below Phase 3 validation threshold."))
    if metrics["quality_bbox_width"] < policy.min_quality_bbox_width and metrics["quality_bbox_height"] < policy.min_quality_bbox_height:
        issues.append(ValidationIssue("quality_bbox_below_threshold", "Bounding box is below Phase 3 validation thresholds."))
    return issues


def _reject_duplicate_samples(samples: list[ValidatedSample], policy: ValidationPolicy) -> tuple[list[ValidatedSample], list[RejectedSample]]:
    seen: dict[tuple[Any, ...], str] = {}
    kept: list[ValidatedSample] = []
    rejected: list[RejectedSample] = []
    for sample in samples:
        rounded = tuple(round(v, policy.duplicate_round_decimals) for v in sample.feature_values)
        key = (sample.gesture_label, rounded)
        if key in seen:
            rejected.append(
                RejectedSample(
                    sample_id=sample.sample_id,
                    gesture_label=sample.gesture_label,
                    user_id=sample.user_id,
                    session_id=sample.session_id,
                    recording_path=sample.recording_path,
                    sample_index=sample.sample_index,
                    issues=(
                        ValidationIssue(
                            "duplicate_feature_vector",
                            f"Feature vector duplicates previously accepted sample {seen[key]}.",
                        ),
                    ),
                )
            )
            continue
        seen[key] = sample.sample_id
        kept.append(sample)
    return kept, rejected


def _reject_outliers(samples: list[ValidatedSample], policy: ValidationPolicy) -> tuple[list[ValidatedSample], list[RejectedSample]]:
    by_label: dict[str, list[ValidatedSample]] = {}
    for sample in samples:
        by_label.setdefault(sample.gesture_label, []).append(sample)

    kept: list[ValidatedSample] = []
    rejected: list[RejectedSample] = []
    for label, label_samples in by_label.items():
        if len(label_samples) < policy.outlier_min_samples_per_label:
            kept.extend(label_samples)
            continue

        matrix = np.asarray([sample.feature_values for sample in label_samples], dtype=np.float64)
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0)
        mad = np.where(mad < 1e-9, 1.0, mad)
        modified_z = 0.6745 * (matrix - median) / mad
        outlier_fraction = np.mean(np.abs(modified_z) > policy.outlier_modified_z_threshold, axis=1)

        for sample, fraction in zip(label_samples, outlier_fraction):
            if fraction > policy.outlier_feature_fraction_threshold:
                rejected.append(
                    RejectedSample(
                        sample_id=sample.sample_id,
                        gesture_label=sample.gesture_label,
                        user_id=sample.user_id,
                        session_id=sample.session_id,
                        recording_path=sample.recording_path,
                        sample_index=sample.sample_index,
                        issues=(
                            ValidationIssue(
                                "feature_outlier",
                                f"Robust MAD filter flagged sample: {fraction:.3f} of features exceeded threshold.",
                            ),
                        ),
                    )
                )
            else:
                kept.append(sample)
    return kept, rejected


def _plan_group_splits(samples: list[ValidatedSample], policy: ValidationPolicy) -> DatasetSplitPlan:
    if not samples:
        return DatasetSplitPlan(
            status="empty",
            assignments={},
            issues=(ValidationIssue("empty_dataset", "No validated samples are available for split planning."),),
        )

    users = sorted({sample.user_id for sample in samples})
    cv_plan = resolve_group_cv_plan(
        labels=[sample.gesture_label for sample in samples],
        groups=[sample.user_id for sample in samples],
        random_state=policy.random_state,
    )
    cv_strategy = cv_plan.strategy
    cv_n_splits = cv_plan.n_splits

    if len(users) < policy.min_users_for_disjoint_split:
        return DatasetSplitPlan(
            status="insufficient_users_for_disjoint_split",
            assignments={},
            issues=(
                ValidationIssue(
                    "insufficient_users_for_disjoint_split",
                    f"Need at least {policy.min_users_for_disjoint_split} users for leakage-safe train/validation/test splits; got {len(users)}.",
                ),
            ),
            cv_strategy=cv_strategy,
            cv_n_splits=cv_n_splits,
        )

    user_counts = {user: 0 for user in users}
    label_counts: dict[str, dict[str, int]] = {user: {} for user in users}
    for sample in samples:
        user_counts[sample.user_id] += 1
        label_counts[sample.user_id][sample.gesture_label] = label_counts[sample.user_id].get(sample.gesture_label, 0) + 1

    split_names = policy.split_names
    target_ratios = np.asarray(policy.split_ratios, dtype=np.float64)
    target_ratios = target_ratios / target_ratios.sum()
    labels = sorted({sample.gesture_label for sample in samples})
    total_label_counts = {
        label: sum(1 for sample in samples if sample.gesture_label == label)
        for label in labels
    }
    total_samples = len(samples)
    train_coverage_possible = _can_cover_all_labels_in_train(
        users=users,
        labels=labels,
        label_counts=label_counts,
        split_names=split_names,
    )

    if len(users) <= max(1, int(policy.split_planner_exhaustive_max_users)):
        planner_name = "exhaustive"
        best_assignment, best_score = _find_best_split_assignment_exhaustive(
            users=users,
            split_names=split_names,
            labels=labels,
            user_counts=user_counts,
            label_counts=label_counts,
            total_label_counts=total_label_counts,
            total_samples=total_samples,
            target_ratios=target_ratios,
        )
    else:
        planner_name = "beam_search"
        best_assignment, best_score = _find_best_split_assignment_beam(
            users=users,
            split_names=split_names,
            labels=labels,
            user_counts=user_counts,
            label_counts=label_counts,
            total_label_counts=total_label_counts,
            total_samples=total_samples,
            target_ratios=target_ratios,
            beam_width=max(4, int(policy.split_planner_beam_width)),
        )

    if best_assignment is None:
        if not train_coverage_possible:
            return DatasetSplitPlan(
                status="insufficient_train_label_coverage",
                assignments={},
                issues=(
                    ValidationIssue(
                        "insufficient_train_label_coverage",
                        "Could not assign leakage-safe train/validation/test splits while keeping every label represented in train.",
                    ),
                ),
                cv_strategy=cv_strategy,
                cv_n_splits=cv_n_splits,
                planner=planner_name,
            )
        return DatasetSplitPlan(
            status="split_failed",
            assignments={},
            issues=(ValidationIssue("split_failed", "Could not find a leakage-safe user split plan."),),
            cv_strategy=cv_strategy,
            cv_n_splits=cv_n_splits,
            planner=planner_name,
        )

    sample_assignments = {
        sample.sample_id: best_assignment[sample.user_id]
        for sample in samples
    }
    return DatasetSplitPlan(
        status="ok",
        assignments=sample_assignments,
        cv_strategy=cv_strategy,
        cv_n_splits=cv_n_splits,
        planner=planner_name,
        assignment_score=best_score,
    )


def _find_best_split_assignment_exhaustive(
    *,
    users: list[str],
    split_names: tuple[str, ...],
    labels: list[str],
    user_counts: dict[str, int],
    label_counts: dict[str, dict[str, int]],
    total_label_counts: dict[str, int],
    total_samples: int,
    target_ratios: np.ndarray,
) -> tuple[dict[str, str] | None, float | None]:
    best_assignment: dict[str, str] | None = None
    best_score: float | None = None
    for split_indexes in product(range(len(split_names)), repeat=len(users)):
        if len(set(split_indexes)) != len(split_names):
            continue
        assignment = {user: split_names[idx] for user, idx in zip(users, split_indexes)}
        if not _assignment_has_full_train_label_coverage(
            assignment=assignment,
            labels=labels,
            label_counts=label_counts,
        ):
            continue
        score = _score_full_assignment(
            assignment=assignment,
            split_names=split_names,
            labels=labels,
            user_counts=user_counts,
            label_counts=label_counts,
            total_label_counts=total_label_counts,
            total_samples=total_samples,
            target_ratios=target_ratios,
        )
        if best_score is None or score < best_score:
            best_assignment = assignment
            best_score = score
    return best_assignment, best_score


def _find_best_split_assignment_beam(
    *,
    users: list[str],
    split_names: tuple[str, ...],
    labels: list[str],
    user_counts: dict[str, int],
    label_counts: dict[str, dict[str, int]],
    total_label_counts: dict[str, int],
    total_samples: int,
    target_ratios: np.ndarray,
    beam_width: int,
) -> tuple[dict[str, str] | None, float | None]:
    if not users:
        return None, None

    ordered_users = _order_users_for_split_planning(
        users=users,
        label_counts=label_counts,
        total_label_counts=total_label_counts,
        user_counts=user_counts,
    )
    remaining_label_union = _remaining_label_union_by_user_order(ordered_users, label_counts)
    states = [_SplitSearchState(assignments={}, score=0.0)]

    for user_index, user_id in enumerate(ordered_users):
        next_states: list[_SplitSearchState] = []
        for state in states:
            for split_name in split_names:
                assignment = dict(state.assignments)
                assignment[user_id] = split_name
                if not _partial_assignment_is_feasible(
                    assignment=assignment,
                    split_names=split_names,
                    labels=labels,
                    remaining_label_union=remaining_label_union[user_index + 1],
                    remaining_users=len(ordered_users) - user_index - 1,
                    label_counts=label_counts,
                ):
                    continue
                score = _score_partial_assignment(
                    assignment=assignment,
                    split_names=split_names,
                    labels=labels,
                    user_counts=user_counts,
                    label_counts=label_counts,
                    total_label_counts=total_label_counts,
                    total_samples=total_samples,
                    target_ratios=target_ratios,
                    remaining_users=len(ordered_users) - user_index - 1,
                )
                next_states.append(_SplitSearchState(assignments=assignment, score=score))
        if not next_states:
            return None, None
        next_states.sort(key=lambda state: (state.score, tuple(sorted(state.assignments.items()))))
        states = next_states[:beam_width]

    best_assignment: dict[str, str] | None = None
    best_score: float | None = None
    for state in states:
        assignment = state.assignments
        if len(set(assignment.values())) != len(split_names):
            continue
        if not _assignment_has_full_train_label_coverage(
            assignment=assignment,
            labels=labels,
            label_counts=label_counts,
        ):
            continue
        score = _score_full_assignment(
            assignment=assignment,
            split_names=split_names,
            labels=labels,
            user_counts=user_counts,
            label_counts=label_counts,
            total_label_counts=total_label_counts,
            total_samples=total_samples,
            target_ratios=target_ratios,
        )
        if best_score is None or score < best_score:
            best_assignment = assignment
            best_score = score
    return best_assignment, best_score


def _can_cover_all_labels_in_train(
    *,
    users: list[str],
    labels: list[str],
    label_counts: dict[str, dict[str, int]],
    split_names: tuple[str, ...],
) -> bool:
    if not labels:
        return False
    max_train_users = len(users) - (len(split_names) - 1)
    if max_train_users <= 0:
        return False
    label_index = {label: idx for idx, label in enumerate(labels)}
    user_masks = []
    for user in users:
        mask = 0
        for label in label_counts[user]:
            mask |= 1 << label_index[label]
        user_masks.append(mask)
    full_mask = (1 << len(labels)) - 1
    reachable = {0: 0}
    for user_mask in user_masks:
        updates: dict[int, int] = {}
        for current_mask, used_users in list(reachable.items()):
            if used_users >= max_train_users:
                continue
            new_mask = current_mask | user_mask
            new_used = used_users + 1
            if new_used < updates.get(new_mask, new_used + 1):
                updates[new_mask] = new_used
        for mask, used_users in updates.items():
            if used_users < reachable.get(mask, used_users + 1):
                reachable[mask] = used_users
    return reachable.get(full_mask, max_train_users + 1) <= max_train_users


def _assignment_has_full_train_label_coverage(
    *,
    assignment: dict[str, str],
    labels: list[str],
    label_counts: dict[str, dict[str, int]],
) -> bool:
    train_labels = {
        label
        for user_id, split_name in assignment.items()
        if split_name == "train"
        for label in label_counts[user_id]
    }
    return train_labels == set(labels)


def _assignment_counts(
    *,
    assignment: dict[str, str],
    split_names: tuple[str, ...],
    labels: list[str],
    user_counts: dict[str, int],
    label_counts: dict[str, dict[str, int]],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    split_sample_counts = {split_name: 0 for split_name in split_names}
    split_label_counts = {split_name: {label: 0 for label in labels} for split_name in split_names}
    for user_id, split_name in assignment.items():
        split_sample_counts[split_name] += user_counts[user_id]
        for label, count in label_counts[user_id].items():
            split_label_counts[split_name][label] += count
    return split_sample_counts, split_label_counts


def _score_full_assignment(
    *,
    assignment: dict[str, str],
    split_names: tuple[str, ...],
    labels: list[str],
    user_counts: dict[str, int],
    label_counts: dict[str, dict[str, int]],
    total_label_counts: dict[str, int],
    total_samples: int,
    target_ratios: np.ndarray,
) -> float:
    split_sample_counts, split_label_counts = _assignment_counts(
        assignment=assignment,
        split_names=split_names,
        labels=labels,
        user_counts=user_counts,
        label_counts=label_counts,
    )
    score = 0.0
    for split_idx, split_name in enumerate(split_names):
        observed_ratio = split_sample_counts[split_name] / max(total_samples, 1)
        score += abs(observed_ratio - target_ratios[split_idx]) * 2.5
        for label in labels:
            overall_ratio = total_label_counts[label] / max(total_samples, 1)
            label_ratio = split_label_counts[split_name][label] / max(split_sample_counts[split_name], 1)
            score += abs(label_ratio - overall_ratio)
    return score


def _score_partial_assignment(
    *,
    assignment: dict[str, str],
    split_names: tuple[str, ...],
    labels: list[str],
    user_counts: dict[str, int],
    label_counts: dict[str, dict[str, int]],
    total_label_counts: dict[str, int],
    total_samples: int,
    target_ratios: np.ndarray,
    remaining_users: int,
) -> float:
    split_sample_counts, split_label_counts = _assignment_counts(
        assignment=assignment,
        split_names=split_names,
        labels=labels,
        user_counts=user_counts,
        label_counts=label_counts,
    )
    score = 0.0
    for split_idx, split_name in enumerate(split_names):
        observed_ratio = split_sample_counts[split_name] / max(total_samples, 1)
        score += abs(observed_ratio - target_ratios[split_idx]) * 2.5
        for label in labels:
            overall_ratio = total_label_counts[label] / max(total_samples, 1)
            label_ratio = split_label_counts[split_name][label] / max(split_sample_counts[split_name], 1)
            score += abs(label_ratio - overall_ratio)

    train_labels = {
        label
        for user_id, split_name in assignment.items()
        if split_name == "train"
        for label in label_counts[user_id]
    }
    missing_train_labels = len(set(labels) - train_labels)
    unused_splits = len(set(split_names) - set(assignment.values()))
    score += missing_train_labels * 4.0
    score += unused_splits * 1.5
    score += remaining_users * 0.01
    return score


def _order_users_for_split_planning(
    *,
    users: list[str],
    label_counts: dict[str, dict[str, int]],
    total_label_counts: dict[str, int],
    user_counts: dict[str, int],
) -> list[str]:
    def _key(user_id: str) -> tuple[float, int, str]:
        rarity_score = 0.0
        for label, count in label_counts[user_id].items():
            rarity_score += float(count) / max(1, total_label_counts[label])
        return (-rarity_score, -user_counts[user_id], user_id)

    return sorted(users, key=_key)


def _remaining_label_union_by_user_order(
    ordered_users: list[str],
    label_counts: dict[str, dict[str, int]],
) -> list[set[str]]:
    unions: list[set[str]] = [set() for _ in range(len(ordered_users) + 1)]
    for idx in range(len(ordered_users) - 1, -1, -1):
        unions[idx] = set(unions[idx + 1]) | set(label_counts[ordered_users[idx]])
    return unions


def _partial_assignment_is_feasible(
    *,
    assignment: dict[str, str],
    split_names: tuple[str, ...],
    labels: list[str],
    remaining_label_union: set[str],
    remaining_users: int,
    label_counts: dict[str, dict[str, int]],
) -> bool:
    used_splits = set(assignment.values())
    if len(set(split_names) - used_splits) > remaining_users:
        return False

    train_labels = {
        label
        for user_id, split_name in assignment.items()
        if split_name == "train"
        for label in label_counts[user_id]
    }
    if (train_labels | remaining_label_union) != set(labels):
        return False

    if remaining_users == 0:
        return len(used_splits) == len(split_names) and train_labels == set(labels)
    return True


def _build_summary(
    valid_samples: list[ValidatedSample],
    rejected_samples: list[RejectedSample],
    rejected_sessions: list[RejectedSession],
    split_plan: DatasetSplitPlan,
) -> dict[str, Any]:
    samples_by_label: dict[str, int] = {}
    samples_by_user: dict[str, int] = {}
    samples_by_split: dict[str, int] = {}
    for sample in valid_samples:
        samples_by_label[sample.gesture_label] = samples_by_label.get(sample.gesture_label, 0) + 1
        samples_by_user[sample.user_id] = samples_by_user.get(sample.user_id, 0) + 1
        if sample.split:
            samples_by_split[sample.split] = samples_by_split.get(sample.split, 0) + 1

    return {
        "validated_sample_count": len(valid_samples),
        "rejected_sample_count": len(rejected_samples),
        "rejected_session_count": len(rejected_sessions),
        "distinct_labels": sorted(samples_by_label),
        "distinct_users": sorted(samples_by_user),
        "samples_by_label": samples_by_label,
        "samples_by_user": samples_by_user,
        "samples_by_split": samples_by_split,
        "split_status": split_plan.status,
    }


def _sample_id(path: Path, sample_payload: dict[str, Any]) -> str:
    return f"{path.name}:{sample_payload.get('sample_index', 'unknown')}"


def _policy_payload(policy: ValidationPolicy) -> dict[str, Any]:
    return {
        "allowed_labels": sorted(policy.allowed_labels),
        "expected_schema_version": policy.expected_schema_version,
        "expected_feature_dimension": policy.expected_feature_dimension,
        "min_quality_scale": policy.min_quality_scale,
        "min_quality_palm_width": policy.min_quality_palm_width,
        "min_quality_bbox_width": policy.min_quality_bbox_width,
        "min_quality_bbox_height": policy.min_quality_bbox_height,
        "duplicate_round_decimals": policy.duplicate_round_decimals,
        "outlier_min_samples_per_label": policy.outlier_min_samples_per_label,
        "outlier_modified_z_threshold": policy.outlier_modified_z_threshold,
        "outlier_feature_fraction_threshold": policy.outlier_feature_fraction_threshold,
        "min_users_for_disjoint_split": policy.min_users_for_disjoint_split,
        "split_ratios": list(policy.split_ratios),
        "split_names": list(policy.split_names),
        "split_planner_exhaustive_max_users": int(policy.split_planner_exhaustive_max_users),
        "split_planner_beam_width": int(policy.split_planner_beam_width),
        "random_state": policy.random_state,
    }
