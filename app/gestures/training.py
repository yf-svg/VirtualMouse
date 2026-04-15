from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION
from app.gestures.model_bundle import (
    CANDIDATE_ARTIFACT_KIND,
    RUNTIME_ARTIFACT_KIND,
    RUNTIME_BUNDLE_VERSION,
)
from app.gestures.validation import resolve_group_cv_plan


TRAINER_VERSION = "phase4.2.v1"


@dataclass(frozen=True)
class TrainingPolicy:
    expected_schema_version: str = FEATURE_SCHEMA_VERSION
    expected_feature_dimension: int = FEATURE_DIMENSION
    required_split_status: str = "ok"
    kernel: str = "rbf"
    c_value: float = 4.0
    gamma: str = "scale"
    class_weight: str = "balanced"
    probability: bool = True
    random_state: int = 42
    search_enabled: bool = True
    search_c_values: tuple[float, ...] = (1.0, 4.0, 8.0)
    search_gamma_values: tuple[str | float, ...] = ("scale", 0.08, 0.03)
    search_refit_metric: str = "macro_f1"


@dataclass(frozen=True)
class SplitMetrics:
    sample_count: int
    accuracy: float
    macro_f1: float
    labels: tuple[str, ...] = ()
    per_label: dict[str, "LabelMetrics"] | None = None
    confusion_matrix: dict[str, dict[str, int]] | None = None


@dataclass(frozen=True)
class LabelMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class TrainingResult:
    trainer_version: str
    trained_at: str
    input_dataset_path: str
    output_model_path: str
    output_report_path: str
    labels: tuple[str, ...]
    split_status: str
    metrics: dict[str, SplitMetrics]
    sample_counts_by_split: dict[str, int]
    model_selection_status: str
    model_selection_reason: str | None
    best_params: dict[str, Any]
    search_param_grid: dict[str, list[Any]]
    training_policy: dict[str, Any]
    cv_strategy: str | None
    cv_n_splits: int | None
    cv_best_score: float | None

    def to_report_payload(self) -> dict[str, Any]:
        return {
            "trainer_version": self.trainer_version,
            "trained_at": self.trained_at,
            "input_dataset_path": self.input_dataset_path,
            "output_model_path": self.output_model_path,
            "output_report_path": self.output_report_path,
            "labels": list(self.labels),
            "split_status": self.split_status,
            "metrics": {name: asdict(metrics) for name, metrics in self.metrics.items()},
            "sample_counts_by_split": dict(self.sample_counts_by_split),
            "model_selection_status": self.model_selection_status,
            "model_selection_reason": self.model_selection_reason,
            "best_params": dict(self.best_params),
            "search_param_grid": dict(self.search_param_grid),
            "training_policy": dict(self.training_policy),
            "cv_strategy": self.cv_strategy,
            "cv_n_splits": self.cv_n_splits,
            "cv_best_score": self.cv_best_score,
        }


@dataclass(frozen=True)
class RuntimeBundleExportResult:
    source_candidate_path: str
    output_model_path: str
    source_report_path: str | None
    bundle_version: str
    labels: tuple[str, ...]
    schema_version: str
    feature_dimension: int
    min_confidence: float


@dataclass(frozen=True)
class ModelSelectionResult:
    model: Any
    status: str
    reason: str | None
    best_params: dict[str, Any]
    search_param_grid: dict[str, list[Any]]
    cv_strategy: str | None
    cv_n_splits: int | None
    cv_best_score: float | None


def load_validated_dataset_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Validated dataset payload must be a JSON object.")
    return payload


def load_training_report_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Training report payload must be a JSON object.")
    return payload


def export_runtime_model_bundle(
    candidate_model_path: str | Path,
    *,
    output_model_path: str | Path,
    min_confidence: float = 0.70,
    training_report_path: str | Path | None = None,
) -> RuntimeBundleExportResult:
    candidate_model_path = Path(candidate_model_path)
    output_model_path = Path(output_model_path)
    training_report_path = Path(training_report_path) if training_report_path is not None else None

    candidate_payload = joblib.load(candidate_model_path)
    if not isinstance(candidate_payload, dict):
        raise ValueError("Candidate artifact must be a dict payload.")
    if candidate_payload.get("artifact_kind") != CANDIDATE_ARTIFACT_KIND:
        raise ValueError("Candidate artifact_kind must be 'training_candidate'.")

    labels = tuple(str(label) for label in candidate_payload.get("labels", []))
    if not labels:
        raise ValueError("Candidate artifact must contain non-empty labels.")
    if candidate_payload.get("model") is None:
        raise ValueError("Candidate artifact must contain a trained model.")

    schema_version = str(candidate_payload.get("schema_version") or "")
    feature_dimension = int(candidate_payload.get("feature_dimension") or -1)
    if schema_version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Candidate artifact schema version must be {FEATURE_SCHEMA_VERSION!r}; got {schema_version!r}"
        )
    if feature_dimension != FEATURE_DIMENSION:
        raise ValueError(
            f"Candidate artifact feature dimension must be {FEATURE_DIMENSION}; got {feature_dimension}"
        )

    report_payload: dict[str, Any] | None = None
    if training_report_path is not None:
        report_payload = load_training_report_payload(training_report_path)
        report_labels = tuple(str(label) for label in report_payload.get("labels", []))
        if report_labels and report_labels != labels:
            raise ValueError("Training report labels do not match candidate artifact labels.")

    runtime_payload = {
        "artifact_kind": RUNTIME_ARTIFACT_KIND,
        "bundle_version": RUNTIME_BUNDLE_VERSION,
        "trainer_version": str(candidate_payload.get("trainer_version") or TRAINER_VERSION),
        "trained_at": str(candidate_payload.get("trained_at") or ""),
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "source_candidate_path": str(candidate_model_path),
        "source_dataset_path": str(candidate_payload.get("source_dataset_path") or ""),
        "source_report_path": str(training_report_path) if training_report_path is not None else None,
        "schema_version": schema_version,
        "feature_dimension": feature_dimension,
        "labels": list(labels),
        "min_confidence": float(min_confidence),
        "metrics": report_payload.get("metrics") if report_payload is not None else None,
        "model": candidate_payload["model"],
    }

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(runtime_payload, output_model_path)
    return RuntimeBundleExportResult(
        source_candidate_path=str(candidate_model_path),
        output_model_path=str(output_model_path),
        source_report_path=str(training_report_path) if training_report_path is not None else None,
        bundle_version=RUNTIME_BUNDLE_VERSION,
        labels=labels,
        schema_version=schema_version,
        feature_dimension=feature_dimension,
        min_confidence=float(min_confidence),
    )


def train_svm_from_validated_dataset(
    dataset_path: str | Path,
    *,
    output_model_path: str | Path,
    output_report_path: str | Path,
    policy: TrainingPolicy | None = None,
) -> TrainingResult:
    policy = policy or TrainingPolicy()
    dataset_path = Path(dataset_path)
    output_model_path = Path(output_model_path)
    output_report_path = Path(output_report_path)

    payload = load_validated_dataset_payload(dataset_path)
    _validate_training_payload(payload, policy)

    samples = list(payload.get("validated_samples", []))
    split_groups = _samples_by_split(samples)

    train_samples = split_groups["train"]
    validation_samples = split_groups["validation"]
    test_samples = split_groups["test"]

    x_train, y_train = _xy(train_samples)
    _validate_training_matrix(x_train, y_train)

    selection = _select_model(train_samples, policy)
    model = selection.model

    metrics = {
        "train": _evaluate_split(model, train_samples),
        "validation": _evaluate_split(model, validation_samples),
        "test": _evaluate_split(model, test_samples),
    }

    trained_at = datetime.now().isoformat(timespec="seconds")
    labels = tuple(sorted({str(label) for label in y_train}))

    model_payload = {
        "artifact_kind": CANDIDATE_ARTIFACT_KIND,
        "trainer_version": TRAINER_VERSION,
        "trained_at": trained_at,
        "source_dataset_path": str(dataset_path),
        "schema_version": payload.get("feature_schema", {}).get("version", policy.expected_schema_version),
        "feature_dimension": payload.get("feature_schema", {}).get("dimension", policy.expected_feature_dimension),
        "labels": list(labels),
        "model_selection_status": selection.status,
        "model_selection_reason": selection.reason,
        "best_params": dict(selection.best_params),
        "search_param_grid": dict(selection.search_param_grid),
        "training_policy": _training_policy_payload(policy),
        "cv_strategy": selection.cv_strategy,
        "cv_n_splits": selection.cv_n_splits,
        "cv_best_score": selection.cv_best_score,
        "model": model,
    }

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_payload, output_model_path)

    result = TrainingResult(
        trainer_version=TRAINER_VERSION,
        trained_at=trained_at,
        input_dataset_path=str(dataset_path),
        output_model_path=str(output_model_path),
        output_report_path=str(output_report_path),
        labels=labels,
        split_status=str(payload.get("summary", {}).get("split_status", payload.get("split_plan", {}).get("status"))),
        metrics=metrics,
        sample_counts_by_split={name: len(items) for name, items in split_groups.items()},
        model_selection_status=selection.status,
        model_selection_reason=selection.reason,
        best_params=dict(selection.best_params),
        search_param_grid=dict(selection.search_param_grid),
        training_policy=_training_policy_payload(policy),
        cv_strategy=selection.cv_strategy,
        cv_n_splits=selection.cv_n_splits,
        cv_best_score=selection.cv_best_score,
    )
    output_report_path.write_text(json.dumps(result.to_report_payload(), indent=2), encoding="utf-8")
    return result


def _validate_training_payload(payload: dict[str, Any], policy: TrainingPolicy) -> None:
    split_status = payload.get("summary", {}).get("split_status", payload.get("split_plan", {}).get("status"))
    if split_status != policy.required_split_status:
        raise ValueError(
            f"Validated dataset split_status must be {policy.required_split_status!r}; got {split_status!r}"
        )

    feature_schema = payload.get("feature_schema", {})
    if feature_schema.get("version") != policy.expected_schema_version:
        raise ValueError(
            f"Validated dataset schema version must be {policy.expected_schema_version!r}; "
            f"got {feature_schema.get('version')!r}"
        )
    if int(feature_schema.get("dimension", -1)) != policy.expected_feature_dimension:
        raise ValueError(
            f"Validated dataset feature dimension must be {policy.expected_feature_dimension}; "
            f"got {feature_schema.get('dimension')!r}"
        )

    samples = payload.get("validated_samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError("Validated dataset must contain non-empty validated_samples.")

    split_names = {str(sample.get("split")) for sample in samples}
    required_splits = {"train", "validation", "test"}
    if not required_splits.issubset(split_names):
        raise ValueError(
            f"Validated dataset must contain train/validation/test samples; got {sorted(split_names)!r}"
        )

    dataset_labels = {str(sample.get("gesture_label")) for sample in samples}
    train_labels = {
        str(sample.get("gesture_label"))
        for sample in samples
        if str(sample.get("split")) == "train"
    }
    missing_train_labels = sorted(dataset_labels - train_labels)
    if missing_train_labels:
        raise ValueError(
            "Training split must contain every dataset label; missing "
            f"{missing_train_labels!r} from train."
        )


def _samples_by_split(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {"train": [], "validation": [], "test": []}
    for sample in samples:
        split = str(sample.get("split"))
        if split in groups:
            groups[split].append(sample)
    return groups


def _xy(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([sample["feature_values"] for sample in samples], dtype=np.float64)
    y = np.asarray([sample["gesture_label"] for sample in samples], dtype=object)
    return x, y


def _xyg(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = _xy(samples)
    groups = np.asarray([sample["user_id"] for sample in samples], dtype=object)
    return x, y, groups


def _validate_training_matrix(x_train: np.ndarray, y_train: np.ndarray) -> None:
    if x_train.ndim != 2 or x_train.shape[0] == 0:
        raise ValueError("Training split must contain at least one sample.")
    if x_train.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"Training feature dimension must be {FEATURE_DIMENSION}; got {x_train.shape[1]}"
        )
    if len(set(str(label) for label in y_train.tolist())) < 2:
        raise ValueError("Training split must contain at least two gesture classes.")


def _build_svm_pipeline(
    policy: TrainingPolicy,
    *,
    c_value: float | None = None,
    gamma: str | float | None = None,
):
    return make_pipeline(
        StandardScaler(),
        SVC(
            kernel=policy.kernel,
            C=float(c_value if c_value is not None else policy.c_value),
            gamma=gamma if gamma is not None else policy.gamma,
            class_weight=policy.class_weight,
            probability=bool(policy.probability),
            random_state=policy.random_state,
            decision_function_shape="ovr",
        ),
    )


def _select_model(train_samples: list[dict[str, Any]], policy: TrainingPolicy) -> ModelSelectionResult:
    x_train, y_train, groups = _xyg(train_samples)
    unique_groups = {str(group) for group in groups.tolist()}
    search_param_grid = _search_param_grid(policy)
    _validate_search_policy(policy, search_param_grid)

    baseline_model = _build_svm_pipeline(policy)
    baseline_params = {"kernel": policy.kernel, "C": float(policy.c_value), "gamma": policy.gamma}

    if not policy.search_enabled:
        baseline_model.fit(x_train, y_train)
        return ModelSelectionResult(
            model=baseline_model,
            status="search_disabled",
            reason="search_disabled_by_policy",
            best_params=baseline_params,
            search_param_grid=search_param_grid,
            cv_strategy=None,
            cv_n_splits=None,
            cv_best_score=None,
        )

    if len(unique_groups) < 2:
        baseline_model.fit(x_train, y_train)
        return ModelSelectionResult(
            model=baseline_model,
            status="search_skipped_insufficient_train_users",
            reason="train_split_has_fewer_than_two_distinct_users",
            best_params=baseline_params,
            search_param_grid=search_param_grid,
            cv_strategy=None,
            cv_n_splits=None,
            cv_best_score=None,
        )

    cv_plan = resolve_group_cv_plan(
        labels=[str(label) for label in y_train.tolist()],
        groups=[str(group) for group in groups.tolist()],
        random_state=policy.random_state,
    )
    if cv_plan.status != "ok" or cv_plan.strategy is None or cv_plan.n_splits is None:
        baseline_model.fit(x_train, y_train)
        return ModelSelectionResult(
            model=baseline_model,
            status="search_skipped_cv_unavailable",
            reason=cv_plan.reason,
            best_params=baseline_params,
            search_param_grid=search_param_grid,
            cv_strategy=None,
            cv_n_splits=None,
            cv_best_score=None,
        )

    cv = _build_svm_pipeline(policy)
    try:
        from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

        if cv_plan.strategy == "StratifiedGroupKFold":
            cv_strategy = StratifiedGroupKFold(
                n_splits=cv_plan.n_splits,
                shuffle=True,
                random_state=policy.random_state,
            )
        elif cv_plan.strategy == "GroupKFold":
            cv_strategy = GroupKFold(n_splits=cv_plan.n_splits)
        else:
            raise ValueError(f"Unsupported CV strategy: {cv_plan.strategy}")

        search = GridSearchCV(
            estimator=cv,
            param_grid=search_param_grid,
            scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
            refit=policy.search_refit_metric,
            cv=cv_strategy,
            n_jobs=None,
        )
        search.fit(x_train, y_train, groups=groups)
    except ValueError as exc:
        raise RuntimeError(f"Grouped grid search failed unexpectedly: {exc}") from exc

    best = search.best_estimator_
    best_params = {
        "kernel": policy.kernel,
        "C": float(search.best_params_["svc__C"]),
        "gamma": search.best_params_["svc__gamma"],
    }
    return ModelSelectionResult(
        model=best,
        status="search_ok",
        reason="grid_search_completed",
        best_params=best_params,
        search_param_grid=search_param_grid,
        cv_strategy=cv_plan.strategy,
        cv_n_splits=cv_plan.n_splits,
        cv_best_score=float(search.best_score_),
    )


def _search_param_grid(policy: TrainingPolicy) -> dict[str, list[Any]]:
    return {
        "svc__C": [float(value) for value in policy.search_c_values],
        "svc__gamma": list(policy.search_gamma_values),
    }


def _validate_search_policy(policy: TrainingPolicy, search_param_grid: dict[str, list[Any]]) -> None:
    if not policy.search_enabled:
        return
    if policy.search_refit_metric not in {"macro_f1", "accuracy"}:
        raise ValueError(
            "TrainingPolicy.search_refit_metric must be one of ['accuracy', 'macro_f1']; "
            f"got {policy.search_refit_metric!r}"
        )
    if not search_param_grid["svc__C"]:
        raise ValueError("TrainingPolicy.search_c_values must not be empty when search_enabled=True.")
    if not search_param_grid["svc__gamma"]:
        raise ValueError("TrainingPolicy.search_gamma_values must not be empty when search_enabled=True.")


def _training_policy_payload(policy: TrainingPolicy) -> dict[str, Any]:
    return {
        "expected_schema_version": policy.expected_schema_version,
        "expected_feature_dimension": policy.expected_feature_dimension,
        "required_split_status": policy.required_split_status,
        "kernel": policy.kernel,
        "c_value": float(policy.c_value),
        "gamma": policy.gamma,
        "class_weight": policy.class_weight,
        "probability": bool(policy.probability),
        "random_state": policy.random_state,
        "search_enabled": bool(policy.search_enabled),
        "search_c_values": [float(value) for value in policy.search_c_values],
        "search_gamma_values": list(policy.search_gamma_values),
        "search_refit_metric": policy.search_refit_metric,
    }


def _evaluate_split(model: Any, samples: list[dict[str, Any]]) -> SplitMetrics:
    if not samples:
        return SplitMetrics(
            sample_count=0,
            accuracy=0.0,
            macro_f1=0.0,
            labels=(),
            per_label={},
            confusion_matrix={},
        )

    x_split, y_true = _xy(samples)
    y_pred = model.predict(x_split)
    labels = tuple(sorted({str(label) for label in y_true.tolist()} | {str(label) for label in y_pred.tolist()}))
    precision, recall, f1_values, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
    )
    per_label = {
        label: LabelMetrics(
            precision=float(precision[idx]),
            recall=float(recall[idx]),
            f1=float(f1_values[idx]),
            support=int(support[idx]),
        )
        for idx, label in enumerate(labels)
    }
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    matrix_payload = {
        actual_label: {
            predicted_label: int(matrix[actual_idx, predicted_idx])
            for predicted_idx, predicted_label in enumerate(labels)
        }
        for actual_idx, actual_label in enumerate(labels)
    }
    return SplitMetrics(
        sample_count=len(samples),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        labels=labels,
        per_label=per_label,
        confusion_matrix=matrix_payload,
    )
