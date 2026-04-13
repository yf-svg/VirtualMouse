from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
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
from app.gestures.validation import instantiate_group_cv


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
    best_params: dict[str, Any]
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
            "best_params": dict(self.best_params),
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
    best_params: dict[str, Any]
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
        "best_params": dict(selection.best_params),
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
        best_params=dict(selection.best_params),
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

    baseline_model = _build_svm_pipeline(policy)
    baseline_params = {"kernel": policy.kernel, "C": float(policy.c_value), "gamma": policy.gamma}

    if not policy.search_enabled:
        baseline_model.fit(x_train, y_train)
        return ModelSelectionResult(
            model=baseline_model,
            status="search_disabled",
            best_params=baseline_params,
            cv_strategy=None,
            cv_n_splits=None,
            cv_best_score=None,
        )

    if len(unique_groups) < 2:
        baseline_model.fit(x_train, y_train)
        return ModelSelectionResult(
            model=baseline_model,
            status="search_skipped_insufficient_train_users",
            best_params=baseline_params,
            cv_strategy=None,
            cv_n_splits=None,
            cv_best_score=None,
        )

    try:
        cv = instantiate_group_cv(
            labels=[str(label) for label in y_train.tolist()],
            groups=[str(group) for group in groups.tolist()],
            random_state=policy.random_state,
        )
        search = GridSearchCV(
            estimator=_build_svm_pipeline(policy),
            param_grid={
                "svc__C": [float(value) for value in policy.search_c_values],
                "svc__gamma": list(policy.search_gamma_values),
            },
            scoring={"macro_f1": "f1_macro", "accuracy": "accuracy"},
            refit=policy.search_refit_metric,
            cv=cv,
            n_jobs=None,
        )
        search.fit(x_train, y_train, groups=groups)
        best = search.best_estimator_
        best_params = {
            "kernel": policy.kernel,
            "C": float(search.best_params_["svc__C"]),
            "gamma": search.best_params_["svc__gamma"],
        }
        return ModelSelectionResult(
            model=best,
            status="search_ok",
            best_params=best_params,
            cv_strategy=cv.__class__.__name__,
            cv_n_splits=int(cv.get_n_splits(X=x_train, y=y_train, groups=groups)),
            cv_best_score=float(search.best_score_),
        )
    except ValueError:
        baseline_model.fit(x_train, y_train)
        return ModelSelectionResult(
            model=baseline_model,
            status="search_skipped_cv_unavailable",
            best_params=baseline_params,
            cv_strategy=None,
            cv_n_splits=None,
            cv_best_score=None,
        )


def _evaluate_split(model: Any, samples: list[dict[str, Any]]) -> SplitMetrics:
    if not samples:
        return SplitMetrics(sample_count=0, accuracy=0.0, macro_f1=0.0)

    x_split, y_true = _xy(samples)
    y_pred = model.predict(x_split)
    return SplitMetrics(
        sample_count=len(samples),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    )
