from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.config import CONFIG
from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION, FeatureVector
from app.gestures.model_bundle import (
    CANDIDATE_ARTIFACT_KIND,
    RUNTIME_ARTIFACT_KIND,
    missing_runtime_bundle_fields,
)


DEFAULT_MODEL_PATH = CONFIG.paths.models_dir / "gesture_svm.joblib"


@dataclass(frozen=True)
class ClassifierPrediction:
    label: str | None
    confidence: float | None
    accepted: bool
    reason: str
    model_path: str | None = None


class SVMClassifier:
    """
    Runtime-facing classifier wrapper.

    The model path is optional so the runtime can remain import-safe and fall back
    to rule logic before any trained artifact exists.
    """

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        min_confidence: float = 0.70,
        allowed: Iterable[str] | None = None,
        model_bundle: Any | None = None,
    ):
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.min_confidence = float(min_confidence)
        self.allowed = set(allowed) if allowed is not None else None

        self._model: Any | None = None
        self._bundle: dict[str, Any] | None = None
        self._load_reason = "model_unavailable"

        if model_bundle is not None:
            self._ingest_bundle(model_bundle)
        else:
            self._load_default_bundle()

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def load_reason(self) -> str:
        return self._load_reason

    def _load_default_bundle(self) -> None:
        if not self.model_path.exists():
            self._load_reason = "model_unavailable"
            return

        try:
            import joblib
        except Exception:
            self._load_reason = "joblib_unavailable"
            return

        try:
            bundle = joblib.load(self.model_path)
        except Exception as exc:
            self._load_reason = f"model_load_failed:{exc.__class__.__name__}"
            return

        self._ingest_bundle(bundle)

    def _ingest_bundle(self, bundle: Any) -> None:
        if isinstance(bundle, dict):
            artifact_kind = bundle.get("artifact_kind")
            if artifact_kind == CANDIDATE_ARTIFACT_KIND:
                self._bundle = dict(bundle)
                self._model = None
                self._load_reason = "artifact_not_runtime_approved"
                return
            if artifact_kind == RUNTIME_ARTIFACT_KIND:
                missing_fields = missing_runtime_bundle_fields(bundle)
                if missing_fields:
                    self._bundle = dict(bundle)
                    self._model = None
                    self._load_reason = "invalid_model_bundle"
                    return
            self._bundle = dict(bundle)
            self._model = bundle.get("model")
            if bundle.get("min_confidence") is not None:
                self.min_confidence = float(bundle["min_confidence"])
        else:
            self._bundle = None
            self._model = bundle

        if self._model is None:
            self._load_reason = "invalid_model_bundle"
        else:
            self._load_reason = "ready"

    def _bundle_schema_version(self) -> str | None:
        if self._bundle is None:
            return None
        return self._bundle.get("schema_version")

    def _bundle_dimension(self) -> int | None:
        if self._bundle is None:
            return None
        value = self._bundle.get("feature_dimension")
        return int(value) if value is not None else None

    def _class_names(self) -> list[str] | None:
        if self._bundle is not None and self._bundle.get("labels") is not None:
            return [str(label) for label in self._bundle["labels"]]

        classes = getattr(self._model, "classes_", None)
        if classes is None:
            return None
        return [str(label) for label in classes]

    def predict(self, feature_vector: FeatureVector | None) -> ClassifierPrediction:
        if feature_vector is None:
            return ClassifierPrediction(
                label=None,
                confidence=None,
                accepted=False,
                reason="no_features",
                model_path=str(self.model_path),
            )

        if not self.available:
            return ClassifierPrediction(
                label=None,
                confidence=None,
                accepted=False,
                reason=self._load_reason,
                model_path=str(self.model_path),
            )

        bundle_schema = self._bundle_schema_version()
        if bundle_schema is not None and bundle_schema != feature_vector.schema_version:
            return ClassifierPrediction(
                label=None,
                confidence=None,
                accepted=False,
                reason="model_schema_mismatch",
                model_path=str(self.model_path),
            )

        bundle_dimension = self._bundle_dimension()
        if bundle_dimension is not None and bundle_dimension != feature_vector.dimension:
            return ClassifierPrediction(
                label=None,
                confidence=None,
                accepted=False,
                reason="model_dimension_mismatch",
                model_path=str(self.model_path),
            )

        if feature_vector.schema_version != FEATURE_SCHEMA_VERSION:
            return ClassifierPrediction(
                label=None,
                confidence=None,
                accepted=False,
                reason="feature_schema_mismatch",
                model_path=str(self.model_path),
            )

        if feature_vector.dimension != FEATURE_DIMENSION:
            return ClassifierPrediction(
                label=None,
                confidence=None,
                accepted=False,
                reason="feature_dimension_mismatch",
                model_path=str(self.model_path),
            )

        if not hasattr(self._model, "predict") and not hasattr(self._model, "predict_proba"):
            return ClassifierPrediction(
                label=None,
                confidence=None,
                accepted=False,
                reason="invalid_model",
                model_path=str(self.model_path),
            )

        row = [list(feature_vector.values)]
        predicted_label: str | None = None
        confidence: float | None = None

        try:
            if hasattr(self._model, "predict_proba"):
                probs = self._model.predict_proba(row)[0]
                labels = self._class_names()
                if labels is None or len(labels) != len(probs):
                    return ClassifierPrediction(
                        label=None,
                        confidence=None,
                        accepted=False,
                        reason="model_classes_missing",
                        model_path=str(self.model_path),
                    )
                best_idx = max(range(len(probs)), key=lambda idx: probs[idx])
                predicted_label = labels[best_idx]
                confidence = float(probs[best_idx])
            else:
                predicted = self._model.predict(row)
                predicted_label = str(predicted[0]) if predicted else None
        except Exception as exc:
            return ClassifierPrediction(
                label=None,
                confidence=None,
                accepted=False,
                reason=f"prediction_failed:{exc.__class__.__name__}",
                model_path=str(self.model_path),
            )

        if predicted_label is None:
            return ClassifierPrediction(
                label=None,
                confidence=confidence,
                accepted=False,
                reason="prediction_empty",
                model_path=str(self.model_path),
            )

        if self.allowed is not None and predicted_label not in self.allowed:
            return ClassifierPrediction(
                label=predicted_label,
                confidence=confidence,
                accepted=False,
                reason="label_not_allowed",
                model_path=str(self.model_path),
            )

        if confidence is None:
            return ClassifierPrediction(
                label=predicted_label,
                confidence=None,
                accepted=False,
                reason="confidence_unavailable",
                model_path=str(self.model_path),
            )

        if confidence < self.min_confidence:
            return ClassifierPrediction(
                label=predicted_label,
                confidence=confidence,
                accepted=False,
                reason="low_confidence",
                model_path=str(self.model_path),
            )

        return ClassifierPrediction(
            label=predicted_label,
            confidence=confidence,
            accepted=True,
            reason="accepted",
            model_path=str(self.model_path),
        )
