from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from app.config import CONFIG
from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA_VERSION


CANDIDATE_ARTIFACT_KIND = "training_candidate"
RUNTIME_ARTIFACT_KIND = "runtime_model_bundle"
RUNTIME_BUNDLE_VERSION = "phase4.runtime.v1"

REQUIRED_RUNTIME_BUNDLE_FIELDS = (
    "artifact_kind",
    "bundle_version",
    "trainer_version",
    "trained_at",
    "schema_version",
    "feature_dimension",
    "labels",
    "min_confidence",
    "model",
)


@dataclass(frozen=True)
class RuntimeModelPromotionResult:
    source_model_path: str
    live_model_path: str
    backup_model_path: str | None
    replaced_existing: bool


def missing_runtime_bundle_fields(bundle: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field_name in REQUIRED_RUNTIME_BUNDLE_FIELDS:
        if bundle.get(field_name) is None:
            missing.append(field_name)
    return missing


def load_runtime_bundle_payload(path: str | Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError("Runtime model bundle must be a dict payload.")
    return payload


def validate_runtime_bundle_payload(bundle: dict[str, Any]) -> None:
    if bundle.get("artifact_kind") != RUNTIME_ARTIFACT_KIND:
        raise ValueError("Live runtime model must be a 'runtime_model_bundle' artifact.")

    missing_fields = missing_runtime_bundle_fields(bundle)
    if missing_fields:
        raise ValueError(f"Runtime model bundle is missing required fields: {', '.join(missing_fields)}")

    if bundle.get("bundle_version") != RUNTIME_BUNDLE_VERSION:
        raise ValueError(
            f"Runtime model bundle version must be {RUNTIME_BUNDLE_VERSION!r}; got {bundle.get('bundle_version')!r}"
        )

    if bundle.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Runtime model bundle schema must be {FEATURE_SCHEMA_VERSION!r}; got {bundle.get('schema_version')!r}"
        )

    if int(bundle.get("feature_dimension", -1)) != FEATURE_DIMENSION:
        raise ValueError(
            f"Runtime model feature dimension must be {FEATURE_DIMENSION}; got {bundle.get('feature_dimension')!r}"
        )


def promote_runtime_model_bundle(
    source_model_path: str | Path,
    *,
    live_model_path: str | Path | None = None,
    backup_dir: str | Path | None = None,
) -> RuntimeModelPromotionResult:
    source_model_path = Path(source_model_path)
    live_model_path = Path(live_model_path) if live_model_path is not None else CONFIG.paths.models_dir / "gesture_svm.joblib"
    backup_dir = Path(backup_dir) if backup_dir is not None else CONFIG.paths.models_dir / "archive"

    bundle = load_runtime_bundle_payload(source_model_path)
    validate_runtime_bundle_payload(bundle)

    source_resolved = source_model_path.resolve()
    live_resolved = live_model_path.resolve()
    if source_resolved == live_resolved:
        return RuntimeModelPromotionResult(
            source_model_path=str(source_model_path),
            live_model_path=str(live_model_path),
            backup_model_path=None,
            replaced_existing=live_model_path.exists(),
        )

    live_model_path.parent.mkdir(parents=True, exist_ok=True)
    backup_model_path: Path | None = None
    replaced_existing = live_model_path.exists()
    if replaced_existing:
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_model_path = backup_dir / f"{live_model_path.stem}_{stamp}{live_model_path.suffix}"
        shutil.copy2(live_model_path, backup_model_path)

    shutil.copy2(source_model_path, live_model_path)
    return RuntimeModelPromotionResult(
        source_model_path=str(source_model_path),
        live_model_path=str(live_model_path),
        backup_model_path=str(backup_model_path) if backup_model_path is not None else None,
        replaced_existing=replaced_existing,
    )
