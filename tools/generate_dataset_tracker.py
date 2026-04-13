from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gestures.sets.labels import AUTH_LABEL_ORDER, OPS_LABEL_ORDER, UNIFIED_LABEL_ORDER


TRACKER_HEADERS = (
    "round",
    "scope",
    "user_id",
    "background",
    "lighting",
    "gesture_label",
    "target_samples",
    "accepted_samples",
    "rejected_attempts",
    "recommended_mode",
    "status",
    "notes",
    "file_path",
)

DEFAULT_OUTPUT = ROOT / "docs" / "dataset_collection_tracker.csv"
DEFAULT_CONDITIONS = (
    ("plain", "bright"),
    ("cluttered", "mixed"),
)


@dataclass(frozen=True)
class CaptureCondition:
    background: str
    lighting: str


def labels_for_scope(scope: str) -> tuple[str, ...]:
    normalized = scope.strip().lower()
    if normalized == "auth":
        return AUTH_LABEL_ORDER
    if normalized == "ops":
        return OPS_LABEL_ORDER
    if normalized == "unified":
        return UNIFIED_LABEL_ORDER
    raise ValueError(f"Unsupported scope: {scope}")


def recommended_mode_for_label(label: str) -> str:
    return "MANUAL" if label.startswith("PINCH_") else "AUTO"


def parse_condition(spec: str) -> CaptureCondition:
    pairs: dict[str, str] = {}
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Condition item must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        pairs[key.strip().lower()] = value.strip()
    background = pairs.get("background")
    lighting = pairs.get("lighting")
    if not background or not lighting:
        raise ValueError(f"Condition must define background and lighting, got {spec!r}")
    return CaptureCondition(background=background, lighting=lighting)


def build_tracker_rows(
    *,
    scope: str,
    user_ids: tuple[str, ...],
    round_tag: str,
    target_samples: int,
    conditions: tuple[CaptureCondition, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for user_id in user_ids:
        for condition in conditions:
            for label in labels_for_scope(scope):
                rows.append(
                    {
                        "round": round_tag,
                        "scope": scope,
                        "user_id": user_id,
                        "background": condition.background,
                        "lighting": condition.lighting,
                        "gesture_label": label,
                        "target_samples": str(target_samples),
                        "accepted_samples": "0",
                        "rejected_attempts": "0",
                        "recommended_mode": recommended_mode_for_label(label),
                        "status": "not_started",
                        "notes": "",
                        "file_path": "",
                    }
                )
    return rows


def write_tracker_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACKER_HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a dataset collection tracker CSV from the canonical gesture label registry."
    )
    parser.add_argument("--scope", choices=("auth", "ops", "unified"), default="unified")
    parser.add_argument("--user-id", action="append", dest="user_ids", default=["U01", "U02"])
    parser.add_argument("--round", dest="round_tag", default="phase4_v2")
    parser.add_argument("--target-samples", type=int, default=60)
    parser.add_argument(
        "--condition",
        action="append",
        dest="conditions",
        default=[],
        help="Repeatable capture condition in the form background=plain,lighting=bright",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    conditions = (
        tuple(parse_condition(spec) for spec in args.conditions)
        if args.conditions
        else tuple(CaptureCondition(background=bg, lighting=light) for bg, light in DEFAULT_CONDITIONS)
    )
    user_ids = tuple(dict.fromkeys(args.user_ids))
    rows = build_tracker_rows(
        scope=args.scope,
        user_ids=user_ids,
        round_tag=args.round_tag,
        target_samples=int(args.target_samples),
        conditions=conditions,
    )
    write_tracker_csv(Path(args.output), rows)
    print(f"Wrote {len(rows)} tracker rows to {Path(args.output)}")


if __name__ == "__main__":
    main()
