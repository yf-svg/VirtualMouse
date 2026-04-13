from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.generate_dataset_tracker import TRACKER_HEADERS, labels_for_scope, recommended_mode_for_label


ALLOWED_STATUS_VALUES = {"not_started", "in_progress", "completed", "discarded", "redo"}


@dataclass(frozen=True)
class ReadinessIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ReadinessReport:
    ok: bool
    scope: str
    round_tag: str
    tracker_rows: int
    user_count: int
    condition_count: int
    issues: tuple[ReadinessIssue, ...]


def _validate_protocol_text(protocol_text: str, *, scope: str, round_tag: str) -> list[ReadinessIssue]:
    issues: list[ReadinessIssue] = []
    if round_tag not in protocol_text:
        issues.append(ReadinessIssue("protocol_round_missing", f"Protocol does not mention round tag {round_tag!r}"))
    if f"scope={scope}" not in protocol_text and f"`{scope}`" not in protocol_text:
        issues.append(ReadinessIssue("protocol_scope_missing", f"Protocol does not mention scope {scope!r}"))
    if "generate_dataset_tracker.py" not in protocol_text:
        issues.append(ReadinessIssue("protocol_generator_missing", "Protocol does not mention the tracker generator command"))
    return issues


def validate_collection_readiness(
    *,
    protocol_path: Path,
    tracker_path: Path,
    scope: str,
    round_tag: str,
) -> ReadinessReport:
    issues: list[ReadinessIssue] = []

    expected_labels = labels_for_scope(scope)
    expected_label_set = set(expected_labels)

    try:
        protocol_text = protocol_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        issues.append(ReadinessIssue("protocol_missing", f"Protocol file not found: {protocol_path}"))
        protocol_text = ""
    else:
        issues.extend(_validate_protocol_text(protocol_text, scope=scope, round_tag=round_tag))

    rows: list[dict[str, str]] = []
    try:
        with tracker_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(TRACKER_HEADERS):
                issues.append(
                    ReadinessIssue(
                        "tracker_header_mismatch",
                        f"Tracker headers do not match expected schema: {reader.fieldnames!r}",
                    )
                )
            rows = list(reader)
    except FileNotFoundError:
        issues.append(ReadinessIssue("tracker_missing", f"Tracker file not found: {tracker_path}"))

    seen_keys: set[tuple[str, str, str, str, str, str]] = set()
    combinations: dict[tuple[str, str, str], set[str]] = {}
    users: set[str] = set()
    conditions: set[tuple[str, str]] = set()

    for row in rows:
        row_round = row.get("round", "")
        row_scope = row.get("scope", "")
        user_id = row.get("user_id", "")
        background = row.get("background", "")
        lighting = row.get("lighting", "")
        gesture_label = row.get("gesture_label", "")
        status = row.get("status", "")
        recommended_mode = row.get("recommended_mode", "")

        key = (row_round, row_scope, user_id, background, lighting, gesture_label)
        if key in seen_keys:
            issues.append(ReadinessIssue("duplicate_tracker_row", f"Duplicate tracker row for {key!r}"))
        seen_keys.add(key)

        if row_round != round_tag:
            issues.append(
                ReadinessIssue(
                    "tracker_round_mismatch",
                    f"Tracker row has round {row_round!r}; expected {round_tag!r} for label {gesture_label!r}",
                )
            )
        if row_scope != scope:
            issues.append(
                ReadinessIssue(
                    "tracker_scope_mismatch",
                    f"Tracker row has scope {row_scope!r}; expected {scope!r} for label {gesture_label!r}",
                )
            )
        if gesture_label not in expected_label_set:
            issues.append(ReadinessIssue("unknown_tracker_label", f"Tracker contains unexpected label {gesture_label!r}"))
        if recommended_mode != recommended_mode_for_label(gesture_label):
            issues.append(
                ReadinessIssue(
                    "recommended_mode_mismatch",
                    f"Tracker mode for {gesture_label!r} is {recommended_mode!r}; expected {recommended_mode_for_label(gesture_label)!r}",
                )
            )
        if status not in ALLOWED_STATUS_VALUES:
            issues.append(ReadinessIssue("invalid_status", f"Tracker status {status!r} is not allowed"))

        users.add(user_id)
        conditions.add((background, lighting))
        combinations.setdefault((user_id, background, lighting), set()).add(gesture_label)

    for combo, labels in combinations.items():
        if labels != expected_label_set:
            missing = sorted(expected_label_set - labels)
            extra = sorted(labels - expected_label_set)
            message = f"Tracker combination {combo!r} does not match expected labels."
            if missing:
                message += f" Missing: {missing!r}."
            if extra:
                message += f" Extra: {extra!r}."
            issues.append(ReadinessIssue("incomplete_label_matrix", message))

    expected_row_count = len(expected_labels) * len(users) * len(conditions) if users and conditions else 0
    if rows and len(rows) != expected_row_count:
        issues.append(
            ReadinessIssue(
                "tracker_row_count_mismatch",
                f"Tracker has {len(rows)} rows; expected {expected_row_count} for scope={scope!r}",
            )
        )

    return ReadinessReport(
        ok=not issues,
        scope=scope,
        round_tag=round_tag,
        tracker_rows=len(rows),
        user_count=len(users),
        condition_count=len(conditions),
        issues=tuple(issues),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate that the dataset collection protocol and tracker match the canonical gesture registry."
    )
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs" / "dataset_collection_protocol.md")
    parser.add_argument("--tracker", type=Path, default=ROOT / "docs" / "dataset_collection_tracker.csv")
    parser.add_argument("--scope", choices=("auth", "ops", "unified"), default="unified")
    parser.add_argument("--round", dest="round_tag", default="phase4_v2")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = validate_collection_readiness(
        protocol_path=Path(args.protocol),
        tracker_path=Path(args.tracker),
        scope=args.scope,
        round_tag=args.round_tag,
    )
    print(
        f"Readiness check: {'OK' if report.ok else 'FAILED'} | "
        f"scope={report.scope} | round={report.round_tag} | "
        f"rows={report.tracker_rows} | users={report.user_count} | "
        f"conditions={report.condition_count}"
    )
    for issue in report.issues:
        print(f"- {issue.code}: {issue.message}")
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
