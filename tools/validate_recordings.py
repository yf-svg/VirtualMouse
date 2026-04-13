from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import CONFIG
from app.gestures.validation import (
    ValidationPolicy,
    save_validated_dataset,
    validate_recording_files,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate raw gesture recording sessions into a training-ready dataset artifact."
    )
    parser.add_argument(
        "--input-glob",
        default=str(CONFIG.paths.data_dir / "recordings" / "*.json"),
        help="Glob pattern for raw recorder JSON sessions.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=CONFIG.paths.data_dir / "datasets" / "validated_dataset.json",
        help="Path for the persisted validated dataset artifact.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=CONFIG.paths.data_dir / "datasets" / "validation_report.json",
        help="Path for the concise validation summary report.",
    )
    parser.add_argument(
        "--skip-outlier-filter",
        action="store_true",
        help="Skip robust MAD-based outlier filtering.",
    )
    parser.add_argument(
        "--skip-split-assignment",
        action="store_true",
        help="Skip leakage-safe split planning.",
    )
    parser.add_argument(
        "--min-users-for-disjoint-split",
        type=int,
        default=3,
        help="Minimum distinct users required before train/validation/test split assignment is allowed.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = sorted(Path(match) for match in glob.glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No recording files matched: {args.input_glob}")

    policy = ValidationPolicy(
        min_users_for_disjoint_split=max(2, int(args.min_users_for_disjoint_split)),
    )
    dataset = validate_recording_files(
        paths,
        policy=policy,
        assign_splits=not args.skip_split_assignment,
        apply_outlier_filter=not args.skip_outlier_filter,
    )
    dataset_path = save_validated_dataset(dataset, args.output_dataset)
    report_payload = {
        "dataset_path": str(dataset_path),
        "summary": dataset.summary,
        "split_plan": {
            "status": dataset.split_plan.status,
            "cv_strategy": dataset.split_plan.cv_strategy,
            "cv_n_splits": dataset.split_plan.cv_n_splits,
            "issues": [issue.message for issue in dataset.split_plan.issues],
        },
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print(f"[Validation] Validated samples: {dataset.summary['validated_sample_count']}")
    print(f"[Validation] Rejected samples: {dataset.summary['rejected_sample_count']}")
    print(f"[Validation] Rejected sessions: {dataset.summary['rejected_session_count']}")
    print(f"[Validation] Split status: {dataset.summary['split_status']}")
    print(f"[Validation] Dataset saved to {dataset_path}")
    print(f"[Validation] Report saved to {args.output_report}")


if __name__ == "__main__":
    main()
