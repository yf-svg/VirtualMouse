from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import CONFIG
from app.gestures.model_bundle import promote_runtime_model_bundle


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote a validated runtime model bundle into the live runtime model path."
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        required=True,
        help="Path to an exported runtime_model_bundle artifact.",
    )
    parser.add_argument(
        "--live-model",
        type=Path,
        default=CONFIG.paths.models_dir / "gesture_svm.joblib",
        help="Destination path for the live runtime model.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=CONFIG.paths.models_dir / "archive",
        help="Directory for automatic backups when replacing an existing live model.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = promote_runtime_model_bundle(
        args.source_model,
        live_model_path=args.live_model,
        backup_dir=args.backup_dir,
    )
    print(f"[Promote] Live model: {result.live_model_path}")
    print(f"[Promote] Replaced existing: {'yes' if result.replaced_existing else 'no'}")
    if result.backup_model_path is not None:
        print(f"[Promote] Backup: {result.backup_model_path}")


if __name__ == "__main__":
    main()
