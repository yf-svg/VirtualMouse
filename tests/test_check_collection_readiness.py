from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.check_collection_readiness import validate_collection_readiness
from tools.generate_dataset_tracker import CaptureCondition, build_tracker_rows, write_tracker_csv


class CheckCollectionReadinessTests(unittest.TestCase):
    def test_readiness_passes_for_generated_unified_tracker_and_matching_protocol(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            protocol = root / "protocol.md"
            tracker = root / "tracker.csv"
            protocol.write_text(
                "round=phase4_v2\nscope=unified\npython tools/generate_dataset_tracker.py --scope unified\n",
                encoding="utf-8",
            )
            rows = build_tracker_rows(
                scope="unified",
                user_ids=("U01", "U02"),
                round_tag="phase4_v2",
                target_samples=60,
                conditions=(
                    CaptureCondition(background="plain", lighting="bright"),
                    CaptureCondition(background="cluttered", lighting="mixed"),
                ),
            )
            write_tracker_csv(tracker, rows)

            report = validate_collection_readiness(
                protocol_path=protocol,
                tracker_path=tracker,
                scope="unified",
                round_tag="phase4_v2",
            )

        self.assertTrue(report.ok)
        self.assertEqual(report.tracker_rows, 84)
        self.assertEqual(report.user_count, 2)
        self.assertEqual(report.condition_count, 2)

    def test_readiness_fails_when_scope_and_round_do_not_match(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            protocol = root / "protocol.md"
            tracker = root / "tracker.csv"
            protocol.write_text("round=phase4_v2\nscope=unified\n", encoding="utf-8")
            rows = build_tracker_rows(
                scope="auth",
                user_ids=("U01",),
                round_tag="phase4_v1",
                target_samples=60,
                conditions=(CaptureCondition(background="plain", lighting="bright"),),
            )
            write_tracker_csv(tracker, rows)

            report = validate_collection_readiness(
                protocol_path=protocol,
                tracker_path=tracker,
                scope="unified",
                round_tag="phase4_v2",
            )

        self.assertFalse(report.ok)
        codes = {issue.code for issue in report.issues}
        self.assertIn("tracker_round_mismatch", codes)
        self.assertIn("tracker_scope_mismatch", codes)


if __name__ == "__main__":
    unittest.main()
