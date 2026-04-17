from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from tools.generate_dataset_tracker import (
    CaptureCondition,
    build_tracker_rows,
    labels_for_scope,
    parse_condition,
    recommended_mode_for_label,
    write_tracker_csv,
)


class GenerateDatasetTrackerTests(unittest.TestCase):
    def test_labels_for_scope_matches_expected_counts(self):
        self.assertEqual(len(labels_for_scope("auth")), 9)
        self.assertEqual(len(labels_for_scope("ops")), 16)
        self.assertEqual(len(labels_for_scope("unified")), 21)

    def test_recommended_mode_marks_pinch_labels_manual(self):
        self.assertEqual(recommended_mode_for_label("PINCH_INDEX"), "MANUAL")
        self.assertEqual(recommended_mode_for_label("FIST"), "AUTO")

    def test_parse_condition_requires_background_and_lighting(self):
        condition = parse_condition("background=plain,lighting=bright")
        self.assertEqual(condition, CaptureCondition(background="plain", lighting="bright"))
        with self.assertRaises(ValueError):
            parse_condition("background=plain")

    def test_build_tracker_rows_generates_scope_and_mode_fields(self):
        rows = build_tracker_rows(
            scope="auth",
            user_ids=("U01",),
            round_tag="phase4_v2",
            target_samples=60,
            conditions=(CaptureCondition(background="plain", lighting="bright"),),
        )
        self.assertEqual(len(rows), 9)
        self.assertEqual(rows[0]["scope"], "auth")
        self.assertEqual(rows[0]["round"], "phase4_v2")
        self.assertEqual(rows[0]["status"], "not_started")
        pinch_rows = build_tracker_rows(
            scope="ops",
            user_ids=("U01",),
            round_tag="phase4_v2",
            target_samples=60,
            conditions=(CaptureCondition(background="plain", lighting="bright"),),
        )
        pinch_index = next(row for row in pinch_rows if row["gesture_label"] == "PINCH_INDEX")
        self.assertEqual(pinch_index["recommended_mode"], "MANUAL")

    def test_write_tracker_csv_writes_expected_header_and_rows(self):
        rows = build_tracker_rows(
            scope="unified",
            user_ids=("U01",),
            round_tag="phase4_v2",
            target_samples=60,
            conditions=(CaptureCondition(background="plain", lighting="bright"),),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tracker.csv"
            write_tracker_csv(path, rows)
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                loaded = list(reader)

        self.assertEqual(len(loaded), 21)
        self.assertEqual(reader.fieldnames[0:3], ["round", "scope", "user_id"])
        self.assertEqual(loaded[0]["scope"], "unified")


if __name__ == "__main__":
    unittest.main()
