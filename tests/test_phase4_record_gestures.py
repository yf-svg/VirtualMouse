from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.check_collection_readiness import ReadinessIssue, ReadinessReport
from tools.record_gestures import (
    CaptureMode,
    RecorderConfig,
    RecorderRuntime,
    RecorderState,
    RecordingSample,
    RecordingSession,
    _build_session_summary,
    _advance_countdown,
    _countdown_remaining,
    _format_duration,
    _handedness_summary,
    _humanize_quality_reason,
    _maybe_validate_readiness,
    _progress_text,
    _record_rejection,
    _sound_asset_path,
    _start_countdown,
    _set_last_result,
    _toggle_capture_mode,
    _update_state,
    build_output_path,
    parse_capture_context,
    save_session,
    session_to_payload,
)


class RecordGesturesToolTests(unittest.TestCase):
    def test_parse_capture_context_supports_repeated_pairs(self):
        context = parse_capture_context(
            ["background=plain", "lighting=bright", "background=cluttered"]
        )
        self.assertEqual(
            context,
            {"background": "cluttered", "lighting": "bright"},
        )

    def test_parse_capture_context_rejects_invalid_items(self):
        with self.assertRaises(ValueError):
            parse_capture_context(["missing-separator"])

    def test_readiness_check_skips_without_scope_and_round(self):
        with patch("tools.record_gestures.validate_collection_readiness") as validate:
            report = _maybe_validate_readiness(capture_context={"background": "plain"})
        self.assertIsNone(report)
        validate.assert_not_called()

    def test_readiness_check_runs_when_scope_and_round_present(self):
        readiness = ReadinessReport(
            ok=True,
            scope="unified",
            round_tag="phase4_v2",
            tracker_rows=80,
            user_count=2,
            condition_count=2,
            issues=(),
        )
        with patch("tools.record_gestures.validate_collection_readiness", return_value=readiness) as validate:
            report = _maybe_validate_readiness(
                capture_context={"scope": "unified", "round": "phase4_v2"},
            )
        self.assertEqual(report, readiness)
        validate.assert_called_once()

    def test_readiness_check_can_be_skipped_explicitly(self):
        with patch("tools.record_gestures.validate_collection_readiness") as validate:
            report = _maybe_validate_readiness(
                capture_context={"scope": "unified", "round": "phase4_v2"},
                skip_check=True,
            )
        self.assertIsNone(report)
        validate.assert_not_called()

    def test_readiness_check_aborts_on_failed_report(self):
        readiness = ReadinessReport(
            ok=False,
            scope="unified",
            round_tag="phase4_v2",
            tracker_rows=0,
            user_count=0,
            condition_count=0,
            issues=(ReadinessIssue("tracker_missing", "Tracker file not found"),),
        )
        with patch("tools.record_gestures.validate_collection_readiness", return_value=readiness):
            with self.assertRaises(SystemExit) as ctx:
                _maybe_validate_readiness(
                    capture_context={"scope": "unified", "round": "phase4_v2"},
                )
        self.assertIn("Dataset readiness check failed", str(ctx.exception))
        self.assertIn("tracker_missing", str(ctx.exception))

    def test_build_output_path_sanitizes_filename_parts(self):
        path = build_output_path(
            "PEACE SIGN",
            "user 01",
            "session/alpha",
            base_dir=Path("recordings"),
        )
        self.assertEqual(
            path.as_posix(),
            "recordings/PEACE-SIGN__user-01__session-alpha.json",
        )

    def test_session_payload_includes_sample_count_and_metadata(self):
        session = RecordingSession(
            gesture_label="FIST",
            user_id="user_a",
            session_id="sess_001",
            schema_version="phase3.v2",
            feature_dimension=92,
            capture_context={"background": "plain"},
            created_at="2026-04-08T21:00:00",
            samples=[
                RecordingSample(
                    sample_index=0,
                    frame_seq=12,
                    captured_at=1.23,
                    gesture_label="FIST",
                    user_id="user_a",
                    session_id="sess_001",
                    handedness="Right",
                    schema_version="phase3.v2",
                    quality_reason="ok",
                    quality_scale=1.0,
                    quality_palm_width=1.0,
                    quality_bbox_width=1.0,
                    quality_bbox_height=1.0,
                    feature_values=[0.1, 0.2],
                )
            ],
        )
        payload = session_to_payload(session)
        self.assertEqual(payload["sample_count"], 1)
        self.assertEqual(payload["capture_context"]["background"], "plain")
        self.assertEqual(payload["samples"][0]["feature_values"], [0.1, 0.2])

    def test_save_session_writes_json_file(self):
        session = RecordingSession(
            gesture_label="BRAVO",
            user_id="user_b",
            session_id="sess_010",
            schema_version="phase3.v2",
            feature_dimension=92,
            capture_context={},
            created_at="2026-04-08T21:05:00",
            samples=[],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "recording.json"
            save_session(session, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["gesture_label"], "BRAVO")
        self.assertEqual(payload["sample_count"], 0)

    def test_humanize_quality_reason_maps_known_states(self):
        self.assertEqual(_humanize_quality_reason("ok", True), "Ready to capture")
        self.assertEqual(_humanize_quality_reason("hand_too_small", False), "Move hand closer to the camera")
        self.assertEqual(_humanize_quality_reason("no_hand", False), "No hand detected")

    def test_progress_text_formats_optional_target(self):
        self.assertEqual(_progress_text(12, None), "12")
        self.assertEqual(_progress_text(12, 100), "12 / 100")

    def test_runtime_helpers_update_state_and_result(self):
        runtime = RecorderRuntime()
        _update_state(runtime, RecorderState.RECORDING_AUTO)
        _set_last_result(runtime, "Accepted sample #3", level="accept")
        self.assertEqual(runtime.state, RecorderState.RECORDING_AUTO)
        self.assertEqual(runtime.last_result, "Accepted sample #3")
        self.assertEqual(runtime.last_result_level, "accept")

    def test_record_rejection_updates_counter_and_message(self):
        runtime = RecorderRuntime()
        _record_rejection(runtime, "bbox_too_small")
        self.assertEqual(runtime.rejected_attempts, 1)
        self.assertIn("Center your hand in frame", runtime.last_result)
        self.assertEqual(runtime.last_result_level, "reject")

    def test_start_countdown_sets_runtime_state(self):
        runtime = RecorderRuntime()
        cfg = RecorderConfig(
            gesture_label="FIST",
            user_id="user01",
            session_id="sess01",
            output_path=Path("out.json"),
            countdown_seconds=3.0,
            sound_enabled=False,
        )
        _start_countdown(runtime, cfg, now=10.0)
        self.assertEqual(runtime.state, RecorderState.COUNTDOWN)
        self.assertEqual(runtime.countdown_started_at, 10.0)
        self.assertEqual(runtime.last_result, "Recording starts in 3")

    def test_countdown_remaining_rounds_up(self):
        runtime = RecorderRuntime(state=RecorderState.COUNTDOWN, countdown_started_at=5.0)
        cfg = RecorderConfig(
            gesture_label="FIST",
            user_id="user01",
            session_id="sess01",
            output_path=Path("out.json"),
            countdown_seconds=3.0,
            sound_enabled=False,
        )
        self.assertEqual(_countdown_remaining(runtime, cfg, now=5.1), 3)
        self.assertEqual(_countdown_remaining(runtime, cfg, now=7.2), 1)
        self.assertEqual(_countdown_remaining(runtime, cfg, now=8.0), 0)

    def test_advance_countdown_transitions_to_recording(self):
        runtime = RecorderRuntime(state=RecorderState.COUNTDOWN, countdown_started_at=2.0)
        cfg = RecorderConfig(
            gesture_label="FIST",
            user_id="user01",
            session_id="sess01",
            output_path=Path("out.json"),
            countdown_seconds=3.0,
            sound_enabled=False,
        )
        advanced = _advance_countdown(runtime, cfg, now=5.1)
        self.assertTrue(advanced)
        self.assertEqual(runtime.state, RecorderState.RECORDING_AUTO)
        self.assertEqual(runtime.last_result, "Auto capture started")

    def test_sound_asset_path_points_to_repo_backup_sounds(self):
        path = _sound_asset_path("save")
        self.assertEqual(path.name, "save.wav")
        self.assertIn("assets", path.parts)
        self.assertIn("sounds", path.parts)

    def test_toggle_capture_mode_switches_to_manual_and_pauses(self):
        runtime = RecorderRuntime(state=RecorderState.RECORDING_AUTO, capture_mode=CaptureMode.AUTO)
        mode = _toggle_capture_mode(runtime)
        self.assertEqual(mode, CaptureMode.MANUAL)
        self.assertEqual(runtime.capture_mode, CaptureMode.MANUAL)
        self.assertEqual(runtime.state, RecorderState.PAUSED)

    def test_toggle_capture_mode_switches_back_to_auto(self):
        runtime = RecorderRuntime(state=RecorderState.RECORDING_MANUAL, capture_mode=CaptureMode.MANUAL)
        mode = _toggle_capture_mode(runtime)
        self.assertEqual(mode, CaptureMode.AUTO)
        self.assertEqual(runtime.capture_mode, CaptureMode.AUTO)
        self.assertEqual(runtime.state, RecorderState.PAUSED)

    def test_format_duration_formats_minutes_and_hours(self):
        self.assertEqual(_format_duration(61), "01:01")
        self.assertEqual(_format_duration(3661), "01:01:01")

    def test_handedness_summary_counts_known_and_unknown(self):
        session = RecordingSession(
            gesture_label="FIST",
            user_id="user_a",
            session_id="sess_001",
            schema_version="phase3.v2",
            feature_dimension=92,
            capture_context={},
            created_at="2026-04-10T21:00:00",
            samples=[
                RecordingSample(0, 1, 1.0, "FIST", "user_a", "sess_001", "Right", "phase3.v2", "ok", 1.0, 1.0, 1.0, 1.0, [0.1]),
                RecordingSample(1, 2, 2.0, "FIST", "user_a", "sess_001", "Left", "phase3.v2", "ok", 1.0, 1.0, 1.0, 1.0, [0.2]),
                RecordingSample(2, 3, 3.0, "FIST", "user_a", "sess_001", None, "phase3.v2", "ok", 1.0, 1.0, 1.0, 1.0, [0.3]),
            ],
        )
        self.assertEqual(_handedness_summary(session), "L:1 R:1 U:1")

    def test_build_session_summary_includes_dataset_relevant_fields(self):
        session = RecordingSession(
            gesture_label="BRAVO",
            user_id="user_b",
            session_id="sess_010",
            schema_version="phase3.v2",
            feature_dimension=92,
            capture_context={},
            created_at="2026-04-10T21:05:00",
            samples=[
                RecordingSample(0, 12, 1.23, "BRAVO", "user_b", "sess_010", "Right", "phase3.v2", "ok", 1.0, 1.0, 1.0, 1.0, [0.1, 0.2]),
            ],
        )
        runtime = RecorderRuntime(accepted_samples=1, rejected_attempts=4, started_at_monotonic=10.0)
        cfg = RecorderConfig(
            gesture_label="BRAVO",
            user_id="user_b",
            session_id="sess_010",
            output_path=Path("data/recordings/bravo.json"),
            sound_enabled=False,
        )
        summary = _build_session_summary(session, runtime, cfg, now=74.0)
        joined = "\n".join(summary)
        self.assertIn("Gesture: BRAVO", joined)
        self.assertIn("Accepted: 1", joined)
        self.assertIn("Rejected: 4", joined)
        self.assertIn("Duration: 01:04", joined)
        self.assertIn("Handedness: L:0 R:1 U:0", joined)
        self.assertIn("Output:", joined)


if __name__ == "__main__":
    unittest.main()
