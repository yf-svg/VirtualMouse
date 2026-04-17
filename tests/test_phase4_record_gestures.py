from __future__ import annotations

import json
import tempfile
import unittest
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.check_collection_readiness import ReadinessIssue, ReadinessReport
from tools.record_gestures import (
    CaptureMode,
    RecorderArtifacts,
    RecorderConfig,
    RecorderGuidance,
    RecorderRuntime,
    RecorderState,
    RecordingSample,
    RecordingSession,
    _build_recorder_guidance,
    _build_session_summary,
    _ensure_output_path_available,
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
    _sync_tracker_row,
    _toggle_capture_mode,
    _undo_last_sample,
    _update_state,
    _validate_gesture_label,
    _discard_session,
    build_output_path,
    parse_capture_context,
    save_session,
    session_to_payload,
)


class RecordGesturesToolTests(unittest.TestCase):
    def test_validate_gesture_label_rejects_unknown_label(self):
        self.assertEqual(_validate_gesture_label("shaka"), "SHAKA")
        with self.assertRaises(SystemExit):
            _validate_gesture_label("NOT_A_LABEL")

    def test_output_path_guard_rejects_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "session.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                _ensure_output_path_available(path, overwrite=False)
            _ensure_output_path_available(path, overwrite=True)

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

    def test_save_session_writes_compact_landmark_sidecar(self):
        session = RecordingSession(
            gesture_label="PINCH_INDEX",
            user_id="user_b",
            session_id="sess_011",
            schema_version="phase3.v2",
            feature_dimension=92,
            capture_context={},
            created_at="2026-04-17T10:00:00",
            samples=[
                RecordingSample(
                    sample_index=0,
                    frame_seq=12,
                    captured_at=1.23,
                    gesture_label="PINCH_INDEX",
                    user_id="user_b",
                    session_id="sess_011",
                    handedness="Right",
                    schema_version="phase3.v2",
                    quality_reason="ok",
                    quality_scale=1.0,
                    quality_palm_width=1.0,
                    quality_bbox_width=1.0,
                    quality_bbox_height=1.0,
                    feature_values=[0.1, 0.2],
                    raw_landmark_index=0,
                )
            ],
        )
        cfg = RecorderConfig(
            gesture_label="PINCH_INDEX",
            user_id="user_b",
            session_id="sess_011",
            output_path=Path("ignored.json"),
            sound_enabled=False,
        )
        artifacts = RecorderArtifacts(
            raw_landmarks=[np.asarray([[0.1, 0.2, 0.3]] * 21, dtype=np.float16)],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "recording.json"
            save_session(session, output_path, cfg=cfg, artifacts=artifacts)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            landmark_path = output_path.with_suffix(".landmarks.npz")
            with np.load(landmark_path) as archive:
                landmark_shape = tuple(archive["landmarks"].shape)
                landmark_dtype = str(archive["landmarks"].dtype)

        self.assertIn("artifacts", payload)
        self.assertIn("raw_landmarks", payload["artifacts"])
        self.assertEqual(payload["artifacts"]["raw_landmarks"]["path"], "recording.landmarks.npz")
        self.assertEqual(payload["samples"][0]["raw_landmark_index"], 0)
        self.assertEqual(landmark_shape, (1, 21, 3))
        self.assertEqual(landmark_dtype, "float16")

    def test_save_session_writes_optional_snapshot_sidecars(self):
        session = RecordingSession(
            gesture_label="SHAKA",
            user_id="user_c",
            session_id="sess_012",
            schema_version="phase3.v2",
            feature_dimension=92,
            capture_context={},
            created_at="2026-04-17T10:05:00",
            samples=[
                RecordingSample(
                    sample_index=0,
                    frame_seq=3,
                    captured_at=1.0,
                    gesture_label="SHAKA",
                    user_id="user_c",
                    session_id="sess_012",
                    handedness="Right",
                    schema_version="phase3.v2",
                    quality_reason="ok",
                    quality_scale=1.0,
                    quality_palm_width=1.0,
                    quality_bbox_width=1.0,
                    quality_bbox_height=1.0,
                    feature_values=[0.1, 0.2],
                    snapshot_relpath="0000.jpg",
                )
            ],
        )
        cfg = RecorderConfig(
            gesture_label="SHAKA",
            user_id="user_c",
            session_id="sess_012",
            output_path=Path("ignored.json"),
            sound_enabled=False,
            save_snapshots=True,
        )
        artifacts = RecorderArtifacts(snapshot_blobs={0: b"fake-jpeg"})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "recording.json"
            save_session(session, output_path, cfg=cfg, artifacts=artifacts)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            snapshot_path = output_path.parent / f"{output_path.stem}__snapshots" / "0000.jpg"
            snapshot_exists = snapshot_path.exists()
            snapshot_bytes = snapshot_path.read_bytes()

        self.assertIn("snapshots", payload["artifacts"])
        self.assertEqual(payload["artifacts"]["snapshots"]["directory"], "recording__snapshots")
        self.assertTrue(snapshot_exists)
        self.assertEqual(snapshot_bytes, b"fake-jpeg")

    def test_humanize_quality_reason_maps_known_states(self):
        self.assertEqual(_humanize_quality_reason("ok", True), "Ready to capture")
        self.assertEqual(_humanize_quality_reason("hand_too_small", False), "Move hand closer to the camera")
        self.assertEqual(_humanize_quality_reason("no_hand", False), "No hand detected")
        self.assertEqual(
            _humanize_quality_reason("label_hold_pending:PINCH_INDEX", False),
            "Hold PINCH_INDEX steady until confirmed",
        )
        self.assertEqual(
            _humanize_quality_reason("label_mismatch:OPEN_PALM", False),
            "Target mismatch: saw OPEN_PALM",
        )

    def test_build_recorder_guidance_requires_eligible_target_by_default(self):
        suite_out = type(
            "SuiteOut",
            (),
            {
                "chosen": "PINCH_INDEX",
                "stable": "PINCH_INDEX",
                "eligible": None,
                "source": "rules",
            },
        )()
        guidance = _build_recorder_guidance(
            target_label="PINCH_INDEX",
            suite_out=suite_out,
            require_label_match=True,
        )
        self.assertFalse(guidance.capture_ready)
        self.assertEqual(guidance.gate_reason, "label_hold_pending:PINCH_INDEX")

        ready = _build_recorder_guidance(
            target_label="PINCH_INDEX",
            suite_out=type(
                "SuiteOut",
                (),
                {
                    "chosen": "PINCH_INDEX",
                    "stable": "PINCH_INDEX",
                    "eligible": "PINCH_INDEX",
                    "source": "rules",
                },
            )(),
            require_label_match=True,
        )
        self.assertTrue(ready.capture_ready)
        self.assertEqual(ready.gate_reason, "ok")

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
        _record_rejection(runtime, "label_mismatch:OPEN_PALM")
        self.assertEqual(runtime.label_rejections, 1)

    def test_undo_last_sample_keeps_artifacts_in_sync(self):
        session = RecordingSession(
            gesture_label="PINCH_INDEX",
            user_id="U01",
            session_id="sess_sync",
            schema_version="phase3.v2",
            feature_dimension=92,
            capture_context={},
            created_at="2026-04-17T10:30:00",
            samples=[
                RecordingSample(0, 1, 1.0, "PINCH_INDEX", "U01", "sess_sync", "Right", "phase3.v2", "ok", 1.0, 1.0, 1.0, 1.0, [0.1], raw_landmark_index=0),
                RecordingSample(1, 2, 2.0, "PINCH_INDEX", "U01", "sess_sync", "Right", "phase3.v2", "ok", 1.0, 1.0, 1.0, 1.0, [0.2], raw_landmark_index=1, snapshot_relpath="0001.jpg"),
            ],
        )
        artifacts = RecorderArtifacts(
            raw_landmarks=[
                np.asarray([[0.1, 0.1, 0.1]] * 21, dtype=np.float16),
                np.asarray([[0.2, 0.2, 0.2]] * 21, dtype=np.float16),
            ],
            snapshot_blobs={1: b"blob"},
        )

        removed = _undo_last_sample(session, artifacts)

        self.assertIsNotNone(removed)
        self.assertEqual(len(session.samples), 1)
        self.assertEqual(len(artifacts.raw_landmarks), 1)
        self.assertEqual(artifacts.snapshot_blobs, {})

    def test_discard_session_clears_samples_and_artifacts(self):
        session = RecordingSession(
            gesture_label="SHAKA",
            user_id="U01",
            session_id="sess_clear",
            schema_version="phase3.v2",
            feature_dimension=92,
            capture_context={},
            created_at="2026-04-17T10:31:00",
            artifacts={"raw_landmarks": {"path": "demo.landmarks.npz"}},
            samples=[
                RecordingSample(0, 1, 1.0, "SHAKA", "U01", "sess_clear", "Right", "phase3.v2", "ok", 1.0, 1.0, 1.0, 1.0, [0.1], raw_landmark_index=0),
            ],
        )
        artifacts = RecorderArtifacts(
            raw_landmarks=[np.asarray([[0.1, 0.1, 0.1]] * 21, dtype=np.float16)],
            snapshot_blobs={0: b"blob"},
        )

        _discard_session(session, artifacts)

        self.assertEqual(session.samples, [])
        self.assertEqual(session.artifacts, {})
        self.assertEqual(artifacts.raw_landmarks, [])
        self.assertEqual(artifacts.snapshot_blobs, {})

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
        self.assertIn("Acceptance: 1/5 (20.0%)", joined)
        self.assertIn("Label rejects: 0", joined)
        self.assertIn("Duration: 01:04", joined)
        self.assertIn("Handedness: L:0 R:1 U:0", joined)
        self.assertIn("Output:", joined)

    def test_sync_tracker_row_updates_matching_entry_on_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker_path = Path(tmpdir) / "tracker.csv"
            with tracker_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "round", "scope", "user_id", "background", "lighting", "gesture_label",
                        "target_samples", "accepted_samples", "rejected_attempts",
                        "recommended_mode", "status", "notes", "file_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "round": "phase4_v2",
                        "scope": "unified",
                        "user_id": "U01",
                        "background": "plain",
                        "lighting": "bright",
                        "gesture_label": "SHAKA",
                        "target_samples": "60",
                        "accepted_samples": "0",
                        "rejected_attempts": "0",
                        "recommended_mode": "AUTO",
                        "status": "not_started",
                        "notes": "",
                        "file_path": "",
                    }
                )

            cfg = RecorderConfig(
                gesture_label="SHAKA",
                user_id="U01",
                session_id="sess_100",
                output_path=Path(tmpdir) / "shaka.json",
                capture_context={
                    "round": "phase4_v2",
                    "scope": "unified",
                    "background": "plain",
                    "lighting": "bright",
                },
                sound_enabled=False,
            )
            session = RecordingSession(
                gesture_label="SHAKA",
                user_id="U01",
                session_id="sess_100",
                schema_version="phase3.v2",
                feature_dimension=92,
                capture_context=dict(cfg.capture_context),
                created_at="2026-04-17T09:00:00",
                samples=[
                    RecordingSample(0, 1, 1.0, "SHAKA", "U01", "sess_100", "Right", "phase3.v2", "ok", 1.0, 1.0, 1.0, 1.0, [0.1]),
                ],
            )
            runtime = RecorderRuntime(accepted_samples=1, rejected_attempts=3)

            status = _sync_tracker_row(
                cfg=cfg,
                session=session,
                runtime=runtime,
                discarded=False,
                saved=True,
                tracker_path=tracker_path,
            )

            self.assertEqual(status, "updated")
            with tracker_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["accepted_samples"], "1")
            self.assertEqual(rows[0]["rejected_attempts"], "3")
            self.assertEqual(rows[0]["status"], "completed")
            self.assertEqual(rows[0]["file_path"], str(cfg.output_path))
            self.assertIn("session_id=sess_100", rows[0]["notes"])

    def test_sync_tracker_row_skips_when_required_context_is_missing(self):
        cfg = RecorderConfig(
            gesture_label="FIST",
            user_id="U01",
            session_id="sess_101",
            output_path=Path("out.json"),
            capture_context={"background": "plain"},
            sound_enabled=False,
        )
        session = RecordingSession(
            gesture_label="FIST",
            user_id="U01",
            session_id="sess_101",
            schema_version="phase3.v2",
            feature_dimension=92,
            capture_context=dict(cfg.capture_context),
            created_at="2026-04-17T09:00:00",
        )
        runtime = RecorderRuntime()

        status = _sync_tracker_row(
            cfg=cfg,
            session=session,
            runtime=runtime,
            discarded=False,
            saved=False,
            tracker_path=Path("missing.csv"),
        )

        self.assertEqual(status, "skipped_missing_context")


if __name__ == "__main__":
    unittest.main()
