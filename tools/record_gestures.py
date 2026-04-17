from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:
    import winsound
except ImportError:  # pragma: no cover - non-Windows fallback
    winsound = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import CONFIG
from app.gestures.classifier import SVMClassifier
from app.gestures.features import assess_hand_input_quality, extract_feature_vector, get_landmarks
from app.gestures.sets.labels import ALL_ALLOWED_LABELS, UNIFIED_LABEL_ORDER
from app.gestures.suite import GestureSuite, GestureSuiteOut
from app.perception.camera import Camera
from app.perception.hand_tracker import HandTracker
from app.perception.preprocessing import Preprocessor
from app.perception.threaded_camera import ThreadedCamera
from tools.check_collection_readiness import validate_collection_readiness

SOUND_ASSET_DIR = ROOT / "assets" / "sounds"
DEFAULT_PROTOCOL_PATH = ROOT / "docs" / "dataset_collection_protocol.md"
DEFAULT_TRACKER_PATH = ROOT / "docs" / "dataset_collection_tracker.csv"
RECORDER_VERSION = "phase4.recorder.v2"
RECORDER_RULES_ONLY_MODEL_PATH = ROOT / "models" / "__recorder_rules_only__.joblib"
LANDMARK_ARTIFACT_VERSION = "phase4.landmarks.v1"
SNAPSHOT_ARTIFACT_VERSION = "phase4.snapshots.v1"
DEFAULT_SNAPSHOT_MAX_EDGE = 256
DEFAULT_SNAPSHOT_JPEG_QUALITY = 70
DEFAULT_SNAPSHOT_PADDING_RATIO = 0.18


class RecorderState(str, Enum):
    STARTUP_HELP = "STARTUP_HELP"
    PAUSED = "PAUSED"
    COUNTDOWN = "COUNTDOWN"
    RECORDING_AUTO = "RECORDING_AUTO"
    RECORDING_MANUAL = "RECORDING_MANUAL"
    CONFIRM_SAVE = "CONFIRM_SAVE"


class CaptureMode(str, Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"


@dataclass(frozen=True)
class RecorderGuidance:
    chosen: str | None
    stable: str | None
    eligible: str | None
    source: str
    gate_reason: str
    capture_ready: bool


def _slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    text = text.strip("-._")
    return text or "session"


def default_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_capture_context(items: list[str]) -> dict[str, str]:
    context: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid capture-context item: {item!r}. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid capture-context item: {item!r}. Key may not be empty.")
        context[key] = value
    return context


def _maybe_validate_readiness(
    *,
    capture_context: dict[str, str],
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    tracker_path: Path = DEFAULT_TRACKER_PATH,
    skip_check: bool = False,
):
    if skip_check:
        return None

    scope = capture_context.get("scope")
    round_tag = capture_context.get("round")
    if not scope or not round_tag:
        return None

    report = validate_collection_readiness(
        protocol_path=protocol_path,
        tracker_path=tracker_path,
        scope=scope,
        round_tag=round_tag,
    )
    if not report.ok:
        issue_lines = [f"- {issue.code}: {issue.message}" for issue in report.issues]
        message = "\n".join(
            [
                "[Recorder] Dataset readiness check failed.",
                f"scope={scope} round={round_tag}",
                *issue_lines,
                "Fix the tracker/protocol mismatch or re-run with --skip-readiness-check if this session is intentionally ad hoc.",
            ]
        )
        raise SystemExit(message)

    print(
        f"[Recorder] Readiness check OK | "
        f"scope={report.scope} | round={report.round_tag} | "
        f"rows={report.tracker_rows}"
    )
    return report


@dataclass(frozen=True)
class RecorderConfig:
    gesture_label: str
    user_id: str
    session_id: str
    output_path: Path
    capture_context: dict[str, str] = field(default_factory=dict)
    sample_interval_s: float = 0.20
    max_samples: int | None = None
    countdown_seconds: float = 3.0
    sound_enabled: bool = True
    mirror_view: bool = True
    overwrite_output: bool = False
    require_label_match: bool = True
    tracker_sync_enabled: bool = True
    save_raw_landmarks: bool = True
    save_snapshots: bool = False
    snapshot_max_edge: int = DEFAULT_SNAPSHOT_MAX_EDGE
    snapshot_jpeg_quality: int = DEFAULT_SNAPSHOT_JPEG_QUALITY
    snapshot_padding_ratio: float = DEFAULT_SNAPSHOT_PADDING_RATIO
    detect_scale: float = CONFIG.detect_scale
    enable_preprocessing: bool = CONFIG.enable_preprocessing


@dataclass(frozen=True)
class RecordingSample:
    sample_index: int
    frame_seq: int
    captured_at: float
    gesture_label: str
    user_id: str
    session_id: str
    handedness: str | None
    schema_version: str
    quality_reason: str
    quality_scale: float
    quality_palm_width: float
    quality_bbox_width: float
    quality_bbox_height: float
    feature_values: list[float]
    guide_chosen: str | None = None
    guide_stable: str | None = None
    guide_eligible: str | None = None
    guide_source: str | None = None
    guide_gate_reason: str | None = None
    raw_landmark_index: int | None = None
    snapshot_relpath: str | None = None


@dataclass
class RecordingSession:
    gesture_label: str
    user_id: str
    session_id: str
    schema_version: str
    feature_dimension: int
    capture_context: dict[str, str]
    created_at: str
    recorder_version: str = RECORDER_VERSION
    artifacts: dict[str, Any] = field(default_factory=dict)
    samples: list[RecordingSample] = field(default_factory=list)


@dataclass
class RecorderRuntime:
    state: RecorderState = RecorderState.STARTUP_HELP
    capture_mode: CaptureMode = CaptureMode.AUTO
    accepted_samples: int = 0
    rejected_attempts: int = 0
    last_result: str = "Waiting for valid hand input"
    last_result_level: str = "info"
    help_visible_until: float = 0.0
    started_at_monotonic: float = 0.0
    countdown_started_at: float | None = None
    countdown_last_whole_second: int | None = None
    label_rejections: int = 0


@dataclass
class RecorderArtifacts:
    raw_landmarks: list[np.ndarray] = field(default_factory=list)
    snapshot_blobs: dict[int, bytes] = field(default_factory=dict)


def build_output_path(
    gesture_label: str,
    user_id: str,
    session_id: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    root = base_dir or (CONFIG.paths.data_dir / "recordings")
    filename = f"{_slugify(gesture_label)}__{_slugify(user_id)}__{_slugify(session_id)}.json"
    return root / filename


def _validate_gesture_label(gesture_label: str) -> str:
    normalized = str(gesture_label).strip().upper()
    if normalized not in ALL_ALLOWED_LABELS:
        allowed = ", ".join(UNIFIED_LABEL_ORDER)
        raise SystemExit(
            f"[Recorder] Unknown gesture label {gesture_label!r}. "
            f"Use one of: {allowed}"
        )
    return normalized


def _ensure_output_path_available(output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise SystemExit(
            f"[Recorder] Refusing to overwrite existing file: {output_path}\n"
            "Use --overwrite or choose a new --session-id/--output path."
        )


def _build_recorder_suite() -> GestureSuite:
    classifier = SVMClassifier(
        model_path=RECORDER_RULES_ONLY_MODEL_PATH,
        allowed=ALL_ALLOWED_LABELS,
    )
    return GestureSuite(
        classifier=classifier,
        allowed=ALL_ALLOWED_LABELS,
        priority=UNIFIED_LABEL_ORDER,
        allow_priority=True,
    )


def _build_recorder_guidance(
    *,
    target_label: str,
    suite_out: GestureSuiteOut | None,
    require_label_match: bool,
) -> RecorderGuidance:
    if suite_out is None:
        return RecorderGuidance(
            chosen=None,
            stable=None,
            eligible=None,
            source="none",
            gate_reason="label_no_match",
            capture_ready=not require_label_match,
        )

    if not require_label_match:
        return RecorderGuidance(
            chosen=suite_out.chosen,
            stable=suite_out.stable,
            eligible=suite_out.eligible,
            source=suite_out.source,
            gate_reason="label_guard_disabled",
            capture_ready=True,
        )

    if suite_out.eligible == target_label:
        gate_reason = "ok"
        capture_ready = True
    elif suite_out.stable == target_label:
        gate_reason = f"label_hold_pending:{target_label}"
        capture_ready = False
    elif suite_out.chosen == target_label:
        gate_reason = f"label_switch_pending:{target_label}"
        capture_ready = False
    else:
        observed = suite_out.stable or suite_out.chosen or suite_out.eligible or "NONE"
        gate_reason = f"label_mismatch:{observed}"
        capture_ready = False

    return RecorderGuidance(
        chosen=suite_out.chosen,
        stable=suite_out.stable,
        eligible=suite_out.eligible,
        source=suite_out.source,
        gate_reason=gate_reason,
        capture_ready=capture_ready,
    )


def session_to_payload(session: RecordingSession) -> dict[str, Any]:
    payload = {
        "gesture_label": session.gesture_label,
        "user_id": session.user_id,
        "session_id": session.session_id,
        "schema_version": session.schema_version,
        "feature_dimension": session.feature_dimension,
        "capture_context": dict(session.capture_context),
        "created_at": session.created_at,
        "recorder_version": session.recorder_version,
        "sample_count": len(session.samples),
        "samples": [asdict(sample) for sample in session.samples],
    }
    if session.artifacts:
        payload["artifacts"] = dict(session.artifacts)
    return payload


def _landmark_sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(".landmarks.npz")


def _snapshot_dir(output_path: Path) -> Path:
    return output_path.parent / f"{output_path.stem}__snapshots"


def _snapshot_relpath(sample_index: int) -> str:
    return f"{sample_index:04d}.jpg"


def _extract_raw_landmark_array(landmarks: Any) -> np.ndarray:
    lm = get_landmarks(landmarks)
    matrix = np.asarray(
        [
            [float(point.x), float(point.y), float(getattr(point, "z", 0.0))]
            for point in lm
        ],
        dtype=np.float16,
    )
    if matrix.shape != (21, 3):
        raise ValueError(f"Expected 21x3 landmarks, got {matrix.shape!r}")
    return matrix


def _encode_snapshot_crop(
    frame_bgr,
    landmarks: Any,
    *,
    max_edge: int,
    jpeg_quality: int,
    padding_ratio: float,
) -> bytes:
    lm = get_landmarks(landmarks)
    h, w = frame_bgr.shape[:2]
    xs = [float(point.x) for point in lm]
    ys = [float(point.y) for point in lm]
    min_x = max(0.0, min(xs))
    max_x = min(1.0, max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(1.0, max(ys))
    bbox_w = max_x - min_x
    bbox_h = max_y - min_y
    pad = max(bbox_w, bbox_h) * max(0.0, padding_ratio)

    x0 = max(0, int(np.floor((min_x - pad) * w)))
    y0 = max(0, int(np.floor((min_y - pad) * h)))
    x1 = min(w, int(np.ceil((max_x + pad) * w)))
    y1 = min(h, int(np.ceil((max_y + pad) * h)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Snapshot crop resolved to an empty region.")

    crop = frame_bgr[y0:y1, x0:x1]
    longest_edge = max(crop.shape[:2])
    if longest_edge > max_edge:
        scale = float(max_edge) / float(longest_edge)
        crop = cv2.resize(
            crop,
            (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise ValueError("OpenCV failed to encode snapshot crop.")
    return encoded.tobytes()


def _persist_artifacts(
    *,
    session: RecordingSession,
    output_path: Path,
    cfg: RecorderConfig,
    artifacts: RecorderArtifacts,
) -> None:
    manifest: dict[str, Any] = {}
    landmark_path = _landmark_sidecar_path(output_path)
    if cfg.overwrite_output and landmark_path.exists():
        landmark_path.unlink()
    if cfg.save_raw_landmarks and artifacts.raw_landmarks:
        landmark_stack = np.stack(artifacts.raw_landmarks, axis=0).astype(np.float16, copy=False)
        np.savez_compressed(landmark_path, landmarks=landmark_stack)
        manifest["raw_landmarks"] = {
            "version": LANDMARK_ARTIFACT_VERSION,
            "path": landmark_path.name,
            "format": "npz",
            "dtype": "float16",
            "shape": [int(v) for v in landmark_stack.shape],
            "coordinate_space": "mediapipe_normalized_xyz",
            "sample_field": "raw_landmark_index",
        }

    snapshot_dir = _snapshot_dir(output_path)
    if cfg.overwrite_output and snapshot_dir.exists():
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    if cfg.save_snapshots and artifacts.snapshot_blobs:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for sample_index, blob in sorted(artifacts.snapshot_blobs.items()):
            (snapshot_dir / _snapshot_relpath(sample_index)).write_bytes(blob)
        manifest["snapshots"] = {
            "version": SNAPSHOT_ARTIFACT_VERSION,
            "directory": snapshot_dir.name,
            "format": "jpg",
            "kind": "accepted_hand_crop",
            "count": len(artifacts.snapshot_blobs),
            "max_edge": int(cfg.snapshot_max_edge),
            "jpeg_quality": int(cfg.snapshot_jpeg_quality),
            "padding_ratio": float(cfg.snapshot_padding_ratio),
            "mirrored": False,
            "sample_field": "snapshot_relpath",
        }

    session.artifacts = manifest


def save_session(
    session: RecordingSession,
    output_path: Path,
    *,
    cfg: RecorderConfig | None = None,
    artifacts: RecorderArtifacts | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cfg is not None and artifacts is not None:
        _persist_artifacts(
            session=session,
            output_path=output_path,
            cfg=cfg,
            artifacts=artifacts,
        )
    payload = session_to_payload(session)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _mirror_landmarks(landmarks):
    mirrored = type(landmarks)()
    mirrored.CopyFrom(landmarks)
    for point in mirrored.landmark:
        point.x = 1.0 - point.x
    return mirrored


def _draw_text(frame_bgr, text: str, xy: tuple[int, int], scale: float = 0.6) -> None:
    cv2.putText(frame_bgr, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def _humanize_quality_reason(quality_reason: str, quality_ok: bool) -> str:
    if quality_ok:
        return "Ready to capture"

    if quality_reason == "label_guard_disabled":
        return "Ready to capture"
    if quality_reason == "label_no_match":
        return "Target gesture not recognized yet"
    if quality_reason.startswith("label_hold_pending:"):
        target = quality_reason.split(":", 1)[1]
        return f"Hold {target} steady until confirmed"
    if quality_reason.startswith("label_switch_pending:"):
        target = quality_reason.split(":", 1)[1]
        return f"Keep {target} steady; confirmation pending"
    if quality_reason.startswith("label_mismatch:"):
        observed = quality_reason.split(":", 1)[1]
        return f"Target mismatch: saw {observed}"

    return {
        "no_hand": "No hand detected",
        "invalid_landmarks": "Invalid hand landmarks",
        "incomplete_landmarks": "Hand not fully detected",
        "hand_too_small": "Move hand closer to the camera",
        "palm_too_narrow": "Show palm and fingers more clearly",
        "bbox_too_small": "Center your hand in frame",
        "non_finite_landmarks": "Tracking unstable, hold still",
    }.get(quality_reason, f"Rejected: {quality_reason}")


def _progress_text(accepted_samples: int, target_samples: int | None) -> str:
    if target_samples is None:
        return f"{accepted_samples}"
    return f"{accepted_samples} / {target_samples}"


def _state_text(state: RecorderState) -> str:
    return {
        RecorderState.STARTUP_HELP: "PAUSED",
        RecorderState.PAUSED: "PAUSED",
        RecorderState.COUNTDOWN: "COUNTDOWN",
        RecorderState.RECORDING_AUTO: "RECORDING",
        RecorderState.RECORDING_MANUAL: "READY",
        RecorderState.CONFIRM_SAVE: "CONFIRM",
    }[state]


def _sound_asset_path(event: str) -> Path:
    filenames = {
        "countdown_tick": "countdown_tick.wav",
        "countdown_go": "start.wav",
        "target_reached": "target_reached.wav",
        "save": "save.wav",
        "discard": "discard.wav",
        "reject": "reject.wav",
    }
    filename = filenames.get(event)
    if filename is None:
        raise KeyError(f"Unsupported sound event: {event}")
    return SOUND_ASSET_DIR / filename


def _play_wave_fallback(path: Path) -> None:
    if winsound is None or not path.exists():
        return
    winsound.PlaySound(
        str(path),
        winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
    )


def _play_sound(event: str, *, enabled: bool = True) -> None:
    if not enabled:
        return

    fallback_path = _sound_asset_path(event)
    if winsound is None:
        return

    try:
        # Short, familiar cues so dataset operators get feedback without distraction.
        if event == "countdown_tick":
            winsound.Beep(880, 90)
        elif event == "countdown_go":
            winsound.Beep(1320, 180)
        elif event == "target_reached":
            winsound.Beep(1175, 140)
            winsound.Beep(1480, 180)
        elif event == "save":
            winsound.MessageBeep(winsound.MB_OK)
        elif event == "discard":
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        elif event == "reject":
            winsound.MessageBeep(winsound.MB_ICONHAND)
        else:
            _play_wave_fallback(fallback_path)
            return
    except RuntimeError:
        _play_wave_fallback(fallback_path)


def _countdown_remaining(runtime: RecorderRuntime, cfg: RecorderConfig, now: float) -> int:
    if runtime.countdown_started_at is None:
        return 0
    elapsed = max(0.0, now - runtime.countdown_started_at)
    remaining = max(0.0, cfg.countdown_seconds - elapsed)
    return int(remaining) + (0 if remaining.is_integer() else 1)


def _start_countdown(runtime: RecorderRuntime, cfg: RecorderConfig, now: float) -> None:
    runtime.state = RecorderState.COUNTDOWN
    runtime.countdown_started_at = now
    runtime.countdown_last_whole_second = None
    _set_last_result(runtime, f"Recording starts in {int(cfg.countdown_seconds)}", level="info")


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    mins, secs = divmod(total, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _handedness_summary(session: RecordingSession) -> str:
    counts: dict[str, int] = {"Left": 0, "Right": 0, "Unknown": 0}
    for sample in session.samples:
        key = sample.handedness if sample.handedness in ("Left", "Right") else "Unknown"
        counts[key] += 1
    return f"L:{counts['Left']} R:{counts['Right']} U:{counts['Unknown']}"


def _truncate_middle(text: str, max_len: int = 56) -> str:
    if len(text) <= max_len:
        return text
    keep = max(8, (max_len - 3) // 2)
    return f"{text[:keep]}...{text[-keep:]}"


def _build_session_summary(
    session: RecordingSession,
    runtime: RecorderRuntime,
    cfg: RecorderConfig,
    *,
    now: float,
) -> list[str]:
    duration = _format_duration(now - runtime.started_at_monotonic)
    lines = [
        f"Gesture: {session.gesture_label}",
        f"User: {session.user_id}",
        f"Session: {session.session_id}",
        f"Accepted: {len(session.samples)}",
        f"Rejected: {runtime.rejected_attempts}",
        (
            f"Acceptance: {len(session.samples)}/{len(session.samples) + runtime.rejected_attempts} "
            f"({(100.0 * len(session.samples) / (len(session.samples) + runtime.rejected_attempts)):.1f}%)"
            if (len(session.samples) + runtime.rejected_attempts) > 0
            else "Acceptance: n/a"
        ),
        f"Label rejects: {runtime.label_rejections}",
        f"Duration: {duration}",
        f"Handedness: {_handedness_summary(session)}",
    ]
    if session.samples:
        lines.append(f"Output: {_truncate_middle(str(cfg.output_path))}")
    else:
        lines.append("Output: no file will be written")
    return lines


def _draw_overlay(
    frame_bgr,
    *,
    cfg: RecorderConfig,
    session: RecordingSession,
    runtime: RecorderRuntime,
    quality_reason: str,
    quality_ok: bool,
    handedness: str | None,
    guidance: RecorderGuidance | None,
) -> None:
    now = time.perf_counter()
    h, w = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (8, 8), (min(w - 8, 610), 270), (18, 18, 18), -1)
    cv2.rectangle(frame_bgr, (8, 8), (min(w - 8, 610), 270), (70, 70, 70), 1)

    guide_text = "Guide: target not checked"
    if guidance is not None:
        guide_text = (
            f"Guide: tgt={cfg.gesture_label} chosen={guidance.chosen or 'NONE'} "
            f"stable={guidance.stable or 'NONE'} eligible={guidance.eligible or 'NONE'}"
        )

    lines = [
        (f"Label: {cfg.gesture_label}", 0.7),
        (f"User: {cfg.user_id}", 0.58),
        (f"Session: {cfg.session_id}", 0.58),
        (f"State: {_state_text(runtime.state)} | Mode: {runtime.capture_mode.value}", 0.58),
        (f"Accepted: {runtime.accepted_samples} | Rejected: {runtime.rejected_attempts}", 0.58),
        (f"Label rejects: {runtime.label_rejections}", 0.58),
        (f"Progress: {_progress_text(runtime.accepted_samples, cfg.max_samples)}", 0.58),
        (f"Handedness: {handedness or 'Unknown'}", 0.58),
        (guide_text, 0.54),
        (f"Gate: {_humanize_quality_reason(quality_reason, quality_ok)}", 0.58),
        (f"Last result: {runtime.last_result}", 0.58),
    ]

    y = 32
    for text, scale in lines:
        _draw_text(frame_bgr, text, (18, y), scale)
        y += 24

    if now <= runtime.help_visible_until:
        cv2.rectangle(frame_bgr, (8, 230), (min(w - 8, 610), 352), (18, 18, 18), -1)
        cv2.rectangle(frame_bgr, (8, 230), (min(w - 8, 610), 352), (70, 70, 70), 1)
        _draw_text(frame_bgr, "How to record", (18, 252), 0.62)
        _draw_text(frame_bgr, "Record one gesture label per session.", (18, 276), 0.54)
        _draw_text(frame_bgr, "Press M to switch between AUTO and MANUAL capture.", (18, 298), 0.54)
        _draw_text(frame_bgr, "AUTO uses interval capture. MANUAL arms with SPACE and captures on C.", (18, 320), 0.54)
        _draw_text(frame_bgr, "Capture now also waits for the target label to be confirmed.", (18, 342), 0.54)

    footer = "M mode | SPACE start/pause | C capture once | U undo | X discard | Q/Esc save+quit"
    _draw_text(frame_bgr, footer, (10, h - 15), 0.55)

    if runtime.state == RecorderState.COUNTDOWN and runtime.countdown_started_at is not None:
        remaining = _countdown_remaining(runtime, cfg, now)
        cv2.rectangle(frame_bgr, (w // 2 - 90, h // 2 - 70), (w // 2 + 90, h // 2 + 70), (16, 16, 16), -1)
        cv2.rectangle(frame_bgr, (w // 2 - 90, h // 2 - 70), (w // 2 + 90, h // 2 + 70), (220, 220, 220), 2)
        _draw_text(frame_bgr, "STARTING", (w // 2 - 72, h // 2 - 18), 0.8)
        _draw_text(frame_bgr, str(max(remaining, 0)), (w // 2 - 18, h // 2 + 34), 1.6)

    if runtime.state == RecorderState.CONFIRM_SAVE:
        left = max(20, w // 2 - 240)
        top = max(30, h // 2 - 150)
        right = min(w - 20, left + 480)
        bottom = min(h - 30, top + 300)
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (14, 14, 14), -1)
        cv2.rectangle(frame_bgr, (left, top), (right, bottom), (220, 220, 220), 2)
        title = "Save And Quit?" if session.samples else "Quit Without Saving?"
        _draw_text(frame_bgr, title, (left + 18, top + 30), 0.72)
        if not session.samples:
            _draw_text(frame_bgr, "No valid samples were captured in this session.", (left + 18, top + 58), 0.54)
        y = top + 88
        for line in _build_session_summary(session, runtime, cfg, now=now):
            _draw_text(frame_bgr, line, (left + 18, y), 0.54)
            y += 24
        footer = "Y confirm | N return | D discard session"
        _draw_text(frame_bgr, footer, (left + 18, bottom - 18), 0.54)


def _build_sample(
    *,
    sample_index: int,
    frame_seq: int,
    cfg: RecorderConfig,
    handedness: str | None,
    feature_values: tuple[float, ...],
    schema_version: str,
    quality,
    guidance: RecorderGuidance | None,
    raw_landmark_index: int | None,
    snapshot_relpath: str | None,
) -> RecordingSample:
    return RecordingSample(
        sample_index=sample_index,
        frame_seq=frame_seq,
        captured_at=time.time(),
        gesture_label=cfg.gesture_label,
        user_id=cfg.user_id,
        session_id=cfg.session_id,
        handedness=handedness,
        schema_version=schema_version,
        quality_reason=quality.reason,
        quality_scale=quality.scale,
        quality_palm_width=quality.palm_width,
        quality_bbox_width=quality.bbox_width,
        quality_bbox_height=quality.bbox_height,
        feature_values=[float(v) for v in feature_values],
        guide_chosen=None if guidance is None else guidance.chosen,
        guide_stable=None if guidance is None else guidance.stable,
        guide_eligible=None if guidance is None else guidance.eligible,
        guide_source=None if guidance is None else guidance.source,
        guide_gate_reason=None if guidance is None else guidance.gate_reason,
        raw_landmark_index=raw_landmark_index,
        snapshot_relpath=snapshot_relpath,
    )


def _capture_sample(
    *,
    frame_seq: int,
    cfg: RecorderConfig,
    frame_bgr,
    detected_hand,
    session: RecordingSession,
    artifacts: RecorderArtifacts,
    guidance: RecorderGuidance | None,
) -> RecordingSample:
    quality = assess_hand_input_quality(detected_hand.landmarks)
    if not quality.passed:
        raise ValueError(f"Cannot capture sample from rejected hand input: {quality.reason}")

    sample_index = len(session.samples)
    feature_vector = extract_feature_vector(detected_hand.landmarks)
    raw_landmark_values = (
        _extract_raw_landmark_array(detected_hand.landmarks)
        if cfg.save_raw_landmarks
        else None
    )
    snapshot_blob = (
        _encode_snapshot_crop(
            frame_bgr,
            detected_hand.landmarks,
            max_edge=cfg.snapshot_max_edge,
            jpeg_quality=cfg.snapshot_jpeg_quality,
            padding_ratio=cfg.snapshot_padding_ratio,
        )
        if cfg.save_snapshots
        else None
    )
    sample = _build_sample(
        sample_index=sample_index,
        frame_seq=frame_seq,
        cfg=cfg,
        handedness=detected_hand.handedness,
        feature_values=feature_vector.values,
        schema_version=feature_vector.schema_version,
        quality=quality,
        guidance=guidance,
        raw_landmark_index=sample_index if raw_landmark_values is not None else None,
        snapshot_relpath=_snapshot_relpath(sample_index) if snapshot_blob is not None else None,
    )
    session.samples.append(sample)
    if raw_landmark_values is not None:
        artifacts.raw_landmarks.append(raw_landmark_values)
    if snapshot_blob is not None:
        artifacts.snapshot_blobs[sample_index] = snapshot_blob
    return sample


def _undo_last_sample(session: RecordingSession, artifacts: RecorderArtifacts) -> RecordingSample | None:
    if not session.samples:
        return None
    removed = session.samples.pop()
    if removed.raw_landmark_index is not None and artifacts.raw_landmarks:
        artifacts.raw_landmarks.pop()
    if removed.snapshot_relpath is not None:
        artifacts.snapshot_blobs.pop(removed.sample_index, None)
    return removed


def _discard_session(session: RecordingSession, artifacts: RecorderArtifacts) -> None:
    session.samples.clear()
    session.artifacts.clear()
    artifacts.raw_landmarks.clear()
    artifacts.snapshot_blobs.clear()


def _set_last_result(runtime: RecorderRuntime, message: str, *, level: str = "info") -> None:
    runtime.last_result = message
    runtime.last_result_level = level


def _update_state(runtime: RecorderRuntime, new_state: RecorderState) -> None:
    runtime.state = new_state
    if new_state != RecorderState.COUNTDOWN:
        runtime.countdown_started_at = None
        runtime.countdown_last_whole_second = None


def _toggle_capture_mode(runtime: RecorderRuntime) -> CaptureMode:
    runtime.capture_mode = (
        CaptureMode.MANUAL if runtime.capture_mode == CaptureMode.AUTO else CaptureMode.AUTO
    )
    if runtime.state != RecorderState.CONFIRM_SAVE:
        _update_state(runtime, RecorderState.PAUSED)
    return runtime.capture_mode


def _record_rejection(runtime: RecorderRuntime, quality_reason: str, *, sound_enabled: bool = False) -> None:
    runtime.rejected_attempts += 1
    if str(quality_reason).startswith("label_"):
        runtime.label_rejections += 1
    _set_last_result(runtime, f"Rejected: {_humanize_quality_reason(quality_reason, False)}", level="reject")
    _play_sound("reject", enabled=sound_enabled)


def _advance_countdown(runtime: RecorderRuntime, cfg: RecorderConfig, now: float) -> bool:
    if runtime.state != RecorderState.COUNTDOWN or runtime.countdown_started_at is None:
        return False

    remaining = _countdown_remaining(runtime, cfg, now)
    if remaining != runtime.countdown_last_whole_second and remaining > 0:
        runtime.countdown_last_whole_second = remaining
        _set_last_result(runtime, f"Recording starts in {remaining}", level="info")
        _play_sound("countdown_tick", enabled=cfg.sound_enabled)

    if (now - runtime.countdown_started_at) >= cfg.countdown_seconds:
        _update_state(runtime, RecorderState.RECORDING_AUTO)
        _set_last_result(runtime, "Auto capture started", level="info")
        _play_sound("countdown_go", enabled=cfg.sound_enabled)
        return True

    return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record labeled gesture feature vectors for the Virtual Gesture Mouse project."
    )
    parser.add_argument("--gesture-label", required=True, help="Target gesture label for this recording session.")
    parser.add_argument("--user-id", required=True, help="User identifier for grouped evaluation later.")
    parser.add_argument("--session-id", default=default_session_id(), help="Session identifier. Defaults to a timestamp.")
    parser.add_argument("--capture-context", action="append", default=[], help="Optional KEY=VALUE metadata. Repeat as needed.")
    parser.add_argument("--sample-interval", type=float, default=0.20, help="Minimum seconds between automatic captures while recording.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional automatic stop once this many samples are captured.")
    parser.add_argument("--countdown-seconds", type=float, default=3.0, help="Seconds to wait before auto recording begins.")
    parser.add_argument("--mute-sounds", action="store_true", help="Disable recorder beep and system sound cues.")
    parser.add_argument("--skip-readiness-check", action="store_true", help="Skip the automatic dataset readiness preflight even when scope and round are provided.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for the saved JSON session.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing output file.")
    parser.add_argument("--allow-label-mismatch", action="store_true", help="Disable the target-label confirmation guard and capture any quality-approved hand pose.")
    parser.add_argument("--no-tracker-sync", action="store_true", help="Do not auto-update the dataset tracker row on save/discard.")
    parser.add_argument("--no-raw-landmarks", action="store_true", help="Do not save the compact raw-landmark sidecar. Default is on because it is tiny and future-proofs the dataset.")
    parser.add_argument("--save-snapshots", action="store_true", help="Save optional accepted-sample hand crops as low-resolution JPEG sidecars.")
    parser.add_argument("--snapshot-max-edge", type=int, default=DEFAULT_SNAPSHOT_MAX_EDGE, help="Maximum edge length for optional accepted-sample hand-crop snapshots.")
    parser.add_argument("--snapshot-jpeg-quality", type=int, default=DEFAULT_SNAPSHOT_JPEG_QUALITY, help="JPEG quality for optional accepted-sample hand-crop snapshots.")
    return parser


def _make_session(cfg: RecorderConfig) -> RecordingSession:
    return RecordingSession(
        gesture_label=cfg.gesture_label,
        user_id=cfg.user_id,
        session_id=cfg.session_id,
        schema_version="phase3.v2",
        feature_dimension=92,
        capture_context=dict(cfg.capture_context),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def _tracker_row_key(cfg: RecorderConfig) -> tuple[str, str, str, str, str, str] | None:
    context = cfg.capture_context
    values = (
        context.get("round"),
        context.get("scope"),
        cfg.user_id,
        context.get("background"),
        context.get("lighting"),
        cfg.gesture_label,
    )
    if any(not value for value in values):
        return None
    return tuple(str(value) for value in values)


def _sync_tracker_row(
    *,
    cfg: RecorderConfig,
    session: RecordingSession,
    runtime: RecorderRuntime,
    discarded: bool,
    saved: bool,
    tracker_path: Path = DEFAULT_TRACKER_PATH,
) -> str:
    if not cfg.tracker_sync_enabled:
        return "disabled"
    row_key = _tracker_row_key(cfg)
    if row_key is None:
        return "skipped_missing_context"
    if not tracker_path.exists():
        return "skipped_missing_tracker"

    with tracker_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    updated = False
    for row in rows:
        current_key = (
            row.get("round", ""),
            row.get("scope", ""),
            row.get("user_id", ""),
            row.get("background", ""),
            row.get("lighting", ""),
            row.get("gesture_label", ""),
        )
        if current_key != row_key:
            continue
        row["accepted_samples"] = str(len(session.samples))
        row["rejected_attempts"] = str(runtime.rejected_attempts)
        if saved:
            row["status"] = "completed"
            row["file_path"] = str(cfg.output_path)
        elif discarded:
            row["status"] = "discarded"
            row["file_path"] = ""
        else:
            row["status"] = "redo" if len(session.samples) > 0 else "not_started"
            row["file_path"] = ""
        notes = [token for token in [row.get("notes", "").strip(), f"session_id={cfg.session_id}"] if token]
        row["notes"] = " | ".join(dict.fromkeys(notes))
        updated = True
        break

    if not updated:
        return "skipped_no_matching_row"

    with tracker_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return "updated"


def main() -> None:
    args = build_arg_parser().parse_args()
    capture_context = parse_capture_context(args.capture_context)
    _maybe_validate_readiness(
        capture_context=capture_context,
        skip_check=args.skip_readiness_check,
    )
    gesture_label = _validate_gesture_label(args.gesture_label)
    output_path = args.output or build_output_path(gesture_label, args.user_id, args.session_id)
    _ensure_output_path_available(output_path, overwrite=bool(args.overwrite))
    cfg = RecorderConfig(
        gesture_label=gesture_label,
        user_id=args.user_id,
        session_id=args.session_id,
        output_path=output_path,
        capture_context=capture_context,
        sample_interval_s=float(args.sample_interval),
        max_samples=args.max_samples,
        countdown_seconds=max(0.0, float(args.countdown_seconds)),
        sound_enabled=not args.mute_sounds,
        mirror_view=CONFIG.mirror_view,
        overwrite_output=bool(args.overwrite),
        require_label_match=not bool(args.allow_label_mismatch),
        tracker_sync_enabled=not bool(args.no_tracker_sync),
        save_raw_landmarks=not bool(args.no_raw_landmarks),
        save_snapshots=bool(args.save_snapshots),
        snapshot_max_edge=max(96, int(args.snapshot_max_edge)),
        snapshot_jpeg_quality=max(30, min(95, int(args.snapshot_jpeg_quality))),
        detect_scale=CONFIG.detect_scale,
        enable_preprocessing=CONFIG.enable_preprocessing,
    )

    session = _make_session(cfg)
    camera = Camera(device_index=CONFIG.camera_index, width=CONFIG.cam_width, height=CONFIG.cam_height)
    tracker = HandTracker(max_num_hands=1)
    recorder_suite = _build_recorder_suite()
    preprocessor = Preprocessor(enable=cfg.enable_preprocessing)
    camera_thread: ThreadedCamera | None = None
    discarded = False
    save_requested = False
    last_capture_t = 0.0
    window_name = f"{CONFIG.app_name} - Record {cfg.gesture_label}"
    artifacts = RecorderArtifacts()
    runtime = RecorderRuntime(
        state=RecorderState.STARTUP_HELP,
        capture_mode=CaptureMode.AUTO,
        help_visible_until=time.perf_counter() + 6.0,
        started_at_monotonic=time.perf_counter(),
    )

    try:
        camera.open()
        camera_thread = ThreadedCamera(camera.cap)
        camera_thread.start()

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(window_name, CONFIG.cam_width, CONFIG.cam_height)
        except Exception:
            pass

        while True:
            frame_raw, frame_seq, _ts, _ok = camera_thread.read_latest()
            if frame_raw is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            frame_disp = cv2.flip(frame_raw, 1) if cfg.mirror_view else frame_raw
            now = time.perf_counter()
            _advance_countdown(runtime, cfg, now)
            small = cv2.resize(
                frame_raw,
                (0, 0),
                fx=cfg.detect_scale,
                fy=cfg.detect_scale,
                interpolation=cv2.INTER_AREA,
            )
            if cfg.enable_preprocessing:
                small = preprocessor.apply(small)

            detected_hand = tracker.detect(small)
            suite_out = recorder_suite.detect(detected_hand)
            guidance = _build_recorder_guidance(
                target_label=cfg.gesture_label,
                suite_out=suite_out,
                require_label_match=cfg.require_label_match,
            )
            quality_reason = "no_hand"
            quality_ok = False
            handedness = None
            if detected_hand is not None:
                quality = assess_hand_input_quality(detected_hand.landmarks)
                quality_reason = quality.reason
                quality_ok = quality.passed
                handedness = detected_hand.handedness
                draw_landmarks = _mirror_landmarks(detected_hand.landmarks) if cfg.mirror_view else detected_hand.landmarks
                tracker.draw(frame_disp, draw_landmarks)

                capture_gate_ok = quality_ok and guidance.capture_ready
                capture_gate_reason = "ok" if capture_gate_ok else (quality_reason if not quality_ok else guidance.gate_reason)

                should_auto_capture = (
                    runtime.state == RecorderState.RECORDING_AUTO
                    and capture_gate_ok
                    and (now - last_capture_t) >= cfg.sample_interval_s
                )
                if should_auto_capture:
                    sample = _capture_sample(
                        frame_seq=frame_seq,
                        cfg=cfg,
                        frame_bgr=frame_raw,
                        detected_hand=detected_hand,
                        session=session,
                        artifacts=artifacts,
                        guidance=guidance,
                    )
                    runtime.accepted_samples = len(session.samples)
                    _set_last_result(runtime, f"Accepted sample #{sample.sample_index}", level="accept")
                    last_capture_t = now
                    if cfg.max_samples is not None and len(session.samples) >= cfg.max_samples:
                        print(f"[Recorder] Reached max_samples={cfg.max_samples}.")
                        _set_last_result(runtime, f"Target reached for {cfg.gesture_label}", level="accept")
                        _update_state(runtime, RecorderState.PAUSED)
                        _play_sound("target_reached", enabled=cfg.sound_enabled)
                elif runtime.state == RecorderState.RECORDING_AUTO and not capture_gate_ok and (now - last_capture_t) >= cfg.sample_interval_s:
                    _record_rejection(runtime, capture_gate_reason, sound_enabled=cfg.sound_enabled)
                    last_capture_t = now
            else:
                guidance = _build_recorder_guidance(
                    target_label=cfg.gesture_label,
                    suite_out=suite_out,
                    require_label_match=cfg.require_label_match,
                )

            _draw_overlay(
                frame_disp,
                cfg=cfg,
                session=session,
                runtime=runtime,
                quality_reason=quality_reason if not quality_ok else guidance.gate_reason,
                quality_ok=quality_ok and guidance.capture_ready,
                handedness=handedness,
                guidance=guidance,
            )
            cv2.imshow(window_name, frame_disp)

            key = cv2.waitKey(1) & 0xFF
            if runtime.state == RecorderState.CONFIRM_SAVE:
                if key in (ord("y"), ord("Y")):
                    save_requested = bool(session.samples)
                    if not session.samples:
                        _set_last_result(runtime, "Quit confirmed without saving", level="info")
                    break
                if key in (ord("n"), ord("N")):
                    _update_state(runtime, RecorderState.PAUSED)
                    _set_last_result(runtime, "Save canceled; session resumed in paused state", level="info")
                    continue
                if key in (ord("d"), ord("D")):
                    discarded = True
                    _discard_session(session, artifacts)
                    runtime.accepted_samples = 0
                    _set_last_result(runtime, "Session discarded", level="info")
                    _play_sound("discard", enabled=cfg.sound_enabled)
                    break
                continue

            if key in (27, ord("q"), ord("Q")):
                _update_state(runtime, RecorderState.CONFIRM_SAVE)
                if session.samples:
                    _set_last_result(runtime, "Review session summary, then press Y to save or N to return", level="info")
                else:
                    _set_last_result(runtime, "No valid samples captured. Press Y to quit or N to continue", level="info")
                continue
            if key in (ord("m"), ord("M")):
                new_mode = _toggle_capture_mode(runtime)
                if new_mode == CaptureMode.MANUAL:
                    _set_last_result(runtime, "Capture mode set to MANUAL. Press SPACE to arm, then C to capture.", level="info")
                else:
                    _set_last_result(runtime, "Capture mode set to AUTO. Press SPACE to begin interval capture.", level="info")
                continue
            if key == ord(" "):
                if runtime.state in (RecorderState.RECORDING_AUTO, RecorderState.RECORDING_MANUAL):
                    _update_state(runtime, RecorderState.PAUSED)
                    if runtime.capture_mode == CaptureMode.MANUAL:
                        _set_last_result(runtime, "Manual mode paused", level="info")
                    else:
                        _set_last_result(runtime, "Auto capture paused", level="info")
                elif runtime.state == RecorderState.COUNTDOWN:
                    _update_state(runtime, RecorderState.PAUSED)
                    _set_last_result(runtime, "Countdown canceled", level="info")
                else:
                    if runtime.capture_mode == CaptureMode.MANUAL:
                        _update_state(runtime, RecorderState.RECORDING_MANUAL)
                        _set_last_result(runtime, "Manual mode armed. Press C to capture approved samples.", level="info")
                    elif cfg.countdown_seconds <= 0.0:
                        _update_state(runtime, RecorderState.RECORDING_AUTO)
                        _set_last_result(runtime, "Auto capture started", level="info")
                        _play_sound("countdown_go", enabled=cfg.sound_enabled)
                    else:
                        _start_countdown(runtime, cfg, now)
                print(f"[Recorder] State={runtime.state.value}")
            elif key in (ord("c"), ord("C")):
                if runtime.state == RecorderState.COUNTDOWN:
                    _set_last_result(runtime, "Wait for countdown to finish before capturing", level="info")
                elif runtime.capture_mode == CaptureMode.MANUAL and runtime.state != RecorderState.RECORDING_MANUAL:
                    _set_last_result(runtime, "Press SPACE to arm manual mode first", level="info")
                elif detected_hand is None:
                    _record_rejection(runtime, "no_hand", sound_enabled=cfg.sound_enabled)
                elif not quality_ok:
                    _record_rejection(runtime, quality_reason, sound_enabled=cfg.sound_enabled)
                elif not guidance.capture_ready:
                    _record_rejection(runtime, guidance.gate_reason, sound_enabled=cfg.sound_enabled)
                else:
                    sample = _capture_sample(
                        frame_seq=frame_seq,
                        cfg=cfg,
                        frame_bgr=frame_raw,
                        detected_hand=detected_hand,
                        session=session,
                        artifacts=artifacts,
                        guidance=guidance,
                    )
                    runtime.accepted_samples = len(session.samples)
                    _set_last_result(runtime, f"Accepted sample #{sample.sample_index}", level="accept")
                    last_capture_t = time.perf_counter()
                    print(f"[Recorder] Captured sample #{len(session.samples) - 1}")
            elif key in (ord("u"), ord("U")) and session.samples:
                removed = _undo_last_sample(session, artifacts)
                if removed is not None:
                    runtime.accepted_samples = len(session.samples)
                    _set_last_result(runtime, f"Removed sample #{removed.sample_index}", level="info")
                    print(f"[Recorder] Removed sample #{removed.sample_index}")
            elif key in (ord("u"), ord("U")):
                _set_last_result(runtime, "Nothing to undo", level="info")
            elif key in (ord("x"), ord("X")):
                discarded = True
                _discard_session(session, artifacts)
                runtime.accepted_samples = 0
                _set_last_result(runtime, "Session discarded", level="info")
                _play_sound("discard", enabled=cfg.sound_enabled)
                print("[Recorder] Session discarded by user.")
                break

            if runtime.state == RecorderState.STARTUP_HELP and time.perf_counter() > runtime.help_visible_until:
                _update_state(runtime, RecorderState.PAUSED)

    finally:
        if camera_thread is not None:
            camera_thread.stop()
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()

    if discarded:
        sync_status = _sync_tracker_row(
            cfg=cfg,
            session=session,
            runtime=runtime,
            discarded=True,
            saved=False,
        )
        print(f"[Recorder] Tracker sync: {sync_status}")
        return

    if not save_requested:
        sync_status = _sync_tracker_row(
            cfg=cfg,
            session=session,
            runtime=runtime,
            discarded=False,
            saved=False,
        )
        print(f"[Recorder] Tracker sync: {sync_status}")
        print("[Recorder] Session closed without saving.")
        return

    if not session.samples:
        sync_status = _sync_tracker_row(
            cfg=cfg,
            session=session,
            runtime=runtime,
            discarded=True,
            saved=False,
        )
        print(f"[Recorder] Tracker sync: {sync_status}")
        _play_sound("discard", enabled=cfg.sound_enabled)
        print("[Recorder] No samples captured. Nothing saved.")
        return

    save_session(session, cfg.output_path, cfg=cfg, artifacts=artifacts)
    sync_status = _sync_tracker_row(
        cfg=cfg,
        session=session,
        runtime=runtime,
        discarded=False,
        saved=True,
    )
    _play_sound("save", enabled=cfg.sound_enabled)
    print(f"[Recorder] Tracker sync: {sync_status}")
    print(f"[Recorder] Saved {len(session.samples)} samples to {cfg.output_path}")


if __name__ == "__main__":
    main()
