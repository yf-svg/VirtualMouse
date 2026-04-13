from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
try:
    import winsound
except ImportError:  # pragma: no cover - non-Windows fallback
    winsound = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import CONFIG
from app.gestures.features import assess_hand_input_quality, extract_feature_vector
from app.perception.camera import Camera
from app.perception.hand_tracker import HandTracker
from app.perception.preprocessing import Preprocessor
from app.perception.threaded_camera import ThreadedCamera
from tools.check_collection_readiness import validate_collection_readiness

SOUND_ASSET_DIR = ROOT / "assets" / "sounds"
DEFAULT_PROTOCOL_PATH = ROOT / "docs" / "dataset_collection_protocol.md"
DEFAULT_TRACKER_PATH = ROOT / "docs" / "dataset_collection_tracker.csv"


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


@dataclass
class RecordingSession:
    gesture_label: str
    user_id: str
    session_id: str
    schema_version: str
    feature_dimension: int
    capture_context: dict[str, str]
    created_at: str
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


def session_to_payload(session: RecordingSession) -> dict[str, Any]:
    return {
        "gesture_label": session.gesture_label,
        "user_id": session.user_id,
        "session_id": session.session_id,
        "schema_version": session.schema_version,
        "feature_dimension": session.feature_dimension,
        "capture_context": dict(session.capture_context),
        "created_at": session.created_at,
        "sample_count": len(session.samples),
        "samples": [asdict(sample) for sample in session.samples],
    }


def save_session(session: RecordingSession, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
) -> None:
    now = time.perf_counter()
    h, w = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (8, 8), (min(w - 8, 610), 220), (18, 18, 18), -1)
    cv2.rectangle(frame_bgr, (8, 8), (min(w - 8, 610), 220), (70, 70, 70), 1)

    lines = [
        (f"Label: {cfg.gesture_label}", 0.7),
        (f"User: {cfg.user_id}", 0.58),
        (f"Session: {cfg.session_id}", 0.58),
        (f"State: {_state_text(runtime.state)} | Mode: {runtime.capture_mode.value}", 0.58),
        (f"Accepted: {runtime.accepted_samples} | Rejected: {runtime.rejected_attempts}", 0.58),
        (f"Progress: {_progress_text(runtime.accepted_samples, cfg.max_samples)}", 0.58),
        (f"Handedness: {handedness or 'Unknown'}", 0.58),
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
        _draw_text(frame_bgr, "Only record one gesture label per session.", (18, 342), 0.54)

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
    )


def _capture_sample(
    *,
    frame_seq: int,
    cfg: RecorderConfig,
    detected_hand,
    session: RecordingSession,
) -> RecordingSample:
    quality = assess_hand_input_quality(detected_hand.landmarks)
    if not quality.passed:
        raise ValueError(f"Cannot capture sample from rejected hand input: {quality.reason}")

    feature_vector = extract_feature_vector(detected_hand.landmarks)
    sample = _build_sample(
        sample_index=len(session.samples),
        frame_seq=frame_seq,
        cfg=cfg,
        handedness=detected_hand.handedness,
        feature_values=feature_vector.values,
        schema_version=feature_vector.schema_version,
        quality=quality,
    )
    session.samples.append(sample)
    return sample


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


def main() -> None:
    args = build_arg_parser().parse_args()
    capture_context = parse_capture_context(args.capture_context)
    _maybe_validate_readiness(
        capture_context=capture_context,
        skip_check=args.skip_readiness_check,
    )
    output_path = args.output or build_output_path(args.gesture_label, args.user_id, args.session_id)
    cfg = RecorderConfig(
        gesture_label=args.gesture_label,
        user_id=args.user_id,
        session_id=args.session_id,
        output_path=output_path,
        capture_context=capture_context,
        sample_interval_s=float(args.sample_interval),
        max_samples=args.max_samples,
        countdown_seconds=max(0.0, float(args.countdown_seconds)),
        sound_enabled=not args.mute_sounds,
        mirror_view=CONFIG.mirror_view,
        detect_scale=CONFIG.detect_scale,
        enable_preprocessing=CONFIG.enable_preprocessing,
    )

    session = _make_session(cfg)
    camera = Camera(device_index=CONFIG.camera_index, width=CONFIG.cam_width, height=CONFIG.cam_height)
    tracker = HandTracker(max_num_hands=1)
    preprocessor = Preprocessor(enable=cfg.enable_preprocessing)
    camera_thread: ThreadedCamera | None = None
    discarded = False
    save_requested = False
    last_capture_t = 0.0
    window_name = f"{CONFIG.app_name} - Record {cfg.gesture_label}"
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

                should_auto_capture = (
                    runtime.state == RecorderState.RECORDING_AUTO
                    and quality_ok
                    and (now - last_capture_t) >= cfg.sample_interval_s
                )
                if should_auto_capture:
                    sample = _capture_sample(
                        frame_seq=frame_seq,
                        cfg=cfg,
                        detected_hand=detected_hand,
                        session=session,
                    )
                    runtime.accepted_samples = len(session.samples)
                    _set_last_result(runtime, f"Accepted sample #{sample.sample_index}", level="accept")
                    last_capture_t = now
                    if cfg.max_samples is not None and len(session.samples) >= cfg.max_samples:
                        print(f"[Recorder] Reached max_samples={cfg.max_samples}.")
                        _set_last_result(runtime, f"Target reached for {cfg.gesture_label}", level="accept")
                        _update_state(runtime, RecorderState.PAUSED)
                        _play_sound("target_reached", enabled=cfg.sound_enabled)
                elif runtime.state == RecorderState.RECORDING_AUTO and not quality_ok and (now - last_capture_t) >= cfg.sample_interval_s:
                    _record_rejection(runtime, quality_reason, sound_enabled=cfg.sound_enabled)
                    last_capture_t = now

            _draw_overlay(
                frame_disp,
                cfg=cfg,
                session=session,
                runtime=runtime,
                quality_reason=quality_reason,
                quality_ok=quality_ok,
                handedness=handedness,
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
                    session.samples.clear()
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
                else:
                    sample = _capture_sample(
                        frame_seq=frame_seq,
                        cfg=cfg,
                        detected_hand=detected_hand,
                        session=session,
                    )
                    runtime.accepted_samples = len(session.samples)
                    _set_last_result(runtime, f"Accepted sample #{sample.sample_index}", level="accept")
                    last_capture_t = time.perf_counter()
                    print(f"[Recorder] Captured sample #{len(session.samples) - 1}")
            elif key in (ord("u"), ord("U")) and session.samples:
                removed = session.samples.pop()
                runtime.accepted_samples = len(session.samples)
                _set_last_result(runtime, f"Removed sample #{removed.sample_index}", level="info")
                print(f"[Recorder] Removed sample #{removed.sample_index}")
            elif key in (ord("u"), ord("U")):
                _set_last_result(runtime, "Nothing to undo", level="info")
            elif key in (ord("x"), ord("X")):
                discarded = True
                session.samples.clear()
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
        return

    if not save_requested:
        print("[Recorder] Session closed without saving.")
        return

    if not session.samples:
        _play_sound("discard", enabled=cfg.sound_enabled)
        print("[Recorder] No samples captured. Nothing saved.")
        return

    save_session(session, cfg.output_path)
    _play_sound("save", enabled=cfg.sound_enabled)
    print(f"[Recorder] Saved {len(session.samples)} samples to {cfg.output_path}")


if __name__ == "__main__":
    main()
