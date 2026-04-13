from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import CONFIG
from app.gestures.engine import GestureEngine
from app.gestures.pointing import PointingDetector
from app.gestures.sets.labels import ALL_ALLOWED_LABELS
from app.perception.camera import Camera
from app.perception.hand_tracker import HandTracker
from app.perception.landmark_smoothing import SelectiveLandmarkSmoother
from app.perception.preprocessing import Preprocessor
from app.perception.threaded_camera import ThreadedCamera
from app.utils.fps import FPSCounter


DEFAULT_ALLOWED_LABELS = tuple(sorted(ALL_ALLOWED_LABELS))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live gesture validator for checking candidate overlap and stable recognition."
    )
    parser.add_argument(
        "--allow-label",
        action="append",
        dest="allowed_labels",
        default=[],
        help="Restrict validation to specific labels. Repeat as needed.",
    )
    return parser


def _mirror_landmarks(landmarks):
    lm2 = type(landmarks)()
    lm2.CopyFrom(landmarks)
    for point in lm2.landmark:
        point.x = 1.0 - point.x
    return lm2


def _format_extra(out, handedness: str | None, quality_reason: str | None) -> str:
    candidates = ",".join(sorted(out.candidates)) if out.candidates else "NONE"
    return (
        f"HAND {handedness or 'Unknown'}  GATE {quality_reason or 'ok'}\n"
        f"CHOSEN {out.decision.active or 'NONE'}  STABLE {out.temporal.stable or 'NONE'}\n"
        f"CAND {candidates}  WHY {out.decision.reason}"
    )


def _format_pointing_debug(detector: PointingDetector, detected) -> str:
    right = detector.analyze(detected, direction="right")
    left = detector.analyze(detected, direction="left")
    return (
        f"PR:{'Y' if right.matched else 'N'} rsn={right.reason} ori={right.orientation or '-'} "
        f"hint={right.handedness_orientation or '-'} vis={right.visible_count} hid={right.hidden_count} "
        f"ext={right.non_index_extended_count} dx={right.display_dx:.2f}\n"
        f"PL:{'Y' if left.matched else 'N'} rsn={left.reason} ori={left.orientation or '-'} "
        f"hint={left.handedness_orientation or '-'} vis={left.visible_count} hid={left.hidden_count} "
        f"ext={left.non_index_extended_count} dx={left.display_dx:.2f}"
    )


def _draw_text(frame_bgr, text: str, xy: tuple[int, int], scale: float = 0.58) -> None:
    cv2.putText(frame_bgr, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def _fit_line(text: str, max_chars: int = 92) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars - 3]}..."


def _draw_debug_panel(frame_bgr, *, fps: float, lines: list[str], allowed: set[str]) -> None:
    height, width = frame_bgr.shape[:2]
    panel_width = min(width - 16, 900)
    panel_height = min(height - 16, 210)
    cv2.rectangle(frame_bgr, (8, 8), (8 + panel_width, 8 + panel_height), (12, 12, 12), -1)
    cv2.rectangle(frame_bgr, (8, 8), (8 + panel_width, 8 + panel_height), (210, 210, 210), 1)

    y = 32
    _draw_text(frame_bgr, f"LIVE VALIDATOR | FPS {fps:.1f} | ESC/Q quit", (18, y), 0.62)
    y += 26
    allowed_text = ",".join(sorted(allowed))
    _draw_text(frame_bgr, _fit_line(f"ALLOWED {allowed_text}"), (18, y), 0.52)
    for line in lines:
        y += 24
        if y > panel_height - 4:
            break
        _draw_text(frame_bgr, _fit_line(line), (18, y), 0.52)


def main() -> None:
    args = build_arg_parser().parse_args()
    allowed = set(args.allowed_labels) if args.allowed_labels else set(DEFAULT_ALLOWED_LABELS)

    cam = Camera(device_index=CONFIG.camera_index, width=CONFIG.cam_width, height=CONFIG.cam_height)
    tracker = HandTracker(max_num_hands=1)
    pre = Preprocessor(enable=CONFIG.enable_preprocessing)
    smoother = SelectiveLandmarkSmoother(
        strong_min_cutoff=1.6,
        strong_beta=0.18,
        tip_min_cutoff=3.5,
        tip_beta=0.45,
        d_cutoff=1.0,
    )
    engine = GestureEngine(allowed=allowed, allow_priority=False)
    pointing_detector = PointingDetector()
    fps_counter = FPSCounter(avg_window=30)
    window_name = f"{CONFIG.app_name} - Live Gesture Validator"
    last_console_debug = 0.0

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(window_name, CONFIG.cam_width, CONFIG.cam_height)
    except Exception:
        pass

    cam.open()
    cam_thread = ThreadedCamera(cam.cap)
    cam_thread.start()

    try:
        while True:
            frame_raw, _seq, _ts, _ok = cam_thread.read_latest()
            if frame_raw is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            frame_disp = cv2.flip(frame_raw, 1) if CONFIG.mirror_view else frame_raw
            small = cv2.resize(
                frame_raw,
                (0, 0),
                fx=CONFIG.detect_scale,
                fy=CONFIG.detect_scale,
                interpolation=cv2.INTER_AREA,
            )
            if CONFIG.enable_preprocessing:
                small = pre.apply(small)

            detected = tracker.detect(small)
            if detected is None:
                smoother.reset()
                engine.reset()
                debug_lines = ["No hand detected"]
            else:
                out = engine.process(detected)
                lm_smooth = smoother.apply(detected.landmarks)
                lm_draw = _mirror_landmarks(lm_smooth) if CONFIG.mirror_view else lm_smooth
                tracker.draw(frame_disp, lm_draw)
                quality_reason = out.quality.reason if out.quality is not None else None
                debug_lines = _format_extra(out, detected.handedness, quality_reason).splitlines()
                if "POINT_RIGHT" in allowed or "POINT_LEFT" in allowed:
                    pointing_debug = _format_pointing_debug(pointing_detector, detected)
                    debug_lines.extend(pointing_debug.splitlines())
                    now = time.perf_counter()
                    if now - last_console_debug >= 1.0:
                        print(pointing_debug)
                        last_console_debug = now

            frame_out = frame_disp.copy()
            _draw_debug_panel(frame_out, fps=fps_counter.tick(), lines=debug_lines, allowed=allowed)
            cv2.imshow(window_name, frame_out)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            time.sleep(0.001)
    finally:
        cam_thread.stop()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
