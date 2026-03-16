from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import cv2

from app.config import CONFIG
from app.constants import AppState
from app.gestures.suite import GestureSuite
from app.perception.camera import Camera
from app.perception.hand_tracker import HandTracker
from app.perception.landmark_smoothing import SelectiveLandmarkSmoother
from app.perception.preprocessing import Preprocessor
from app.perception.threaded_camera import ThreadedCamera
from app.ui.overlay import Overlay
from app.utils.fps import FPSCounter


@dataclass(slots=True)
class RuntimeContext:
    cam: Camera
    cam_thread: ThreadedCamera
    pre: Preprocessor
    tracker: HandTracker
    smoother: SelectiveLandmarkSmoother
    suite: GestureSuite
    overlay: Overlay
    fps_counter: FPSCounter
    window_name: str
    detect_scale: float
    stride: int


@dataclass(slots=True)
class RuntimeState:
    frame_i: int = 0
    last_hand: Any | None = None
    last_seq_processed: int = -1
    last_frame_disp: Any | None = None
    last_extra: str = "Starting..."
    cam_frames: int = 0
    cam_fps: float = 0.0
    cam_fps_t0: float = 0.0


def _mirror_landmarks(landmarks):
    lm2 = type(landmarks)()
    lm2.CopyFrom(landmarks)
    for p in lm2.landmark:
        p.x = 1.0 - p.x
    return lm2


def _build_runtime_context() -> RuntimeContext:
    cam = Camera(
        device_index=CONFIG.camera_index,
        width=CONFIG.cam_width,
        height=CONFIG.cam_height,
    )
    pre = Preprocessor(enable=CONFIG.enable_preprocessing)
    tracker = HandTracker(max_num_hands=1)
    smoother = SelectiveLandmarkSmoother(
        strong_min_cutoff=1.6,
        strong_beta=0.18,
        tip_min_cutoff=3.5,
        tip_beta=0.45,
        d_cutoff=1.0,
    )
    suite = GestureSuite()
    overlay = Overlay()
    fps_counter = FPSCounter(avg_window=30)

    window_name = f"{CONFIG.app_name} - Gesture Validation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(window_name, CONFIG.cam_width, CONFIG.cam_height)
    except Exception:
        pass

    cam.open()
    cam_thread = ThreadedCamera(cam.cap)
    cam_thread.start()

    return RuntimeContext(
        cam=cam,
        cam_thread=cam_thread,
        pre=pre,
        tracker=tracker,
        smoother=smoother,
        suite=suite,
        overlay=overlay,
        fps_counter=fps_counter,
        window_name=window_name,
        detect_scale=CONFIG.detect_scale,
        stride=max(1, CONFIG.inference_stride),
    )


def _initial_runtime_state() -> RuntimeState:
    return RuntimeState(cam_fps_t0=time.perf_counter())


def _should_exit(key: int) -> bool:
    return key in (27, ord("q"), ord("Q"))


def _update_camera_fps(seq: int, state: RuntimeState) -> None:
    if seq != state.last_seq_processed:
        state.cam_frames += 1

    now = time.perf_counter()
    if now - state.cam_fps_t0 >= 1.0:
        state.cam_fps = state.cam_frames / (now - state.cam_fps_t0)
        state.cam_frames = 0
        state.cam_fps_t0 = now


def _format_hand_overlay(handed: str | None, out) -> str:
    return (
        f"Hand:{handed or 'Unknown'} | "
        f"CHOSEN:{out.chosen or 'NONE'} | "
        f"STABLE:{out.stable or 'NONE'} | "
        f"CAND:{','.join(sorted(out.candidates)) if out.candidates else 'NONE'} | "
        f"EDGE:+{out.down or '-'} -{out.up or '-'} | "
        f"WHY:{out.reason}"
    )

def run_loop(initial_state: AppState) -> None:
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    state_text = initial_state.value
    ctx = _build_runtime_context()
    state = _initial_runtime_state()

    try:
        while True:
            frame_raw, seq, _ts, _ok = ctx.cam_thread.read_latest()

            if frame_raw is None:
                key = cv2.waitKey(1) & 0xFF
                if _should_exit(key):
                    break
                continue

            _update_camera_fps(seq, state)

            if seq == state.last_seq_processed and state.last_frame_disp is not None:
                disp = state.last_frame_disp.copy()
                fps = ctx.fps_counter.tick()
                ctx.overlay.draw(
                    disp,
                    state_text=state_text,
                    fps=fps,
                    extra=f"{state.last_extra} | CAM_FPS:{state.cam_fps:.1f}",
                )
                cv2.imshow(ctx.window_name, disp)
                key = cv2.waitKey(1) & 0xFF
                if _should_exit(key):
                    break
                continue

            state.last_seq_processed = seq

            frame_disp = cv2.flip(frame_raw, 1) if CONFIG.mirror_view else frame_raw

            small = cv2.resize(
                frame_raw,
                (0, 0),
                fx=ctx.detect_scale,
                fy=ctx.detect_scale,
                interpolation=cv2.INTER_AREA,
            )
            if CONFIG.enable_preprocessing:
                small = ctx.pre.apply(small)

            if (state.frame_i % ctx.stride) == 0:
                state.last_hand = ctx.tracker.detect(small)
            state.frame_i += 1

            if state.last_hand:
                lm_raw = state.last_hand.landmarks
                handed = state.last_hand.handedness

                out = ctx.suite.detect(lm_raw)
                lm_smooth = ctx.smoother.apply(lm_raw)
                lm_draw = _mirror_landmarks(lm_smooth) if CONFIG.mirror_view else lm_smooth
                ctx.tracker.draw(frame_disp, lm_draw)
                extra = _format_hand_overlay(handed, out)
            else:
                ctx.smoother.reset()
                ctx.suite.reset()
                extra = "No hand detected"

            fps = ctx.fps_counter.tick()

            frame_out = frame_disp.copy()
            ctx.overlay.draw(
                frame_out,
                state_text=state_text,
                fps=fps,
                extra=f"{extra} | CAM_FPS:{state.cam_fps:.1f}",
            )

            state.last_frame_disp = frame_out
            state.last_extra = extra

            cv2.imshow(ctx.window_name, frame_out)
            key = cv2.waitKey(1) & 0xFF
            if _should_exit(key):
                break

    finally:
        ctx.cam_thread.stop()
        ctx.tracker.close()
        ctx.cam.release()
        cv2.destroyAllWindows()
