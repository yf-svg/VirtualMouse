from __future__ import annotations

import time
import cv2

from app.config import CONFIG
from app.constants import AppState
from app.perception.camera import Camera
from app.perception.threaded_camera import ThreadedCamera
from app.perception.preprocessing import Preprocessor
from app.perception.hand_tracker import HandTracker
from app.perception.landmark_smoothing import SelectiveLandmarkSmoother

# 🔑 Validation pipeline
from app.gestures.suite import GestureSuite

from app.ui.overlay import Overlay
from app.utils.fps import FPSCounter


def _mirror_landmarks(landmarks):
    lm2 = type(landmarks)()
    lm2.CopyFrom(landmarks)
    for p in lm2.landmark:
        p.x = 1.0 - p.x
    return lm2


def run_loop(initial_state: AppState) -> None:
    cv2.setUseOptimized(True)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    # ---------------- Camera & Perception ----------------
    cam = Camera(
        device_index=CONFIG.camera_index,
        width=CONFIG.cam_width,
        height=CONFIG.cam_height,
    )
    pre = Preprocessor(enable=CONFIG.enable_preprocessing)
    tracker = HandTracker(max_num_hands=1)

    # Smooth ONLY for drawing
    smoother = SelectiveLandmarkSmoother(
        strong_min_cutoff=1.6,
        strong_beta=0.18,
        tip_min_cutoff=3.5,
        tip_beta=0.45,
        d_cutoff=1.0,
    )

    # ---------------- Gesture Validation Suite ----------------
    suite = GestureSuite()

    overlay = Overlay()
    fps_counter = FPSCounter(avg_window=30)

    window_name = f"{CONFIG.app_name} - Gesture Validation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(window_name, CONFIG.cam_width, CONFIG.cam_height)
    except Exception:
        pass

    detect_scale = CONFIG.detect_scale
    stride = max(1, CONFIG.inference_stride)

    frame_i = 0
    last_hand = None

    last_seq_processed = -1
    last_frame_disp = None
    last_extra = "Starting..."

    cam_frames = 0
    cam_fps = 0.0
    cam_fps_t0 = time.perf_counter()

    cam.open()
    cam_thread = ThreadedCamera(cam.cap)
    cam_thread.start()

    try:
        while True:
            frame_raw, seq, _ts, _ok = cam_thread.read_latest()

            if frame_raw is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            # Camera FPS
            if seq != last_seq_processed:
                cam_frames += 1
            now = time.perf_counter()
            if now - cam_fps_t0 >= 1.0:
                cam_fps = cam_frames / (now - cam_fps_t0)
                cam_frames = 0
                cam_fps_t0 = now

            # Duplicate frame → redraw
            if seq == last_seq_processed and last_frame_disp is not None:
                disp = last_frame_disp.copy()
                fps = fps_counter.tick()
                overlay.draw(
                    disp,
                    state_text="GESTURE_VALIDATION",
                    fps=fps,
                    extra=f"{last_extra} | CAM_FPS:{cam_fps:.1f}",
                )
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            last_seq_processed = seq

            # Display frame
            frame_disp = cv2.flip(frame_raw, 1) if CONFIG.mirror_view else frame_raw

            # Detection frame (unmirrored)
            small = cv2.resize(
                frame_raw,
                (0, 0),
                fx=detect_scale,
                fy=detect_scale,
                interpolation=cv2.INTER_AREA,
            )
            if CONFIG.enable_preprocessing:
                small = pre.apply(small)

            if (frame_i % stride) == 0:
                last_hand = tracker.detect(small)
            frame_i += 1

            if last_hand:
                lm_raw = last_hand["landmarks"]
                handed = last_hand.get("handedness", None)

                # 🔍 VALIDATION OUTPUT
                out = suite.detect(lm_raw)

                # Draw landmarks
                lm_smooth = smoother.apply(lm_raw)
                lm_draw = _mirror_landmarks(lm_smooth) if CONFIG.mirror_view else lm_smooth
                tracker.draw(frame_disp, lm_draw)

                extra = (
                    f"Hand:{handed or 'Unknown'} | "
                    f"CHOSEN:{out.chosen or 'NONE'} | "
                    f"STABLE:{out.stable or 'NONE'} | "
                    f"CAND:{','.join(sorted(out.candidates)) if out.candidates else 'NONE'} | "
                    f"EDGE:+{out.down or '-'} -{out.up or '-'} | "
                    f"WHY:{out.reason}"
                )

            else:
                smoother.reset()
                suite.reset()
                extra = "No hand detected"

            fps = fps_counter.tick()

            frame_out = frame_disp.copy()
            overlay.draw(
                frame_out,
                state_text="GESTURE_VALIDATION",
                fps=fps,
                extra=f"{extra} | CAM_FPS:{cam_fps:.1f}",
            )

            last_frame_disp = frame_out
            last_extra = extra

            cv2.imshow(window_name, frame_out)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break

    finally:
        cam_thread.stop()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
