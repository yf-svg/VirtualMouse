from __future__ import annotations

import cv2

from app.config import CONFIG
from app.constants import AppState
from app.perception.camera import Camera
from app.perception.preprocessing import Preprocessor
from app.perception.hand_tracker import HandTracker
from app.perception.landmark_smoothing import LandmarkSmoother
from app.ui.overlay import Overlay
from app.utils.fps import FPSCounter
from app.gestures.pinch import detect_pinch_type


def _mirror_landmarks(landmarks):
    """Mirror normalized landmarks for selfie-style display."""
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

    cam = Camera(
        device_index=CONFIG.camera_index,
        width=CONFIG.cam_width,
        height=CONFIG.cam_height,
    )

    # Preprocessing is applied ONLY on small detection frames
    pre = Preprocessor(enable=CONFIG.enable_preprocessing)

    tracker = HandTracker(max_num_hands=1)

    # One Euro Filter for drawing stability (do NOT use for gesture logic)
    smoother = LandmarkSmoother(
        min_cutoff=1.6,
        beta=0.18,
        d_cutoff=1.0,
    )

    overlay = Overlay()
    fps_counter = FPSCounter(avg_window=30)

    window_name = f"{CONFIG.app_name} - Phase 1"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    detect_scale = CONFIG.detect_scale
    stride = max(1, CONFIG.inference_stride)
    frame_i = 0

    last_hand = None

    cam.open()
    try:
        while True:
            frame_raw = cam.read()

            # Display frame (mirrored for cursor feel)
            frame_disp = cv2.flip(frame_raw, 1) if CONFIG.mirror_view else frame_raw

            # Detection path (UNMIRRORED, SMALL)
            small = cv2.resize(
                frame_raw,
                (0, 0),
                fx=detect_scale,
                fy=detect_scale,
                interpolation=cv2.INTER_AREA,
            )

            if CONFIG.enable_preprocessing:
                small = pre.apply(small)

            # Inference stride
            if (frame_i % stride) == 0:
                last_hand = tracker.detect(small)
            frame_i += 1

            # Drawing + pinch classification
            if last_hand:
                lm_raw = last_hand["landmarks"]
                handed = last_hand.get("handedness", None)

                # Gesture recognition on RAW landmarks (accurate, no lag)
                pinch = detect_pinch_type(lm_raw)

                # Smooth ONLY for drawing (stable skeleton)
                lm_smooth = smoother.apply(lm_raw)

                lm_draw = _mirror_landmarks(lm_smooth) if CONFIG.mirror_view else lm_smooth
                tracker.draw(frame_disp, lm_draw)

                extra = f"Hand: {handed or 'Unknown'} | {pinch or 'NO_PINCH'}"
            else:
                smoother.reset()
                extra = "No hand detected"

            fps = fps_counter.tick()
            overlay.draw(
                frame_disp,
                state_text=initial_state.value,
                fps=fps,
                extra=extra,
            )

            cv2.imshow(window_name, frame_disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()