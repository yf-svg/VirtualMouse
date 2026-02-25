from __future__ import annotations

import cv2

from app.config import CONFIG
from app.constants import AppState
from app.perception.camera import Camera
from app.perception.preprocessing import Preprocessor
from app.perception.hand_tracker import HandTracker
from app.ui.overlay import Overlay
from app.utils.fps import FPSCounter


def run_loop(initial_state: AppState) -> None:
    cv2.setUseOptimized(True)

    cam = Camera(
        device_index=CONFIG.camera_index,
        width=CONFIG.cam_width,
        height=CONFIG.cam_height,
    )
    pre = Preprocessor(enable=CONFIG.enable_preprocessing, blur=False)
    tracker = HandTracker(max_num_hands=1)
    overlay = Overlay()
    fps_counter = FPSCounter(avg_window=30)

    window_name = f"{CONFIG.app_name} - Phase 1"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    cam.open()
    try:
        while True:
            frame = cam.read()

            # Mirror view (selfie-style) for cursor feel
            if CONFIG.mirror_view:
                frame = cv2.flip(frame, 1)

            processed = pre.apply(frame)

            # Run detection on a smaller frame for speed
            small = cv2.resize(
                processed, (0, 0),
                fx=0.33, fy=0.33,
                interpolation=cv2.INTER_AREA
            )

            hand = tracker.detect(small)

            if hand:
                # Draw landmarks on display frame (landmarks are normalized, OK)
                tracker.draw(frame, hand)

                handed = hand.get("handedness", None)

                # Mirror view swaps visual left/right, so invert label to match what you SEE
                if CONFIG.mirror_view and handed in ("Left", "Right"):
                    handed = "Right" if handed == "Left" else "Left"

                extra = f"Hand: {handed or 'Unknown'}"
            else:
                extra = "No hand detected"

            fps = fps_counter.tick()
            overlay.draw(frame, state_text=initial_state.value, fps=fps, extra=extra)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()

 