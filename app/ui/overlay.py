import cv2

from app.ui.auth_overlay import AuthOverlay


class Overlay:
    def __init__(self):
        self._auth_overlay = AuthOverlay()

    def draw(self, frame_bgr, state_text: str, fps: float, extra: str = "", auth_overlay_state=None):
        h, w = frame_bgr.shape[:2]
        y = 30

        cv2.putText(frame_bgr, f"STATE: {state_text}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"STATE: {state_text}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        y += 30
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if extra:
            for line in str(extra).splitlines():
                y += 30
                cv2.putText(frame_bgr, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame_bgr, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if auth_overlay_state is not None:
            self._auth_overlay.draw(frame_bgr, auth_overlay_state)

        # Footer
        cv2.putText(frame_bgr, "ESC/Q: quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, "ESC/Q: quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
