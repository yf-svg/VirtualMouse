from __future__ import annotations

import threading
import time
from typing import Optional

import cv2

from app.utils.errors import CameraOpenError, FrameReadError
from app.config import CONFIG


class Camera:
    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480):
        self.device_index = int(device_index)
        self.width = int(width)
        self.height = int(height)
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise CameraOpenError(f"Failed to open camera index {self.device_index}")

        # Prefer MJPG for higher FPS on many webcams
        if getattr(CONFIG, "prefer_mjpg", True):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Request FPS (best-effort)
        req_fps = float(getattr(CONFIG, "requested_fps", 30))
        self.cap.set(cv2.CAP_PROP_FPS, req_fps)

        # Reduce buffering (best-effort)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Exposure policy (best-effort; depends on driver)
        if getattr(CONFIG, "use_auto_exposure", True):
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # often "auto" on DirectShow
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # often "manual"
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(getattr(CONFIG, "exposure_value", -4.0)))
            self.cap.set(cv2.CAP_PROP_GAIN, float(getattr(CONFIG, "gain_value", 10.0)))

        # Print what the camera reports
        w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        ae = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        exp = self.cap.get(cv2.CAP_PROP_EXPOSURE)
        gain = self.cap.get(cv2.CAP_PROP_GAIN)
        print(f"[Camera] {w:.0f}x{h:.0f} @ {fps:.2f} FPS | auto_exp={ae} exp={exp} gain={gain}")

    def read(self):
        if self.cap is None:
            raise FrameReadError("Camera not opened")

        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise FrameReadError("Failed to read frame from camera")
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class ThreadedCamera:
    """
    Continuously grabs frames on a background thread and keeps only the latest.
    This prevents the main loop from blocking on camera timing/buffering.
    """

    def __init__(self, cam: Camera):
        self.cam = cam
        self._lock = threading.Lock()
        self._latest = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def open(self) -> None:
        self.cam.open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                frame = self.cam.read()
                with self._lock:
                    self._latest = frame
            except Exception:
                # avoid tight crash loop
                time.sleep(0.01)

    def read_latest(self):
        with self._lock:
            frame = None if self._latest is None else self._latest.copy()
        if frame is None:
            raise FrameReadError("No frame available yet")
        return frame

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cam.release()