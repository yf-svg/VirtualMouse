# import cv2


# class Webcam:
#     def __init__(self, camera_index=0, width=640, height=480):
#         self.camera_index = camera_index
#         self.width = width
#         self.height = height
#         self.cap = None

#     def start(self):
#         self.cap = cv2.VideoCapture(self.camera_index)

#         if not self.cap.isOpened():
#             raise RuntimeError("❌ Cannot open webcam")

#         # Set resolution (important for stability)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

#         print("✅ Webcam started")

#     def read(self):
#         if self.cap is None:
#             raise RuntimeError("❌ Webcam not started")

#         ret, frame = self.cap.read()
#         if not ret:
#             return None

#         return frame

#     def stop(self):
#         if self.cap:
#             self.cap.release()
#             cv2.destroyAllWindows()
#             print("🛑 Webcam stopped")


from __future__ import annotations

import cv2

from app.utils.errors import CameraOpenError, FrameReadError


class Camera:
    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720):
        self.device_index = int(device_index)
        self.width = int(width)
        self.height = int(height)
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if not self.cap.isOpened():
            raise CameraOpenError(f"Failed to open camera index {self.device_index}")

        # Try to set resolution (may not be honored by all cameras)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

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