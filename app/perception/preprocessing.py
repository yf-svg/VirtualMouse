from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """
    Fast background robustness pipeline (CPU-friendly):
    - Gray-world white balance
    - CLAHE on luminance
    - Light Gaussian blur (optional)
    """

    def __init__(self, enable: bool = True, blur: bool = False):
        self.enable = bool(enable)
        self.blur = bool(blur)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def _gray_world_white_balance(frame_bgr: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(frame_bgr.astype(np.float32))
        mean_b, mean_g, mean_r = float(np.mean(b)), float(np.mean(g)), float(np.mean(r))
        mean_gray = (mean_b + mean_g + mean_r) / 3.0

        scale_b = mean_gray / (mean_b + 1e-6)
        scale_g = mean_gray / (mean_g + 1e-6)
        scale_r = mean_gray / (mean_r + 1e-6)

        b = np.clip(b * scale_b, 0, 255)
        g = np.clip(g * scale_g, 0, 255)
        r = np.clip(r * scale_r, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)

    def apply(self, frame_bgr):
        if not self.enable:
            return frame_bgr

        wb = self._gray_world_white_balance(frame_bgr)

        lab = cv2.cvtColor(wb, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = self._clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        if self.blur:
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return enhanced