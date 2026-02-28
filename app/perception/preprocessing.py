from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """
    Fast detection-preprocessor intended to run on a *small* frame.
    - mild gamma correction (LUT)
    - mild CLAHE on luminance
    """

    def __init__(self, enable: bool = True, target_mean: float = 110.0):
        self.enable = bool(enable)
        self.target_mean = float(target_mean)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._lut_cache: dict[int, np.ndarray] = {}

    @staticmethod
    def _mean_luma_y(frame_bgr: np.ndarray) -> float:
        # Compute mean luminance quickly from Y channel
        y = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        return float(np.mean(y))

    def _gamma_lut(self, gamma: float) -> np.ndarray:
        # Quantize gamma so cache is effective and we don't rebuild LUT every frame.
        # 80..180 corresponds to gamma 0.80..1.80
        gq = int(np.clip(round(gamma * 100), 80, 180))
        if gq in self._lut_cache:
            return self._lut_cache[gq]

        g = gq / 100.0
        inv = 1.0 / g
        lut = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.uint8)
        self._lut_cache[gq] = lut
        return lut

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.enable:
            return frame_bgr

        mean_luma = self._mean_luma_y(frame_bgr)
        if mean_luma < 1.0:
            gamma = 1.0
        else:
            gamma = self.target_mean / mean_luma

        gamma = float(np.clip(gamma, 0.8, 1.8))
        out = cv2.LUT(frame_bgr, self._gamma_lut(gamma))

        # CLAHE on luminance (mild)
        ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        ycrcb[:, :, 0] = self._clahe.apply(y)
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return out