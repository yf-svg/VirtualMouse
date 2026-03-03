from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """
    Fast detection-preprocessor intended to run on a *small* frame.
    Adaptive for dim, normal, and bright scenes:
    - dynamic gamma correction (LUT)
    - dynamic CLAHE on luminance
    - mild highlight compression in very bright scenes
    - light denoising in very dark scenes
    """

    def __init__(self, enable: bool = True, target_mean: float = 122.0):
        self.enable = bool(enable)
        self.target_mean = float(target_mean)
        self._clahe_cache: dict[tuple[float, int], cv2.CLAHE] = {}
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

    def _clahe(self, clip_limit: float, tile_size: int):
        key = (round(float(clip_limit), 1), int(tile_size))
        if key not in self._clahe_cache:
            self._clahe_cache[key] = cv2.createCLAHE(
                clipLimit=key[0],
                tileGridSize=(key[1], key[1]),
            )
        return self._clahe_cache[key]

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self.enable:
            return frame_bgr

        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]
        mean_luma = float(np.mean(y))
        std_luma = float(np.std(y))

        if mean_luma < 1.0:
            return frame_bgr

        # Adaptive gamma:
        # brighten dark scenes, leave normal scenes mostly alone,
        # slightly compress bright scenes.
        if mean_luma < 85.0:
            gamma = 1.45
        elif mean_luma < 115.0:
            gamma = 1.20
        elif mean_luma > 180.0:
            gamma = 0.88
        else:
            gamma = self.target_mean / mean_luma
        gamma = float(np.clip(gamma, 0.88, 1.55))
        out = cv2.LUT(frame_bgr, self._gamma_lut(gamma))

        ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0]

        # Adaptive local contrast:
        # stronger in flat/dim images, milder in already-contrasty scenes.
        if mean_luma < 95.0 or std_luma < 40.0:
            clip_limit = 2.6
            tile_size = 8
        elif mean_luma > 180.0:
            clip_limit = 1.6
            tile_size = 8
        else:
            clip_limit = 2.0
            tile_size = 8

        y = self._clahe(clip_limit, tile_size).apply(y)

        # Compress highlights slightly in very bright scenes so skin detail survives.
        if mean_luma > 190.0:
            y = np.minimum(y.astype(np.float32) * 0.94, 255.0).astype(np.uint8)

        ycrcb[:, :, 0] = y
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        # Light denoising only when the frame is genuinely dark.
        if mean_luma < 80.0:
            out = cv2.bilateralFilter(out, d=5, sigmaColor=20, sigmaSpace=20)

        return out
