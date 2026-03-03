from __future__ import annotations

import math
import time


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + (tau / max(dt, 1e-6)))

    def filter(self, x: float, t: float) -> float:
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            self._dx_prev = 0.0
            return x

        dt = t - self._t_prev
        self._t_prev = t

        dx = (x - self._x_prev) / max(dt, 1e-6)

        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


class SelectiveLandmarkSmoother:
    """
    One Euro filtering with different strengths for fingertip landmarks vs the rest.

    - Fingertips (thumb/index/middle tips) get LIGHT filtering -> accurate pinch.
    - The rest get stronger filtering -> stable skeleton.
    """

    TIP_IDS = {4, 8, 12}  # thumb_tip, index_tip, middle_tip

    def __init__(
        self,
        # strong smoothing (palm + joints)
        strong_min_cutoff: float = 1.6,
        strong_beta: float = 0.18,
        # light smoothing (tips for pinch accuracy)
        tip_min_cutoff: float = 3.5,
        tip_beta: float = 0.45,
        d_cutoff: float = 1.0,
    ):
        self.strong_min_cutoff = float(strong_min_cutoff)
        self.strong_beta = float(strong_beta)
        self.tip_min_cutoff = float(tip_min_cutoff)
        self.tip_beta = float(tip_beta)
        self.d_cutoff = float(d_cutoff)

        self._fx = []
        self._fy = []
        self.reset()

    def reset(self) -> None:
        self._fx.clear()
        self._fy.clear()
        for i in range(21):
            if i in self.TIP_IDS:
                self._fx.append(OneEuroFilter(self.tip_min_cutoff, self.tip_beta, self.d_cutoff))
                self._fy.append(OneEuroFilter(self.tip_min_cutoff, self.tip_beta, self.d_cutoff))
            else:
                self._fx.append(OneEuroFilter(self.strong_min_cutoff, self.strong_beta, self.d_cutoff))
                self._fy.append(OneEuroFilter(self.strong_min_cutoff, self.strong_beta, self.d_cutoff))

    def apply(self, landmarks):
        t = time.perf_counter()
        out = type(landmarks)()
        out.CopyFrom(landmarks)

        for i, p in enumerate(out.landmark):
            p.x = float(self._fx[i].filter(p.x, t))
            p.y = float(self._fy[i].filter(p.y, t))
            # z left untouched (you can filter z later if needed)
        return out
