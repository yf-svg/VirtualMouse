from __future__ import annotations

import math
import time


class OneEuroFilter:
    """
    One Euro Filter for a single scalar signal.
    Reference: Casiez et al. "The 1€ Filter".
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        # alpha = 1 / (1 + tau/dt), tau = 1/(2*pi*cutoff)
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

        # Derivative of the signal
        dx = (x - self._x_prev) / max(dt, 1e-6)

        # Filter the derivative
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        # Adaptive cutoff
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Filter the signal
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


class LandmarkSmoother:
    """
    Applies One Euro filtering to all 21 hand landmarks (x,y) in normalized coords.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.08, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)

        self._filters_x = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(21)]
        self._filters_y = [OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(21)]

    def reset(self) -> None:
        self._filters_x = [OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff) for _ in range(21)]
        self._filters_y = [OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff) for _ in range(21)]

    def apply(self, landmarks):
        """
        landmarks: MediaPipe landmark proto with .landmark list (len=21).
        returns: new landmarks proto with filtered x,y (z unchanged).
        """
        t = time.perf_counter()

        out = type(landmarks)()
        out.CopyFrom(landmarks)

        for i, p in enumerate(out.landmark):
            p.x = float(self._filters_x[i].filter(p.x, t))
            p.y = float(self._filters_y[i].filter(p.y, t))
            # leave z as-is (or filter later if needed)

        return out