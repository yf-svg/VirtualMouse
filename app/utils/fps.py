import time
from collections import deque


class FPSCounter:
    def __init__(self, avg_window: int = 30):
        self.avg_window = max(1, int(avg_window))
        self._times = deque(maxlen=self.avg_window)
        self._sum_dt = 0.0
        self._last = None

    def tick(self) -> float:
        """Call once per frame. Returns smoothed FPS."""
        now = time.perf_counter()
        if self._last is None:
            self._last = now
            return 0.0

        dt = now - self._last
        self._last = now

        if dt > 0:
            if len(self._times) == self._times.maxlen:
                self._sum_dt -= self._times[0]
            self._times.append(dt)
            self._sum_dt += dt

        if not self._times:
            return 0.0

        avg_dt = self._sum_dt / len(self._times)
        return (1.0 / avg_dt) if avg_dt > 0 else 0.0
