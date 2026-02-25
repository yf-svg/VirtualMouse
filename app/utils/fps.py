import time


class FPSCounter:
    def __init__(self, avg_window: int = 30):
        self.avg_window = max(1, int(avg_window))
        self._times = []
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
            self._times.append(dt)
            if len(self._times) > self.avg_window:
                self._times.pop(0)

        if not self._times:
            return 0.0

        avg_dt = sum(self._times) / len(self._times)
        return (1.0 / avg_dt) if avg_dt > 0 else 0.0