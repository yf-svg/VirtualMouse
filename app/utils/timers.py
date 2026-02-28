import time


class StepTimer:
    def __init__(self):
        self._t0 = None
        self.ms = {}

    def start(self):
        self._t0 = time.perf_counter()
        self.ms.clear()

    def mark(self, name: str):
        t = time.perf_counter()
        if self._t0 is None:
            self._t0 = t
        self.ms[name] = (t - self._t0) * 1000.0
        self._t0 = t