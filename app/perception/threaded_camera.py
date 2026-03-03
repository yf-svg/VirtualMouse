from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


@dataclass(frozen=True)
class FramePacket:
    frame: "cv2.Mat"
    seq: int
    ts: float


class ThreadedCamera:
    """
    Thread-safe camera capture that keeps ONLY the most recent frame.

    Concurrency model:
      - Capture thread: the ONLY thread that calls cap.read()
      - Main thread: reads the latest snapshot via read_latest()
    """

    def __init__(self, cap: cv2.VideoCapture, *, sleep_on_fail_s: float = 0.005):
        self._cap = cap
        self._sleep_on_fail_s = float(sleep_on_fail_s)

        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._last_ok: bool = False

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seq: int = 0

    def start(self) -> None:
        if self._thread is not None:
            return  # idempotent start

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ThreadedCamera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # Only this thread touches cap.read()
        while not self._stop.is_set():
            ok, frame = self._cap.read()

            ts = time.perf_counter()

            if ok and frame is not None:
                # Update shared state quickly, under a single lock.
                # No heavy work inside lock.
                with self._lock:
                    self._seq += 1
                    self._latest = FramePacket(frame=frame, seq=self._seq, ts=ts)
                    self._last_ok = True
            else:
                with self._lock:
                    self._last_ok = False
                time.sleep(self._sleep_on_fail_s)

    def read_latest(self) -> Tuple[Optional["cv2.Mat"], int, float, bool]:
        """
        Returns a SAFE snapshot: (frame_copy_or_none, seq, timestamp, last_ok).

        - The returned frame is a copy (so caller can draw/modify safely).
        - seq lets caller know if frame changed since last call.
        """
        with self._lock:
            pkt = self._latest
            last_ok = self._last_ok

        if pkt is None:
            return None, 0, 0.0, last_ok

        # Copy OUTSIDE lock to avoid blocking capture thread.
        return pkt.frame.copy(), pkt.seq, pkt.ts, last_ok

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=1.0)
        self._thread = None
