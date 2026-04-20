from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time

from app.control.cursor_space import CursorPoint


@dataclass(frozen=True)
class TimedPoint:
    point: CursorPoint
    timestamp: float


class PointHistory:
    def __init__(self, *, maxlen: int):
        self._points: deque[TimedPoint] = deque(maxlen=max(2, int(maxlen)))

    def append(self, point: CursorPoint, *, timestamp: float | None = None) -> None:
        self._points.append(
            TimedPoint(
                point=point,
                timestamp=float(time.monotonic() if timestamp is None else timestamp),
            )
        )

    def clear(self) -> None:
        self._points.clear()

    @property
    def last_point(self) -> CursorPoint | None:
        if not self._points:
            return None
        return self._points[-1].point

    def mean_step_distance(self) -> float:
        if len(self._points) < 2:
            return 0.0
        points = tuple(item.point for item in self._points)
        deltas = [
            math.hypot(second.x - first.x, second.y - first.y)
            for first, second in zip(points, points[1:])
        ]
        if not deltas:
            return 0.0
        return sum(deltas) / float(len(deltas))


class AdaptiveEmaFilter:
    def __init__(
        self,
        *,
        history_size: int,
        alpha_min: float,
        alpha_max: float | None = None,
        speed_low: float = 0.0,
        speed_high: float = 1.0,
    ):
        self._history = PointHistory(maxlen=history_size)
        self._default_alpha_min = float(alpha_min)
        self._default_alpha_max = float(alpha_min if alpha_max is None else alpha_max)
        self._default_speed_low = float(speed_low)
        self._default_speed_high = float(speed_high)
        self._smoothed: CursorPoint | None = None

    @property
    def current(self) -> CursorPoint | None:
        return self._smoothed

    def reset(self) -> None:
        self._history.clear()
        self._smoothed = None

    def update(
        self,
        point: CursorPoint,
        *,
        timestamp: float | None = None,
        alpha_min: float | None = None,
        alpha_max: float | None = None,
        speed_low: float | None = None,
        speed_high: float | None = None,
    ) -> CursorPoint:
        self._history.append(point, timestamp=timestamp)
        resolved_alpha = self._resolve_alpha(
            alpha_min=self._default_alpha_min if alpha_min is None else float(alpha_min),
            alpha_max=self._default_alpha_max if alpha_max is None else float(alpha_max),
            speed_low=self._default_speed_low if speed_low is None else float(speed_low),
            speed_high=self._default_speed_high if speed_high is None else float(speed_high),
        )
        previous = self._smoothed
        if previous is None or resolved_alpha >= 1.0:
            smoothed = point
        else:
            smoothed = CursorPoint(
                x=(previous.x * (1.0 - resolved_alpha)) + (point.x * resolved_alpha),
                y=(previous.y * (1.0 - resolved_alpha)) + (point.y * resolved_alpha),
            )
        self._smoothed = smoothed
        return smoothed

    def _resolve_alpha(
        self,
        *,
        alpha_min: float,
        alpha_max: float,
        speed_low: float,
        speed_high: float,
    ) -> float:
        alpha_min = max(0.0, min(1.0, float(alpha_min)))
        alpha_max = max(alpha_min, min(1.0, float(alpha_max)))
        if alpha_min == alpha_max:
            return alpha_max
        motion = self._history.mean_step_distance()
        speed_low = max(0.0, float(speed_low))
        speed_high = max(speed_low + 1e-6, float(speed_high))
        if motion <= speed_low:
            return alpha_min
        if motion >= speed_high:
            return alpha_max
        ratio = (motion - speed_low) / (speed_high - speed_low)
        return alpha_min + ((alpha_max - alpha_min) * ratio)


class DropoutHold:
    def __init__(self, *, hold_frames: int):
        self._hold_frames = max(0, int(hold_frames))
        self._last_point: CursorPoint | None = None
        self._missing_frames = 0

    def reset(self) -> None:
        self._last_point = None
        self._missing_frames = 0

    def apply(self, point: CursorPoint | None) -> CursorPoint | None:
        if point is not None:
            self._last_point = point
            self._missing_frames = 0
            return point
        if self._last_point is None:
            return None
        if self._missing_frames < self._hold_frames:
            self._missing_frames += 1
            return self._last_point
        return None
