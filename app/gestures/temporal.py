from __future__ import annotations

import statistics
from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Dict

from app.gestures.features import FeatureVector


@dataclass
class TemporalCfg:
    window: int = 7           # rolling window length (frames)
    confirm: int = 4          # votes needed to switch stable label
    min_hold: int = 2         # frames to keep stable after switching (anti-flicker)


@dataclass
class TemporalOut:
    stable: Optional[str]
    down: Optional[str]       # None->gesture edge
    up: Optional[str]         # gesture->None edge
    changed: bool


@dataclass
class FeatureTemporalCfg:
    window: int = 5
    min_frames: int = 3
    instability_threshold: float = 0.12


@dataclass
class FeatureTemporalOut:
    smoothed: Optional[FeatureVector]
    ready: bool
    window_size: int
    schema_version: Optional[str]
    instability_score: float
    pairwise_delta_median: float
    latest_deviation: float
    passed: bool
    reason: str


class TemporalFilter:
    """
    Majority-vote temporal stabilizer for a single label stream.
    Input: chosen label per frame (or None).
    Output: stable label + edges.
    """

    def __init__(self, cfg: TemporalCfg | None = None):
        self.cfg = cfg or TemporalCfg()
        self.reset()

    def reset(self) -> None:
        self._buf: Deque[Optional[str]] = deque(maxlen=self.cfg.window)
        self._stable: Optional[str] = None
        self._hold: int = 0

    def _counts(self) -> Dict[Optional[str], int]:
        counts: Dict[Optional[str], int] = {}
        for x in self._buf:
            counts[x] = counts.get(x, 0) + 1
        return counts

    def update(self, chosen: Optional[str]) -> TemporalOut:
        self._buf.append(chosen)

        prev = self._stable
        down = None
        up = None
        changed = False

        # prevent rapid switching right after a change
        if self._hold > 0:
            self._hold -= 1
            return TemporalOut(stable=self._stable, down=None, up=None, changed=False)

        counts = self._counts()

        # Find best non-None label (most votes)
        best_label = None
        best_votes = 0
        for k, v in counts.items():
            if k is None:
                continue
            if v > best_votes:
                best_label, best_votes = k, v

        # Candidate stable target:
        # - if we have a strong enough best label -> that label
        # - else -> None (release)
        target: Optional[str]
        if best_label is not None and best_votes >= self.cfg.confirm:
            target = best_label
        else:
            # release only if None dominates enough
            none_votes = counts.get(None, 0)
            if none_votes >= self.cfg.confirm:
                target = None
            else:
                target = self._stable  # keep current

        if target != self._stable:
            self._stable = target
            changed = True
            self._hold = self.cfg.min_hold

            if prev is None and target is not None:
                down = target
            elif prev is not None and target is None:
                up = prev

        return TemporalOut(stable=self._stable, down=down, up=up, changed=changed)


class FeatureWindow:
    """
    Sliding-window smoother for fixed-length feature vectors.
    Uses a per-dimension rolling median, which is robust to single-frame landmark
    spikes and is a better fit for noisy hand geometry than a plain mean.
    """

    def __init__(self, cfg: FeatureTemporalCfg | None = None):
        self.cfg = cfg or FeatureTemporalCfg()
        self.reset()

    def reset(self) -> None:
        self._buf: Deque[FeatureVector] = deque(maxlen=self.cfg.window)
        self._schema_version: Optional[str] = None
        self._dimension: Optional[int] = None

    def _smoothed_vector(self) -> Optional[FeatureVector]:
        if not self._buf:
            return None

        columns = zip(*(fv.values for fv in self._buf))
        values = tuple(float(statistics.median(col)) for col in columns)
        return FeatureVector(
            values=values,
            schema_version=self._schema_version or self._buf[-1].schema_version,
        )

    @staticmethod
    def _mean_abs_delta(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        if not left:
            return 0.0
        return sum(abs(a - b) for a, b in zip(left, right)) / len(left)

    def _instability_components(self, smoothed: FeatureVector) -> tuple[float, float, float]:
        if len(self._buf) < 2:
            return 0.0, 0.0, 0.0

        buf = list(self._buf)
        pairwise_deltas = [
            self._mean_abs_delta(curr.values, prev.values)
            for prev, curr in zip(buf, buf[1:])
        ]
        latest_deviation = self._mean_abs_delta(buf[-1].values, smoothed.values)
        pairwise_delta_median = float(statistics.median(pairwise_deltas))
        return (
            pairwise_delta_median,
            latest_deviation,
            max(pairwise_delta_median, latest_deviation),
        )

    def update(self, feature_vector: Optional[FeatureVector]) -> FeatureTemporalOut:
        if feature_vector is None:
            self.reset()
            return FeatureTemporalOut(
                smoothed=None,
                ready=False,
                window_size=0,
                schema_version=None,
                instability_score=0.0,
                pairwise_delta_median=0.0,
                latest_deviation=0.0,
                passed=False,
                reason="none",
            )

        if self._schema_version is None:
            self._schema_version = feature_vector.schema_version
            self._dimension = feature_vector.dimension
        elif (
            feature_vector.schema_version != self._schema_version
            or feature_vector.dimension != self._dimension
        ):
            raise ValueError(
                "Feature window received an incompatible feature-vector schema"
            )

        self._buf.append(feature_vector)
        smoothed = self._smoothed_vector()
        pairwise_delta_median, latest_deviation, score = (
            self._instability_components(smoothed) if smoothed is not None else (0.0, 0.0, 0.0)
        )
        ready = len(self._buf) >= self.cfg.min_frames
        passed = ready and score <= self.cfg.instability_threshold
        reason = "ok" if passed else ("warming_up" if not ready else "unstable")
        return FeatureTemporalOut(
            smoothed=smoothed,
            ready=ready,
            window_size=len(self._buf),
            schema_version=self._schema_version,
            instability_score=score,
            pairwise_delta_median=pairwise_delta_median,
            latest_deviation=latest_deviation,
            passed=passed,
            reason=reason,
        )
