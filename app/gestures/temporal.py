from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Optional, Dict


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


class TemporalFilter:
    """
    Majority-vote temporal stabilizer for a single label stream.
    Input: chosen label per frame (or None).
    Output: stable label + edges.
    """

    def __init__(self, cfg: TemporalCfg = TemporalCfg()):
        self.cfg = cfg
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