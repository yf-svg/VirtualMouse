from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GestureAuthCfg:
    sequence: tuple[str, ...] = ("ONE", "TWO", "THREE")
    step_timeout_s: float = 4.0
    reset_gestures: frozenset[str] = field(default_factory=lambda: frozenset({"FIST"}))
    max_failures: int = 3
    cooldown_s: float = 6.0


@dataclass(frozen=True)
class GestureAuthOut:
    authenticated: bool
    status: str
    matched_steps: int
    total_steps: int
    expected_next: str | None
    consumed_label: str | None
    failed_attempts: int
    max_failures: int
    retry_after_s: float | None


class GestureAuth:
    def __init__(self, cfg: GestureAuthCfg | None = None):
        self.cfg = cfg or GestureAuthCfg()
        if not self.cfg.sequence:
            raise ValueError("GestureAuth sequence must not be empty.")
        self.reset()

    def _reset_progress(self) -> None:
        self._matched_steps = 0
        self._authenticated = False
        self._last_progress_at: float | None = None

    def reset(self) -> None:
        self._reset_progress()
        self._failed_attempts = 0
        self._locked_until_s: float | None = None

    def lock(self) -> None:
        self.reset()

    def _current_out(
        self,
        *,
        status: str,
        consumed_label: str | None,
        now: float | None = None,
    ) -> GestureAuthOut:
        expected_next = None
        if not self._authenticated and self._matched_steps < len(self.cfg.sequence):
            expected_next = self.cfg.sequence[self._matched_steps]
        retry_after_s = None
        if self._locked_until_s is not None and now is not None and now < self._locked_until_s:
            retry_after_s = max(0.0, self._locked_until_s - now)
        return GestureAuthOut(
            authenticated=self._authenticated,
            status=status,
            matched_steps=self._matched_steps,
            total_steps=len(self.cfg.sequence),
            expected_next=expected_next,
            consumed_label=consumed_label,
            failed_attempts=self._failed_attempts,
            max_failures=self.cfg.max_failures,
            retry_after_s=retry_after_s,
        )

    def _register_failure(self, *, status: str, consumed_label: str | None, now: float) -> GestureAuthOut:
        self._reset_progress()
        self._failed_attempts += 1
        if self._failed_attempts >= self.cfg.max_failures:
            self._locked_until_s = now + self.cfg.cooldown_s
            return self._current_out(status="locked_out", consumed_label=consumed_label, now=now)
        return self._current_out(status=status, consumed_label=consumed_label, now=now)

    def update(self, gesture_label: str | None, *, now: float | None = None) -> GestureAuthOut:
        now = time.monotonic() if now is None else float(now)

        if self._locked_until_s is not None:
            if now < self._locked_until_s:
                return self._current_out(status="locked_out", consumed_label=gesture_label, now=now)
            self._locked_until_s = None
            self._failed_attempts = 0

        if (
            not self._authenticated
            and self._matched_steps > 0
            and self._last_progress_at is not None
            and (now - self._last_progress_at) > self.cfg.step_timeout_s
        ):
            timeout_out = self._register_failure(status="reset_timeout", consumed_label=None, now=now)
            if gesture_label is None:
                return timeout_out

        if self._authenticated:
            return self._current_out(status="authenticated", consumed_label=gesture_label, now=now)

        if gesture_label is None:
            return self._current_out(
                status="progress" if self._matched_steps > 0 else "idle",
                consumed_label=None,
                now=now,
            )

        if gesture_label in self.cfg.reset_gestures:
            self._reset_progress()
            return self._current_out(status="reset_cancel", consumed_label=gesture_label, now=now)

        expected = self.cfg.sequence[self._matched_steps]
        if gesture_label == expected:
            self._matched_steps += 1
            self._last_progress_at = now
            self._locked_until_s = None
            if self._matched_steps >= len(self.cfg.sequence):
                self._authenticated = True
                self._failed_attempts = 0
                return self._current_out(status="success", consumed_label=gesture_label, now=now)
            return self._current_out(
                status="started" if self._matched_steps == 1 else "progress",
                consumed_label=gesture_label,
                now=now,
            )

        if gesture_label == self.cfg.sequence[0]:
            self._matched_steps = 1
            self._authenticated = False
            self._last_progress_at = now
            return self._current_out(status="started", consumed_label=gesture_label, now=now)

        return self._register_failure(status="reset_wrong", consumed_label=gesture_label, now=now)
