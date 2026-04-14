from __future__ import annotations

import math
import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GestureAuthCfg:
    sequence: tuple[str, ...] = ("ONE", "TWO", "THREE")
    step_timeout_s: float = 4.0
    reset_gestures: tuple[str, ...] = ("FIST",)
    approve_gestures: tuple[str, ...] = ("BRAVO",)
    back_gestures: tuple[str, ...] = ("THUMBS_DOWN",)
    max_failures: int = 5
    cooldown_s: float = 10.0


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
        if not self.cfg.approve_gestures:
            raise ValueError("GestureAuth approve_gestures must not be empty.")
        self.reset()

    @property
    def total_steps(self) -> int:
        return len(self.cfg.sequence) + 1

    @property
    def expected_first(self) -> str:
        return self.cfg.sequence[0]

    @property
    def current_expected_next(self) -> str | None:
        return self._expected_next()

    def _expected_next(self) -> str | None:
        if self._authenticated:
            return None
        if self._matched_steps < len(self.cfg.sequence):
            return self.cfg.sequence[self._matched_steps]
        return self.cfg.approve_gestures[0]

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
        retry_after_s = None
        if self._locked_until_s is not None and now is not None and now < self._locked_until_s:
            retry_after_s = max(0.0, self._locked_until_s - now)
        return GestureAuthOut(
            authenticated=self._authenticated,
            status=status,
            matched_steps=self._matched_steps,
            total_steps=self.total_steps,
            expected_next=self._expected_next(),
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

        if gesture_label in self.cfg.back_gestures:
            if self._matched_steps > 0:
                self._matched_steps -= 1
                self._authenticated = False
                self._last_progress_at = now
                return self._current_out(status="step_back", consumed_label=gesture_label, now=now)
            return self._current_out(status="idle", consumed_label=gesture_label, now=now)

        if self._matched_steps >= len(self.cfg.sequence):
            if gesture_label in self.cfg.approve_gestures:
                self._matched_steps += 1
                self._authenticated = True
                self._failed_attempts = 0
                self._last_progress_at = now
                self._locked_until_s = None
                return self._current_out(status="success", consumed_label=gesture_label, now=now)
            self._last_progress_at = now
            return self._current_out(status="progress", consumed_label=gesture_label, now=now)

        if gesture_label in self.cfg.approve_gestures:
            return self._current_out(
                status="progress" if self._matched_steps > 0 else "idle",
                consumed_label=gesture_label,
                now=now,
            )

        if self._matched_steps < len(self.cfg.sequence):
            expected = self.cfg.sequence[self._matched_steps]
            if gesture_label == expected:
                self._matched_steps += 1
                self._last_progress_at = now
                self._locked_until_s = None
                return self._current_out(
                    status="started" if self._matched_steps == 1 else "progress",
                    consumed_label=gesture_label,
                    now=now,
                )

            return self._register_failure(status="reset_wrong", consumed_label=gesture_label, now=now)

        return self._register_failure(status="reset_wrong", consumed_label=gesture_label, now=now)
