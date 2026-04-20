from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class AuthInputState:
    committed_sequence: tuple[str, ...]
    max_length: int
    accepting_digits: bool
    ready_to_submit: bool
    locked_out: bool


@dataclass(frozen=True)
class GestureAuthCfg:
    sequence: tuple[str, ...] = ("ONE", "TWO", "THREE")
    step_timeout_s: float = 4.0
    reset_gestures: tuple[str, ...] = ("SHAKA",)
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
    committed_sequence: tuple[str, ...]
    buffer_full: bool


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
        return len(self.cfg.sequence)

    @property
    def expected_first(self) -> str:
        return self.cfg.sequence[0]

    @property
    def current_expected_next(self) -> str | None:
        return self._expected_next()

    @property
    def current_input_state(self) -> AuthInputState:
        committed = tuple(self._committed_sequence)
        return AuthInputState(
            committed_sequence=committed,
            max_length=len(self.cfg.sequence),
            accepting_digits=len(committed) < len(self.cfg.sequence),
            ready_to_submit=len(committed) == len(self.cfg.sequence),
            locked_out=self._locked_until_s is not None,
        )

    def _expected_next(self) -> str | None:
        if self._authenticated:
            return None
        if len(self._committed_sequence) < len(self.cfg.sequence):
            return self.cfg.sequence[len(self._committed_sequence)]
        return self.cfg.approve_gestures[0]

    def _reset_progress(self) -> None:
        self._committed_sequence = ()
        self._authenticated = False

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
        committed_sequence = tuple(self._committed_sequence)
        return GestureAuthOut(
            authenticated=self._authenticated,
            status=status,
            matched_steps=len(committed_sequence),
            total_steps=self.total_steps,
            expected_next=self._expected_next(),
            consumed_label=consumed_label,
            failed_attempts=self._failed_attempts,
            max_failures=self.cfg.max_failures,
            retry_after_s=retry_after_s,
            committed_sequence=committed_sequence,
            buffer_full=len(committed_sequence) == len(self.cfg.sequence),
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

        if self._authenticated:
            return self._current_out(status="authenticated", consumed_label=gesture_label, now=now)

        if gesture_label is None:
            return self._current_out(status=self._entry_status(), consumed_label=None, now=now)

        if gesture_label in self.cfg.reset_gestures:
            self._reset_progress()
            return self._current_out(status="reset_cancel", consumed_label=gesture_label, now=now)

        if gesture_label in self.cfg.back_gestures:
            if self._committed_sequence:
                self._committed_sequence = self._committed_sequence[:-1]
                return self._current_out(status="step_back", consumed_label=gesture_label, now=now)
            return self._current_out(status="idle", consumed_label=gesture_label, now=now)

        if gesture_label in self.cfg.approve_gestures:
            if len(self._committed_sequence) < len(self.cfg.sequence):
                return self._current_out(status=self._entry_status(), consumed_label=gesture_label, now=now)
            if tuple(self._committed_sequence) == tuple(self.cfg.sequence):
                self._authenticated = True
                self._failed_attempts = 0
                self._locked_until_s = None
                return self._current_out(status="success", consumed_label=gesture_label, now=now)
            return self._register_failure(status="reset_wrong", consumed_label=gesture_label, now=now)

        if gesture_label not in self.cfg.sequence:
            return self._current_out(status=self._entry_status(), consumed_label=gesture_label, now=now)

        if len(self._committed_sequence) >= len(self.cfg.sequence):
            return self._current_out(status="ready_to_submit", consumed_label=gesture_label, now=now)

        self._committed_sequence = tuple((*self._committed_sequence, gesture_label))
        return self._current_out(
            status="started" if len(self._committed_sequence) == 1 else self._entry_status(),
            consumed_label=gesture_label,
            now=now,
        )

    def _entry_status(self) -> str:
        if not self._committed_sequence:
            return "idle"
        if len(self._committed_sequence) >= len(self.cfg.sequence):
            return "ready_to_submit"
        return "progress"
