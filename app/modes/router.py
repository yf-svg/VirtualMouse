from __future__ import annotations

import time
import math
from dataclasses import dataclass

from app.constants import AppState
from app.lifecycle.state_machine import StateMachine
from app.security.auth import GestureAuth, GestureAuthOut


@dataclass(frozen=True)
class RouteOut:
    state: AppState
    suite_key: str
    auth_status: str
    auth_progress_text: str
    auth_out: GestureAuthOut


class ModeRouter:
    def __init__(
        self,
        *,
        initial_state: AppState = AppState.IDLE_LOCKED,
        auth: GestureAuth | None = None,
    ):
        self._sm = StateMachine(initial_state)
        self._auth = auth or GestureAuth()

    @property
    def state(self) -> AppState:
        return self._sm.state

    @staticmethod
    def _format_auth_progress(auth_out: GestureAuthOut) -> str:
        if auth_out.status == "locked_out":
            retry_after = max(1, math.ceil(auth_out.retry_after_s or 0.0))
            return f"Auth locked {retry_after}s"
        return f"Auth {auth_out.matched_steps}/{auth_out.total_steps} next:{auth_out.expected_next or 'DONE'}"

    def _idle_auth_out(self, *, status: str = "idle") -> GestureAuthOut:
        return GestureAuthOut(
            authenticated=False,
            status=status,
            matched_steps=0,
            total_steps=len(self._auth.cfg.sequence),
            expected_next=self._auth.cfg.sequence[0],
            consumed_label=None,
            failed_attempts=0,
            max_failures=self._auth.cfg.max_failures,
            retry_after_s=None,
        )

    def _suite_key(self) -> str:
        if self.state in {AppState.IDLE_LOCKED, AppState.AUTHENTICATING}:
            return "auth"
        if self.state == AppState.SLEEP:
            return "sleep"
        return "ops"

    def route_auth_edge(self, gesture_label: str | None, *, now: float | None = None) -> RouteOut:
        auth_out = self._auth.update(gesture_label, now=time.monotonic() if now is None else now)

        if auth_out.status in {"started", "progress"} and self.state == AppState.IDLE_LOCKED:
            self._sm.set_state(AppState.AUTHENTICATING)
        elif auth_out.status in {"reset_wrong", "reset_timeout", "reset_cancel", "locked_out"} and self.state == AppState.AUTHENTICATING:
            self._sm.set_state(AppState.IDLE_LOCKED)
        elif auth_out.status == "success":
            self._sm.set_state(AppState.ACTIVE_GENERAL)
            self._auth.reset()
            auth_out = GestureAuthOut(
                authenticated=False,
                status="success",
                matched_steps=0,
                total_steps=auth_out.total_steps,
                expected_next=None,
                consumed_label=auth_out.consumed_label,
                failed_attempts=0,
                max_failures=auth_out.max_failures,
                retry_after_s=None,
            )

        progress_text = "Auth complete" if auth_out.status == "success" else self._format_auth_progress(auth_out)
        return RouteOut(
            state=self.state,
            suite_key=self._suite_key(),
            auth_status=auth_out.status,
            auth_progress_text=progress_text,
            auth_out=auth_out,
        )

    def lock(self) -> RouteOut:
        self._auth.lock()
        self._sm.set_state(AppState.IDLE_LOCKED)
        auth_out = self._idle_auth_out(status="idle")
        return RouteOut(
            state=self.state,
            suite_key=self._suite_key(),
            auth_status=auth_out.status,
            auth_progress_text=self._format_auth_progress(auth_out),
            auth_out=auth_out,
        )

    def request_sleep(self) -> RouteOut:
        if self.state in {AppState.ACTIVE_GENERAL, AppState.ACTIVE_PRESENTATION}:
            self._sm.set_state(AppState.SLEEP)
        auth_out = self._idle_auth_out(status="sleep")
        return RouteOut(
            state=self.state,
            suite_key=self._suite_key(),
            auth_status="sleep",
            auth_progress_text="Sleeping",
            auth_out=auth_out,
        )

    def wake_for_auth(self) -> RouteOut:
        if self.state == AppState.SLEEP:
            self._auth.lock()
            self._sm.set_state(AppState.AUTHENTICATING)
        auth_out = self._idle_auth_out(status="idle")
        return RouteOut(
            state=self.state,
            suite_key=self._suite_key(),
            auth_status=auth_out.status,
            auth_progress_text=self._format_auth_progress(auth_out),
            auth_out=auth_out,
        )

    def sync_presentation_permission(self, allowed: bool) -> RouteOut:
        if allowed and self.state == AppState.ACTIVE_GENERAL:
            self._sm.set_state(AppState.ACTIVE_PRESENTATION)
        elif not allowed and self.state == AppState.ACTIVE_PRESENTATION:
            self._sm.set_state(AppState.ACTIVE_GENERAL)

        auth_out = self._idle_auth_out(status="presentation_active" if self.state == AppState.ACTIVE_PRESENTATION else "idle")
        return RouteOut(
            state=self.state,
            suite_key=self._suite_key(),
            auth_status=auth_out.status,
            auth_progress_text="Presentation active" if self.state == AppState.ACTIVE_PRESENTATION else self._format_auth_progress(auth_out),
            auth_out=auth_out,
        )
