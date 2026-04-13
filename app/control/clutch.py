from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from app.config import CONFIG, ClutchInteractionConfig
from app.control.actions import ActionIntent, dry_run_action, no_action
from app.control.cursor_space import CursorPoint


class ClutchState(str, Enum):
    NEUTRAL = "NEUTRAL"
    CLUTCH_ACTIVE = "CLUTCH_ACTIVE"
    HAND_LOST_SAFE = "HAND_LOST_SAFE"


@dataclass(frozen=True)
class ClutchOut:
    state: ClutchState
    intent: ActionIntent
    owns_state: bool


class ClutchController:
    """
    Stateful dry-run clutch controller.
    Consumes gated FIST signals only and suppresses lower-priority interaction
    ownership while active.
    """

    def __init__(self, cfg: ClutchInteractionConfig | None = None):
        self.cfg = cfg or CONFIG.clutch_interaction
        self.reset()

    @property
    def state(self) -> ClutchState:
        return self._state

    def reset(self) -> None:
        self._state = ClutchState.NEUTRAL
        self._loss_started_s: float | None = None

    def update(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float | None = None,
    ) -> ClutchOut:
        now = time.monotonic() if now is None else float(now)

        if self._state == ClutchState.HAND_LOST_SAFE:
            return self._update_hand_loss(gesture_label=gesture_label, cursor_point=cursor_point, now=now)
        if self._state == ClutchState.CLUTCH_ACTIVE:
            return self._update_active(gesture_label=gesture_label, cursor_point=cursor_point, now=now)
        return self._update_neutral(gesture_label=gesture_label)

    def _update_neutral(self, *, gesture_label: str | None) -> ClutchOut:
        if gesture_label == "FIST":
            self._state = ClutchState.CLUTCH_ACTIVE
            return ClutchOut(
                state=self._state,
                intent=dry_run_action("CLUTCH_HOLD", gesture_label="FIST", reason="clutch_active"),
                owns_state=True,
            )

        return ClutchOut(
            state=self._state,
            intent=no_action(reason="clutch_idle", gesture_label=gesture_label),
            owns_state=False,
        )

    def _update_active(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float,
    ) -> ClutchOut:
        if gesture_label == "FIST":
            return ClutchOut(
                state=self._state,
                intent=dry_run_action("CLUTCH_HOLD", gesture_label="FIST", reason="clutch_active"),
                owns_state=True,
            )

        if cursor_point is None:
            self._state = ClutchState.HAND_LOST_SAFE
            self._loss_started_s = now
            return ClutchOut(
                state=self._state,
                intent=no_action(reason="clutch_hand_loss_wait", gesture_label="FIST"),
                owns_state=True,
            )

        self._state = ClutchState.NEUTRAL
        self._loss_started_s = None
        return ClutchOut(
            state=self._state,
            intent=dry_run_action("CLUTCH_RELEASE", gesture_label="FIST", reason="clutch_released"),
            owns_state=True,
        )

    def _update_hand_loss(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float,
    ) -> ClutchOut:
        if gesture_label == "FIST" and cursor_point is not None:
            self._state = ClutchState.CLUTCH_ACTIVE
            self._loss_started_s = None
            return ClutchOut(
                state=self._state,
                intent=dry_run_action("CLUTCH_HOLD", gesture_label="FIST", reason="clutch_resumed_after_hand_loss"),
                owns_state=True,
            )

        if self._loss_started_s is not None and (now - self._loss_started_s) <= self.cfg.hand_loss_grace_s:
            return ClutchOut(
                state=self._state,
                intent=no_action(reason="clutch_hand_loss_wait", gesture_label="FIST"),
                owns_state=True,
            )

        self._state = ClutchState.NEUTRAL
        self._loss_started_s = None
        return ClutchOut(
            state=self._state,
            intent=no_action(reason="clutch_cancelled_hand_loss", gesture_label="FIST"),
            owns_state=False,
        )
