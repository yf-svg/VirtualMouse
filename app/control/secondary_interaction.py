from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from app.config import CONFIG, SecondaryInteractionConfig
from app.control.actions import ActionIntent, dry_run_action, no_action
from app.control.cursor_space import CursorPoint, cursor_distance


class SecondaryInteractionState(str, Enum):
    NEUTRAL = "NEUTRAL"
    SECONDARY_PINCH_CANDIDATE = "SECONDARY_PINCH_CANDIDATE"
    HAND_LOST_SAFE = "HAND_LOST_SAFE"


@dataclass(frozen=True)
class SecondaryInteractionOut:
    state: SecondaryInteractionState
    intent: ActionIntent
    owns_state: bool
    movement: float


class SecondaryInteractionController:
    """
    Stateful dry-run implementation of the approved PINCH_MIDDLE right-click path.
    It consumes gated gesture labels plus abstract cursor-space movement only.
    """

    def __init__(self, cfg: SecondaryInteractionConfig | None = None):
        self.cfg = cfg or CONFIG.secondary_interaction
        self.reset()

    @property
    def state(self) -> SecondaryInteractionState:
        return self._state

    def reset(self) -> None:
        self._state = SecondaryInteractionState.NEUTRAL
        self._candidate_anchor: CursorPoint | None = None
        self._candidate_max_movement = 0.0
        self._loss_started_s: float | None = None

    def update(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float | None = None,
    ) -> SecondaryInteractionOut:
        now = time.monotonic() if now is None else float(now)

        if self._state == SecondaryInteractionState.HAND_LOST_SAFE:
            return self._update_hand_loss(gesture_label=gesture_label, cursor_point=cursor_point, now=now)
        if self._state == SecondaryInteractionState.SECONDARY_PINCH_CANDIDATE:
            return self._update_candidate(gesture_label=gesture_label, cursor_point=cursor_point, now=now)
        return self._update_neutral(gesture_label=gesture_label, cursor_point=cursor_point)

    def _update_neutral(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
    ) -> SecondaryInteractionOut:
        if gesture_label == "PINCH_MIDDLE" and cursor_point is not None:
            self._state = SecondaryInteractionState.SECONDARY_PINCH_CANDIDATE
            self._candidate_anchor = cursor_point
            self._candidate_max_movement = 0.0
            return self._candidate_out(reason="secondary_candidate_started")

        return SecondaryInteractionOut(
            state=self._state,
            intent=no_action(reason="secondary_idle", gesture_label=gesture_label),
            owns_state=False,
            movement=0.0,
        )

    def _update_candidate(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float,
    ) -> SecondaryInteractionOut:
        if gesture_label == "PINCH_MIDDLE" and cursor_point is not None:
            movement = self._movement_from_anchor(cursor_point)
            self._candidate_max_movement = max(self._candidate_max_movement, movement)
            if self._candidate_max_movement > self.cfg.release_tolerance:
                self._state = SecondaryInteractionState.NEUTRAL
                self._candidate_anchor = None
                self._candidate_max_movement = 0.0
                return SecondaryInteractionOut(
                    state=self._state,
                    intent=no_action(reason="secondary_candidate_cancelled_movement", gesture_label="PINCH_MIDDLE"),
                    owns_state=False,
                    movement=0.0,
                )
            return self._candidate_out(reason="secondary_candidate_tracking")

        if cursor_point is None:
            self._state = SecondaryInteractionState.HAND_LOST_SAFE
            self._loss_started_s = now
            return SecondaryInteractionOut(
                state=self._state,
                intent=no_action(reason="secondary_hand_loss_wait", gesture_label="PINCH_MIDDLE"),
                owns_state=True,
                movement=0.0,
            )

        valid_click = self._candidate_max_movement <= self.cfg.release_tolerance
        self._state = SecondaryInteractionState.NEUTRAL
        self._candidate_anchor = None
        self._candidate_max_movement = 0.0
        return SecondaryInteractionOut(
            state=self._state,
            intent=dry_run_action(
                "SECONDARY_RIGHT_CLICK",
                gesture_label="PINCH_MIDDLE",
                reason="secondary_release_valid",
            ) if valid_click else no_action(
                reason="secondary_candidate_cancelled_release",
                gesture_label="PINCH_MIDDLE",
            ),
            owns_state=False,
            movement=0.0,
        )

    def _update_hand_loss(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float,
    ) -> SecondaryInteractionOut:
        if gesture_label == "PINCH_MIDDLE" and cursor_point is not None:
            self._state = SecondaryInteractionState.SECONDARY_PINCH_CANDIDATE
            self._loss_started_s = None
            self._candidate_anchor = cursor_point
            self._candidate_max_movement = 0.0
            return self._candidate_out(reason="secondary_candidate_resumed_after_hand_loss")

        if self._loss_started_s is not None and (now - self._loss_started_s) <= self.cfg.hand_loss_grace_s:
            return SecondaryInteractionOut(
                state=self._state,
                intent=no_action(reason="secondary_hand_loss_wait", gesture_label="PINCH_MIDDLE"),
                owns_state=True,
                movement=0.0,
            )

        self._state = SecondaryInteractionState.NEUTRAL
        self._candidate_anchor = None
        self._candidate_max_movement = 0.0
        self._loss_started_s = None
        return SecondaryInteractionOut(
            state=self._state,
            intent=no_action(reason="secondary_candidate_cancelled_hand_loss", gesture_label="PINCH_MIDDLE"),
            owns_state=False,
            movement=0.0,
        )

    def _candidate_out(self, *, reason: str) -> SecondaryInteractionOut:
        return SecondaryInteractionOut(
            state=self._state,
            intent=no_action(reason=reason, gesture_label="PINCH_MIDDLE"),
            owns_state=True,
            movement=self._candidate_max_movement,
        )

    def _movement_from_anchor(self, cursor_point: CursorPoint) -> float:
        if self._candidate_anchor is None:
            return 0.0
        return cursor_distance(self._candidate_anchor, cursor_point)
