from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from app.config import CONFIG, PrimaryInteractionConfig
from app.control.actions import ActionIntent, dry_run_action, no_action
from app.control.cursor_space import CursorPoint, cursor_distance


class PrimaryInteractionState(str, Enum):
    NEUTRAL = "NEUTRAL"
    PRIMARY_PINCH_CANDIDATE = "PRIMARY_PINCH_CANDIDATE"
    DRAG_ACTIVE = "DRAG_ACTIVE"
    CLICK_PENDING = "CLICK_PENDING"
    HAND_LOST_SAFE = "HAND_LOST_SAFE"


@dataclass(frozen=True)
class PrimaryInteractionOut:
    state: PrimaryInteractionState
    intent: ActionIntent
    owns_state: bool
    movement: float
    cursor_point: CursorPoint | None


class PrimaryInteractionController:
    """
    Stateful dry-run implementation of the approved PINCH_INDEX interaction path.
    It depends only on gated gesture labels and abstract cursor-space movement.
    """

    def __init__(self, cfg: PrimaryInteractionConfig | None = None):
        self.cfg = cfg or CONFIG.primary_interaction
        self.reset()

    @property
    def state(self) -> PrimaryInteractionState:
        return self._state

    def reset(self) -> None:
        self._state = PrimaryInteractionState.NEUTRAL
        self._candidate_anchor: CursorPoint | None = None
        self._candidate_max_movement = 0.0
        self._pending_deadline_s: float | None = None
        self._loss_started_s: float | None = None
        self._loss_resume_state: PrimaryInteractionState | None = None

    def update(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float | None = None,
    ) -> PrimaryInteractionOut:
        now = time.monotonic() if now is None else float(now)

        timeout_out = self._resolve_pending_click_timeout(now)
        if timeout_out is not None:
            return timeout_out

        if self._state == PrimaryInteractionState.HAND_LOST_SAFE:
            return self._update_hand_loss(gesture_label=gesture_label, cursor_point=cursor_point, now=now)
        if self._state == PrimaryInteractionState.DRAG_ACTIVE:
            return self._update_drag(gesture_label=gesture_label, cursor_point=cursor_point, now=now)
        if self._state == PrimaryInteractionState.PRIMARY_PINCH_CANDIDATE:
            return self._update_candidate(gesture_label=gesture_label, cursor_point=cursor_point, now=now)
        if self._state == PrimaryInteractionState.CLICK_PENDING:
            return self._update_click_pending(gesture_label=gesture_label, cursor_point=cursor_point)

        return self._update_neutral(gesture_label=gesture_label, cursor_point=cursor_point)

    def _resolve_pending_click_timeout(self, now: float) -> PrimaryInteractionOut | None:
        if self._state != PrimaryInteractionState.CLICK_PENDING:
            return None
        if self._pending_deadline_s is None or now < self._pending_deadline_s:
            return None

        self._pending_deadline_s = None
        self._state = PrimaryInteractionState.NEUTRAL
        return PrimaryInteractionOut(
            state=self._state,
            intent=dry_run_action(
                "PRIMARY_CLICK",
                gesture_label="PINCH_INDEX",
                reason="pending_window_expired",
            ),
            owns_state=False,
            movement=0.0,
            cursor_point=None,
        )

    def _update_neutral(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
    ) -> PrimaryInteractionOut:
        if gesture_label == "PINCH_INDEX" and cursor_point is not None:
            self._state = PrimaryInteractionState.PRIMARY_PINCH_CANDIDATE
            self._candidate_anchor = cursor_point
            self._candidate_max_movement = 0.0
            return self._candidate_out(reason="candidate_started")

        return PrimaryInteractionOut(
            state=self._state,
            intent=no_action(reason="primary_idle", gesture_label=gesture_label),
            owns_state=False,
            movement=0.0,
            cursor_point=cursor_point,
        )

    def _update_candidate(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float,
    ) -> PrimaryInteractionOut:
        if gesture_label == "PINCH_INDEX" and cursor_point is not None:
            movement = self._movement_from_anchor(cursor_point)
            self._candidate_max_movement = max(self._candidate_max_movement, movement)
            if self._candidate_max_movement >= self.cfg.drag_start_distance:
                self._pending_deadline_s = None
                self._state = PrimaryInteractionState.DRAG_ACTIVE
                self._candidate_anchor = cursor_point
                return PrimaryInteractionOut(
                    state=self._state,
                    intent=dry_run_action(
                        "PRIMARY_DRAG_START",
                        gesture_label="PINCH_INDEX",
                        reason="drag_threshold_reached",
                    ),
                    owns_state=True,
                    movement=self._candidate_max_movement,
                    cursor_point=cursor_point,
                )
            return self._candidate_out(reason="candidate_tracking")

        if cursor_point is None:
            return self._enter_hand_loss(now=now, resume_state=PrimaryInteractionState.PRIMARY_PINCH_CANDIDATE)

        if self._candidate_max_movement <= self.cfg.click_release_tolerance:
            if self._pending_deadline_s is not None:
                within_window = now <= self._pending_deadline_s
                self._pending_deadline_s = None
                self._state = PrimaryInteractionState.NEUTRAL
                self._candidate_anchor = None
                self._candidate_max_movement = 0.0
                return PrimaryInteractionOut(
                    state=self._state,
                    intent=dry_run_action(
                        "PRIMARY_DOUBLE_CLICK" if within_window else "PRIMARY_CLICK",
                        gesture_label="PINCH_INDEX",
                        reason="double_click_resolved" if within_window else "double_click_window_missed",
                    ),
                    owns_state=False,
                    movement=0.0,
                    cursor_point=cursor_point,
                )

            self._state = PrimaryInteractionState.CLICK_PENDING
            self._pending_deadline_s = now + self.cfg.double_click_window_s
            self._candidate_anchor = None
            self._candidate_max_movement = 0.0
            return PrimaryInteractionOut(
                state=self._state,
                intent=no_action(reason="click_pending", gesture_label="PINCH_INDEX"),
                owns_state=True,
                movement=0.0,
                cursor_point=cursor_point,
            )

        self._state = PrimaryInteractionState.NEUTRAL
        self._pending_deadline_s = None
        self._candidate_anchor = None
        self._candidate_max_movement = 0.0
        return PrimaryInteractionOut(
            state=self._state,
            intent=no_action(reason="primary_candidate_cancelled", gesture_label="PINCH_INDEX"),
            owns_state=False,
            movement=0.0,
            cursor_point=cursor_point,
        )

    def _update_drag(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float,
    ) -> PrimaryInteractionOut:
        if gesture_label == "PINCH_INDEX" and cursor_point is not None:
            movement = self._movement_from_anchor(cursor_point)
            self._candidate_anchor = cursor_point
            return PrimaryInteractionOut(
                state=self._state,
                intent=dry_run_action(
                    "PRIMARY_DRAG_HOLD",
                    gesture_label="PINCH_INDEX",
                    reason="drag_active",
                ),
                owns_state=True,
                movement=movement,
                cursor_point=cursor_point,
            )

        if cursor_point is None:
            return self._enter_hand_loss(now=now, resume_state=PrimaryInteractionState.DRAG_ACTIVE)

        self._state = PrimaryInteractionState.NEUTRAL
        self._candidate_anchor = None
        self._candidate_max_movement = 0.0
        return PrimaryInteractionOut(
            state=self._state,
            intent=dry_run_action(
                "PRIMARY_DRAG_END",
                gesture_label="PINCH_INDEX",
                reason="drag_release",
            ),
            owns_state=False,
            movement=0.0,
            cursor_point=cursor_point,
        )

    def _update_click_pending(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
    ) -> PrimaryInteractionOut:
        if gesture_label == "PINCH_INDEX" and cursor_point is not None:
            self._state = PrimaryInteractionState.PRIMARY_PINCH_CANDIDATE
            self._candidate_anchor = cursor_point
            self._candidate_max_movement = 0.0
            return self._candidate_out(reason="second_click_candidate_started")

        return PrimaryInteractionOut(
            state=self._state,
            intent=no_action(reason="click_pending", gesture_label="PINCH_INDEX"),
            owns_state=True,
            movement=0.0,
            cursor_point=cursor_point,
        )

    def _update_hand_loss(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float,
    ) -> PrimaryInteractionOut:
        if self._loss_started_s is None or self._loss_resume_state is None:
            self.reset()
            return PrimaryInteractionOut(
                state=self._state,
                intent=no_action(reason="primary_reset_after_invalid_loss_state"),
                owns_state=False,
                movement=0.0,
                cursor_point=cursor_point,
            )

        if cursor_point is not None and gesture_label == "PINCH_INDEX":
            self._state = self._loss_resume_state
            self._loss_started_s = None
            self._loss_resume_state = None
            self._candidate_anchor = cursor_point
            self._candidate_max_movement = 0.0
            reason = "candidate_resumed_after_hand_loss"
            if self._state == PrimaryInteractionState.DRAG_ACTIVE:
                reason = "drag_resumed_after_hand_loss"
            return PrimaryInteractionOut(
                state=self._state,
                intent=no_action(reason=reason, gesture_label="PINCH_INDEX"),
                owns_state=True,
                movement=0.0,
                cursor_point=cursor_point,
            )

        if (now - self._loss_started_s) <= self.cfg.hand_loss_grace_s:
            return PrimaryInteractionOut(
                state=self._state,
                intent=no_action(reason="primary_hand_loss_wait"),
                owns_state=True,
                movement=0.0,
                cursor_point=cursor_point,
            )

        cancelled_drag = self._loss_resume_state == PrimaryInteractionState.DRAG_ACTIVE
        self._state = PrimaryInteractionState.NEUTRAL
        self._candidate_anchor = None
        self._candidate_max_movement = 0.0
        self._loss_started_s = None
        self._loss_resume_state = None
        return PrimaryInteractionOut(
            state=self._state,
            intent=no_action(
                reason="drag_cancelled_hand_loss" if cancelled_drag else "primary_candidate_cancelled_hand_loss",
                gesture_label="PINCH_INDEX",
            ),
            owns_state=False,
            movement=0.0,
            cursor_point=cursor_point,
        )

    def _enter_hand_loss(
        self,
        *,
        now: float,
        resume_state: PrimaryInteractionState,
    ) -> PrimaryInteractionOut:
        if self._pending_deadline_s is not None:
            self._pending_deadline_s = None
        self._state = PrimaryInteractionState.HAND_LOST_SAFE
        self._loss_started_s = now
        self._loss_resume_state = resume_state
        return PrimaryInteractionOut(
            state=self._state,
            intent=no_action(reason=f"primary_hand_loss:{resume_state.value.lower()}", gesture_label="PINCH_INDEX"),
            owns_state=True,
            movement=0.0,
            cursor_point=None,
        )

    def _candidate_out(self, *, reason: str) -> PrimaryInteractionOut:
        return PrimaryInteractionOut(
            state=self._state,
            intent=no_action(reason=reason, gesture_label="PINCH_INDEX"),
            owns_state=True,
            movement=self._candidate_max_movement,
            cursor_point=self._candidate_anchor,
        )

    def _movement_from_anchor(self, cursor_point: CursorPoint) -> float:
        if self._candidate_anchor is None:
            return 0.0
        return cursor_distance(self._candidate_anchor, cursor_point)
