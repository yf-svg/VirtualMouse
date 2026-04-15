from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum

from app.config import CONFIG, ScrollInteractionConfig
from app.control.actions import ActionIntent, dry_run_action, no_action
from app.control.cursor_space import CursorPoint


class ScrollState(str, Enum):
    NEUTRAL = "NEUTRAL"
    SCROLL_MODE_ACTIVE = "SCROLL_MODE_ACTIVE"
    HAND_LOST_SAFE = "HAND_LOST_SAFE"


class ScrollAxis(str, Enum):
    NONE = "NONE"
    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"


@dataclass(frozen=True)
class ScrollOut:
    state: ScrollState
    axis: ScrollAxis
    intent: ActionIntent
    owns_state: bool
    movement: float


class ScrollModeController:
    """
    Stateful dry-run scroll-mode controller.
    Scroll is mode-based, toggled by a dedicated gesture edge, and driven only by
    abstract cursor-space movement after activation.
    """

    def __init__(self, cfg: ScrollInteractionConfig | None = None):
        self.cfg = cfg or CONFIG.scroll_interaction
        self.reset()

    @property
    def state(self) -> ScrollState:
        return self._state

    @property
    def toggle_label(self) -> str:
        return self.cfg.toggle_gesture_label

    def reset(self) -> None:
        self._state = ScrollState.NEUTRAL
        self._axis = ScrollAxis.NONE
        self._toggle_armed = True
        self._motion_anchor: CursorPoint | None = None
        self._low_motion_started_s: float | None = None
        self._loss_started_s: float | None = None

    def update(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float | None = None,
    ) -> ScrollOut:
        now = time.monotonic() if now is None else float(now)
        toggle_label = self.toggle_label

        if gesture_label != toggle_label:
            self._toggle_armed = True

        if self._state != ScrollState.HAND_LOST_SAFE and gesture_label == toggle_label and self._toggle_armed:
            self._toggle_armed = False
            return self._toggle_mode(cursor_point=cursor_point)

        if self._state == ScrollState.HAND_LOST_SAFE:
            return self._update_hand_loss(gesture_label=gesture_label, cursor_point=cursor_point, now=now)
        if self._state == ScrollState.SCROLL_MODE_ACTIVE:
            return self._update_active(cursor_point=cursor_point, now=now)

        return ScrollOut(
            state=self._state,
            axis=self._axis,
            intent=no_action(reason="scroll_idle", gesture_label=gesture_label),
            owns_state=False,
            movement=0.0,
        )

    def _toggle_mode(self, *, cursor_point: CursorPoint | None) -> ScrollOut:
        toggle_label = self.toggle_label
        if self._state == ScrollState.SCROLL_MODE_ACTIVE:
            self._state = ScrollState.NEUTRAL
            self._axis = ScrollAxis.NONE
            self._motion_anchor = None
            self._low_motion_started_s = None
            self._loss_started_s = None
            return ScrollOut(
                state=self._state,
                axis=self._axis,
                intent=dry_run_action("SCROLL_MODE_EXIT", gesture_label=toggle_label, reason="scroll_mode_toggled_off"),
                owns_state=True,
                movement=0.0,
            )

        if cursor_point is None:
            return ScrollOut(
                state=self._state,
                axis=self._axis,
                intent=no_action(reason="scroll_toggle_ignored_no_cursor", gesture_label=toggle_label),
                owns_state=False,
                movement=0.0,
            )

        self._state = ScrollState.SCROLL_MODE_ACTIVE
        self._axis = ScrollAxis.NONE
        self._motion_anchor = cursor_point
        self._low_motion_started_s = None
        self._loss_started_s = None
        return ScrollOut(
            state=self._state,
            axis=self._axis,
            intent=dry_run_action("SCROLL_MODE_ENTER", gesture_label=toggle_label, reason="scroll_mode_toggled_on"),
            owns_state=True,
            movement=0.0,
        )

    def _update_active(self, *, cursor_point: CursorPoint | None, now: float) -> ScrollOut:
        toggle_label = self.toggle_label
        if cursor_point is None:
            self._state = ScrollState.HAND_LOST_SAFE
            self._loss_started_s = now
            return ScrollOut(
                state=self._state,
                axis=self._axis,
                intent=no_action(reason="scroll_hand_loss_wait", gesture_label=toggle_label),
                owns_state=True,
                movement=0.0,
            )

        if self._motion_anchor is None:
            self._motion_anchor = cursor_point

        dx = cursor_point.x - self._motion_anchor.x
        dy = cursor_point.y - self._motion_anchor.y

        if self._axis == ScrollAxis.NONE:
            motion_mag = math.hypot(dx, dy)
            if motion_mag < self.cfg.dead_zone:
                return ScrollOut(
                    state=self._state,
                    axis=self._axis,
                    intent=no_action(reason="scroll_dead_zone", gesture_label=toggle_label),
                    owns_state=True,
                    movement=0.0,
                )

            x_mag = abs(dx)
            y_mag = abs(dy)
            if (y_mag - x_mag) >= self.cfg.axis_dominance_margin:
                self._axis = ScrollAxis.VERTICAL
                self._motion_anchor = cursor_point
                self._low_motion_started_s = None
                return ScrollOut(
                    state=self._state,
                    axis=self._axis,
                    intent=no_action(reason="scroll_axis_locked_vertical", gesture_label=toggle_label),
                    owns_state=True,
                    movement=0.0,
                )
            if (x_mag - y_mag) >= self.cfg.axis_dominance_margin:
                self._axis = ScrollAxis.HORIZONTAL
                self._motion_anchor = cursor_point
                self._low_motion_started_s = None
                return ScrollOut(
                    state=self._state,
                    axis=self._axis,
                    intent=no_action(reason="scroll_axis_locked_horizontal", gesture_label=toggle_label),
                    owns_state=True,
                    movement=0.0,
                )

            return ScrollOut(
                state=self._state,
                axis=self._axis,
                intent=no_action(reason="scroll_axis_undetermined", gesture_label=toggle_label),
                owns_state=True,
                movement=0.0,
            )

        axis_delta = dy if self._axis == ScrollAxis.VERTICAL else dx
        if abs(axis_delta) <= self.cfg.pause_motion_epsilon:
            if self._low_motion_started_s is None:
                self._low_motion_started_s = now
            elif (now - self._low_motion_started_s) >= self.cfg.pause_reset_s:
                self._axis = ScrollAxis.NONE
                self._motion_anchor = cursor_point
                self._low_motion_started_s = None
                return ScrollOut(
                    state=self._state,
                    axis=self._axis,
                    intent=no_action(reason="scroll_axis_reset_pause", gesture_label=toggle_label),
                    owns_state=True,
                    movement=0.0,
                )

            return ScrollOut(
                state=self._state,
                axis=self._axis,
                intent=no_action(reason="scroll_low_motion_wait", gesture_label=toggle_label),
                owns_state=True,
                movement=0.0,
            )

        self._low_motion_started_s = None
        self._motion_anchor = cursor_point
        movement = axis_delta * self.cfg.scroll_gain
        return ScrollOut(
            state=self._state,
            axis=self._axis,
            intent=dry_run_action(
                "SCROLL_VERTICAL" if self._axis == ScrollAxis.VERTICAL else "SCROLL_HORIZONTAL",
                gesture_label=toggle_label,
                reason="scroll_move",
            ),
            owns_state=True,
            movement=movement,
        )

    def _update_hand_loss(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        now: float,
    ) -> ScrollOut:
        toggle_label = self.toggle_label
        if cursor_point is not None and self._loss_started_s is not None and (now - self._loss_started_s) <= self.cfg.hand_loss_grace_s:
            self._state = ScrollState.SCROLL_MODE_ACTIVE
            self._loss_started_s = None
            self._motion_anchor = cursor_point
            self._low_motion_started_s = None
            return ScrollOut(
                state=self._state,
                axis=self._axis,
                intent=no_action(reason="scroll_resumed_after_hand_loss", gesture_label=gesture_label),
                owns_state=True,
                movement=0.0,
            )

        if self._loss_started_s is not None and (now - self._loss_started_s) <= self.cfg.hand_loss_grace_s:
            return ScrollOut(
                state=self._state,
                axis=self._axis,
                intent=no_action(reason="scroll_hand_loss_wait", gesture_label=toggle_label),
                owns_state=True,
                movement=0.0,
            )

        self._state = ScrollState.NEUTRAL
        self._axis = ScrollAxis.NONE
        self._motion_anchor = None
        self._low_motion_started_s = None
        self._loss_started_s = None
        return ScrollOut(
            state=self._state,
            axis=self._axis,
            intent=dry_run_action("SCROLL_MODE_EXIT", gesture_label=toggle_label, reason="scroll_exit_hand_loss"),
            owns_state=True,
            movement=0.0,
        )
