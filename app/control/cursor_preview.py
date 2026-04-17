from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum

from app.config import CONFIG, CursorPreviewConfig
from app.control.actions import ActionIntent, dry_run_action, no_action
from app.control.cursor_policy import CursorPolicy, CursorPolicyDecision
from app.control.cursor_space import CursorPoint


class CursorPreviewState(str, Enum):
    NEUTRAL = "NEUTRAL"
    CURSOR_ACTIVE = "CURSOR_ACTIVE"


@dataclass(frozen=True)
class CursorPreviewOut:
    state: CursorPreviewState
    intent: ActionIntent
    owns_state: bool
    preview_point: CursorPoint | None
    movement: float
    policy: CursorPolicyDecision


class CursorPreviewController:
    """
    Non-OS cursor movement dry run on the shared abstract cursor-space seam.
    Preview state is independent from the final OS emission layer.
    """

    def __init__(
        self,
        *,
        policy: CursorPolicy | None = None,
        cfg: CursorPreviewConfig | None = None,
    ):
        self.policy = policy or CursorPolicy()
        self.cfg = cfg or CONFIG.cursor_preview
        self.reset()

    @property
    def state(self) -> CursorPreviewState:
        return self._state

    def reset(self) -> None:
        self._state = CursorPreviewState.NEUTRAL
        self._preview_point: CursorPoint | None = None
        self._anchor_point: CursorPoint | None = None

    def seed_preview_point(self, preview_point: CursorPoint | None) -> None:
        if preview_point is None:
            return
        if self._state != CursorPreviewState.NEUTRAL:
            return
        if self._preview_point is not None:
            return
        self._preview_point = preview_point

    def update(
        self,
        *,
        gesture_label: str | None,
        cursor_point: CursorPoint | None,
        higher_priority_owned: bool,
        now: float | None = None,
    ) -> CursorPreviewOut:
        _ = time.monotonic() if now is None else float(now)
        policy = self.policy.evaluate(gesture_label)

        if higher_priority_owned:
            self._state = CursorPreviewState.NEUTRAL
            self._anchor_point = None
            return CursorPreviewOut(
                state=self._state,
                intent=no_action(reason="cursor_suppressed", gesture_label=gesture_label),
                owns_state=False,
                preview_point=self._preview_point,
                movement=0.0,
                policy=policy,
            )

        if not policy.eligible or cursor_point is None:
            self._state = CursorPreviewState.NEUTRAL
            self._anchor_point = None
            return CursorPreviewOut(
                state=self._state,
                intent=no_action(reason=policy.reason, gesture_label=gesture_label),
                owns_state=False,
                preview_point=self._preview_point,
                movement=0.0,
                policy=policy,
            )

        if self._preview_point is None:
            self._preview_point = cursor_point

        if self._anchor_point is None:
            self._anchor_point = cursor_point
            self._state = CursorPreviewState.CURSOR_ACTIVE
            return CursorPreviewOut(
                state=self._state,
                intent=dry_run_action(
                    "CURSOR_PREVIEW_READY",
                    gesture_label=policy.gesture_label,
                    reason="cursor_reanchored",
                ),
                owns_state=True,
                preview_point=self._preview_point,
                movement=0.0,
                policy=policy,
            )

        dx = cursor_point.x - self._anchor_point.x
        dy = cursor_point.y - self._anchor_point.y
        movement = math.hypot(dx, dy)
        self._anchor_point = cursor_point
        self._state = CursorPreviewState.CURSOR_ACTIVE

        if movement <= self.cfg.move_epsilon:
            return CursorPreviewOut(
                state=self._state,
                intent=dry_run_action(
                    "CURSOR_PREVIEW_HOLD",
                    gesture_label=policy.gesture_label,
                    reason="cursor_active_hold",
                ),
                owns_state=True,
                preview_point=self._preview_point,
                movement=0.0,
                policy=policy,
            )

        next_x = min(1.0, max(0.0, self._preview_point.x + (dx * self.cfg.gain)))
        next_y = min(1.0, max(0.0, self._preview_point.y + (dy * self.cfg.gain)))
        self._preview_point = CursorPoint(x=next_x, y=next_y)
        return CursorPreviewOut(
            state=self._state,
            intent=dry_run_action(
                "CURSOR_PREVIEW_MOVE",
                gesture_label=policy.gesture_label,
                reason="cursor_preview_move",
            ),
            owns_state=True,
            preview_point=self._preview_point,
            movement=movement,
            policy=policy,
        )
