from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import TYPE_CHECKING

from app.config import CONFIG, PresentationToolConfig
from app.control.actions import ActionIntent, dry_run_action, no_action
from app.control.cursor_space import CursorPoint
from app.control.pointer_filters import AdaptiveEmaFilter
from app.control.presentation_panel import (
    default_draw_color_key,
    default_draw_pen_key,
    default_draw_size_key,
    panel_contains_point,
    panel_item_layouts,
    resolve_panel_state,
    PresentationPanelState,
)
from app.modes.presentation_erase import PresentationEraseHandler

if TYPE_CHECKING:
    from app.gestures.suite import GestureSuiteOut


class PresentationToolState(str, Enum):
    NONE = "NONE"
    LASER = "LASER"
    DRAW_IDLE = "DRAW_IDLE"
    DRAW_STROKING = "DRAW_STROKING"


@dataclass(frozen=True)
class PresentationToolOut:
    state: PresentationToolState
    intent: ActionIntent
    pointer_point: CursorPoint | None
    owns_presentation: bool
    stroke_active: bool
    stroke_capturing: bool
    selected_color_key: str
    selected_pen_key: str
    selected_size_key: str
    panel_state: PresentationPanelState
    reason: str

    def status_text(self) -> str:
        selection = f"[{self.selected_color_key}/{self.selected_pen_key}/{self.selected_size_key}]"
        if self.intent.action_name == "NO_ACTION":
            return f"TOOL:{self.state.value}{selection}:{self.reason}"
        return f"TOOL:{self.state.value}{selection}:{self.intent.action_name}:{self.reason}"


class PresentationToolController:
    """
    Presentation-only dry-run controller for tool selection.
    It is intentionally separate from playback routing so laser/draw state can
    evolve without changing slideshow navigation semantics.
    """

    def __init__(
        self,
        *,
        cfg: PresentationToolConfig | None = None,
        clock=None,
    ):
        self.cfg = cfg or CONFIG.presentation_tools
        self._clock = clock or time.monotonic
        self._erase = PresentationEraseHandler(cfg=self.cfg, clock=self._clock)
        self.reset()

    @property
    def state(self) -> PresentationToolState:
        return self._state

    def reset(self) -> None:
        self._state = PresentationToolState.NONE
        self._draw_release_frames = 0
        self._selected_color_key = default_draw_color_key(self.cfg)
        self._selected_pen_key = default_draw_pen_key(self.cfg)
        self._selected_size_key = default_draw_size_key(self.cfg)
        self._panel_pointer_filter = AdaptiveEmaFilter(
            history_size=max(2, int(getattr(self.cfg, "panel_pointer_history_size", 6) or 6)),
            alpha_min=float(self.cfg.panel_pointer_slow_alpha),
            alpha_max=float(self.cfg.panel_pointer_smoothing_alpha),
        )
        self._panel_selection_consumed = False
        self._panel_open = False
        self._panel_sticky_until = 0.0
        self._panel_leave_frames = 0
        self._panel_layouts_by_id = {layout.option_id: layout for layout in panel_item_layouts(1.0, cfg=self.cfg)}
        self._panel_locked_option_id: str | None = None
        self._erase.reset()

    def update(
        self,
        *,
        suite_out: GestureSuiteOut | None,
        hand_present: bool,
        pointer_point: CursorPoint | None,
    ) -> PresentationToolOut:
        if not hand_present or suite_out is None:
            erase_event = self._erase.update(
                suite_out=suite_out,
                hand_present=hand_present,
                enabled=self._state in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING},
            )
            self._draw_release_frames = 0
            self._panel_pointer_filter.reset()
            self._panel_selection_consumed = False
            self._panel_open = False
            self._panel_sticky_until = 0.0
            self._panel_leave_frames = 0
            self._clear_panel_selection_lock()
            if erase_event.action_name is not None:
                self._state = PresentationToolState.DRAW_IDLE
                return self._build_out(
                    dry_run_action(
                        erase_event.action_name,
                        gesture_label="FIST",
                        reason=erase_event.reason,
                        mode="PRESENTATION",
                    ),
                    pointer_point=pointer_point,
                    reason=erase_event.reason,
                )
            if self._state == PresentationToolState.DRAW_STROKING:
                self._state = PresentationToolState.DRAW_IDLE
                return self._build_out(
                    dry_run_action(
                        "PRESENTATION_DRAW_STROKE_END",
                        gesture_label="PINCH_INDEX",
                        reason="draw_hand_lost",
                        mode="PRESENTATION",
                    ),
                    pointer_point=None,
                    reason="draw_hand_lost",
                )
            return self._build_out(no_action(reason="presentation_tool_idle", mode="PRESENTATION"), pointer_point=None, reason="idle")

        down = suite_out.down
        up = suite_out.up
        hold_frames = int(getattr(suite_out, "hold_frames", 0) or 0)
        eligible = suite_out.eligible
        chosen = getattr(suite_out, "chosen", eligible)
        pinch_visible = self._is_pinch_index_visible(suite_out)
        if eligible != "PINCH_INDEX" or not pinch_visible:
            self._panel_selection_consumed = False
            self._clear_panel_selection_lock()
        interaction_pointer = self._interaction_pointer(pointer_point, pinch_visible=pinch_visible)
        self._update_panel_visibility(interaction_pointer)
        panel_state = self._panel_state(interaction_pointer)

        if down == "L" and hold_frames >= max(1, int(self.cfg.laser_toggle_confirm_frames)):
            self._draw_release_frames = 0
            self._panel_selection_consumed = False
            self._panel_open = False
            self._panel_sticky_until = 0.0
            self._panel_leave_frames = 0
            if self._state == PresentationToolState.LASER:
                self._state = PresentationToolState.NONE
                return self._build_out(
                    dry_run_action(
                        "PRESENTATION_LASER_OFF",
                        gesture_label="L",
                        reason="laser_toggled_off",
                        mode="PRESENTATION",
                    ),
                    pointer_point=interaction_pointer,
                    reason="laser_toggled_off",
                )
            self._state = PresentationToolState.LASER
            return self._build_out(
                    dry_run_action(
                        "PRESENTATION_LASER_ON",
                        gesture_label="L",
                        reason="laser_toggled_on",
                        mode="PRESENTATION",
                    ),
                    pointer_point=interaction_pointer,
                    reason="laser_toggled_on",
                )

        if down == "BRAVO" and hold_frames >= max(1, int(self.cfg.draw_toggle_confirm_frames)):
            self._draw_release_frames = 0
            self._panel_selection_consumed = False
            self._panel_open = False
            self._panel_sticky_until = 0.0
            self._panel_leave_frames = 0
            self._erase.reset()
            if self._state in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING}:
                self._state = PresentationToolState.NONE
                return self._build_out(
                    dry_run_action(
                        "PRESENTATION_DRAW_OFF",
                        gesture_label="BRAVO",
                        reason="draw_toggled_off",
                        mode="PRESENTATION",
                    ),
                    pointer_point=interaction_pointer,
                    reason="draw_toggled_off",
                )
            self._state = PresentationToolState.DRAW_IDLE
            return self._build_out(
                    dry_run_action(
                        "PRESENTATION_DRAW_ON",
                        gesture_label="BRAVO",
                        reason="draw_toggled_on",
                        mode="PRESENTATION",
                    ),
                    pointer_point=interaction_pointer,
                    reason="draw_toggled_on",
                )

        if self._state in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING}:
            if (
                self._state == PresentationToolState.DRAW_IDLE
                and down == "PEACE_SIGN"
                and hold_frames >= max(1, int(getattr(self.cfg, "panel_toggle_confirm_frames", 2)))
            ):
                self._panel_open = not self._panel_open
                self._panel_sticky_until = (
                    self._clock() + (max(0, int(getattr(self.cfg, "panel_min_open_ms", 0) or 0)) / 1000.0)
                    if self._panel_open
                    else 0.0
                )
                self._panel_leave_frames = 0
                self._panel_selection_consumed = False
                self._clear_panel_selection_lock()
                return self._build_out(
                    no_action(
                        reason="panel_opened" if self._panel_open else "panel_closed",
                        gesture_label="PEACE_SIGN",
                        mode="PRESENTATION",
                    ),
                    pointer_point=interaction_pointer,
                    reason="panel_opened" if self._panel_open else "panel_closed",
                )
            erase_event = self._erase.update(
                suite_out=suite_out,
                hand_present=hand_present,
                enabled=True,
            )
            if erase_event.action_name is not None:
                self._draw_release_frames = 0
                self._panel_selection_consumed = False
                self._state = PresentationToolState.DRAW_IDLE
                return self._build_out(
                    dry_run_action(
                        erase_event.action_name,
                        gesture_label="FIST",
                        reason=erase_event.reason,
                        mode="PRESENTATION",
                    ),
                    pointer_point=interaction_pointer,
                    reason=erase_event.reason,
                )
            if erase_event.fist_pressed:
                self._draw_release_frames = 0
                if self._state == PresentationToolState.DRAW_STROKING:
                    self._panel_selection_consumed = False
                    self._state = PresentationToolState.DRAW_IDLE
                    return self._build_out(
                        dry_run_action(
                            "PRESENTATION_DRAW_STROKE_END",
                            gesture_label="PINCH_INDEX",
                            reason="draw_stroke_interrupted",
                            mode="PRESENTATION",
                        ),
                        pointer_point=interaction_pointer,
                        reason="draw_stroke_interrupted",
                    )
                return self._build_out(
                    no_action(reason="draw_erase_pending", gesture_label="FIST", mode="PRESENTATION"),
                    pointer_point=interaction_pointer,
                    reason="draw_erase_pending",
                )
            if self._state == PresentationToolState.DRAW_IDLE and eligible == "PINCH_INDEX" and pinch_visible:
                if self._panel_locked_option_id is None and panel_state.hovered_option_id is not None and not self._panel_selection_consumed:
                    self._panel_locked_option_id = panel_state.hovered_option_id
                select_confirm_frames = max(1, int(self.cfg.panel_select_confirm_frames))
                if (
                    panel_state.expanded
                    and panel_state.hovered_kind == "color"
                    and panel_state.hovered_key is not None
                    and not self._panel_selection_consumed
                    and hold_frames >= select_confirm_frames
                ):
                    self._panel_selection_consumed = True
                    self._selected_color_key = panel_state.hovered_key
                    return self._build_out(
                        dry_run_action(
                            "PRESENTATION_DRAW_COLOR_SET",
                            gesture_label="PINCH_INDEX",
                            reason=f"draw_color_selected:{self._selected_color_key}",
                            mode="PRESENTATION",
                        ),
                        pointer_point=interaction_pointer,
                        reason=f"draw_color_selected:{self._selected_color_key}",
                    )
                if (
                    panel_state.expanded
                    and panel_state.hovered_kind == "pen"
                    and panel_state.hovered_key is not None
                    and not self._panel_selection_consumed
                    and hold_frames >= select_confirm_frames
                ):
                    self._panel_selection_consumed = True
                    self._selected_pen_key = panel_state.hovered_key
                    return self._build_out(
                        dry_run_action(
                            "PRESENTATION_DRAW_PEN_SET",
                            gesture_label="PINCH_INDEX",
                            reason=f"draw_pen_selected:{self._selected_pen_key}",
                            mode="PRESENTATION",
                        ),
                        pointer_point=interaction_pointer,
                        reason=f"draw_pen_selected:{self._selected_pen_key}",
                    )
                if (
                    panel_state.expanded
                    and panel_state.hovered_kind == "size"
                    and panel_state.hovered_key is not None
                    and not self._panel_selection_consumed
                    and hold_frames >= select_confirm_frames
                ):
                    self._panel_selection_consumed = True
                    self._selected_size_key = panel_state.hovered_key
                    return self._build_out(
                        dry_run_action(
                            "PRESENTATION_DRAW_SIZE_SET",
                            gesture_label="PINCH_INDEX",
                            reason=f"draw_size_selected:{self._selected_size_key}",
                            mode="PRESENTATION",
                        ),
                        pointer_point=interaction_pointer,
                        reason=f"draw_size_selected:{self._selected_size_key}",
                    )
                if panel_state.blocked_by_panel:
                    reason = (
                        "panel_selection_pending"
                        if panel_state.hovered_option_id is not None and hold_frames < select_confirm_frames
                        else "panel_engaged"
                    )
                    return self._build_out(
                        no_action(reason=reason, gesture_label="PINCH_INDEX", mode="PRESENTATION"),
                        pointer_point=interaction_pointer,
                        reason=reason,
                    )
            if (
                self._state == PresentationToolState.DRAW_IDLE
                and down == "PINCH_INDEX"
                and pinch_visible
                and hold_frames >= max(1, int(self.cfg.draw_activation_confirm_frames))
            ):
                self._draw_release_frames = 0
                self._panel_selection_consumed = False
                self._panel_open = False
                self._panel_sticky_until = 0.0
                self._panel_leave_frames = 0
                self._clear_panel_selection_lock()
                self._state = PresentationToolState.DRAW_STROKING
                return self._build_out(
                    dry_run_action(
                        "PRESENTATION_DRAW_STROKE_START",
                        gesture_label="PINCH_INDEX",
                        reason="draw_stroke_started",
                        mode="PRESENTATION",
                    ),
                    pointer_point=interaction_pointer,
                    reason="draw_stroke_started",
                )

            if self._state == PresentationToolState.DRAW_STROKING:
                if up == "PINCH_INDEX":
                    self._draw_release_frames = 0
                    self._panel_selection_consumed = False
                    self._state = PresentationToolState.DRAW_IDLE
                    return self._build_out(
                        dry_run_action(
                            "PRESENTATION_DRAW_STROKE_END",
                            gesture_label="PINCH_INDEX",
                            reason="draw_stroke_released",
                            mode="PRESENTATION",
                        ),
                        pointer_point=interaction_pointer,
                        reason="draw_stroke_released",
                    )
                if eligible != "PINCH_INDEX":
                    self._draw_release_frames += 1
                    if self._draw_release_frames < max(1, int(self.cfg.draw_release_grace_frames)):
                        return self._build_out(
                            no_action(reason="draw_release_grace", gesture_label="PINCH_INDEX", mode="PRESENTATION"),
                            pointer_point=interaction_pointer,
                            reason="draw_release_grace",
                            stroke_capturing=False,
                        )
                    self._draw_release_frames = 0
                    self._panel_selection_consumed = False
                    self._state = PresentationToolState.DRAW_IDLE
                    return self._build_out(
                        dry_run_action(
                            "PRESENTATION_DRAW_STROKE_END",
                            gesture_label="PINCH_INDEX",
                            reason="draw_stroke_interrupted",
                            mode="PRESENTATION",
                        ),
                        pointer_point=interaction_pointer,
                        reason="draw_stroke_interrupted",
                    )
                if not pinch_visible or chosen != "PINCH_INDEX":
                    self._draw_release_frames = 0
                    return self._build_out(
                        no_action(reason="draw_capture_pending_release", gesture_label="PINCH_INDEX", mode="PRESENTATION"),
                        pointer_point=interaction_pointer,
                        reason="draw_capture_pending_release",
                        stroke_capturing=False,
                    )
                self._draw_release_frames = 0
                return self._build_out(
                    no_action(reason="draw_stroke_held", gesture_label="PINCH_INDEX", mode="PRESENTATION"),
                    pointer_point=interaction_pointer,
                    reason="draw_stroke_held",
                )

            self._draw_release_frames = 0
            return self._build_out(
                no_action(
                    reason="panel_open" if self._panel_open else "draw_ready",
                    gesture_label="PEACE_SIGN" if self._panel_open else "BRAVO",
                    mode="PRESENTATION",
                ),
                pointer_point=interaction_pointer,
                reason="panel_open" if self._panel_open else "draw_ready",
            )

        if self._state == PresentationToolState.LASER:
            self._draw_release_frames = 0
            self._panel_selection_consumed = False
            self._panel_open = False
            self._panel_sticky_until = 0.0
            self._panel_leave_frames = 0
            self._erase.reset()
            return self._build_out(
                no_action(reason="laser_ready", gesture_label="L", mode="PRESENTATION"),
                pointer_point=interaction_pointer,
                reason="laser_ready",
            )

        self._draw_release_frames = 0
        self._panel_selection_consumed = False
        self._erase.reset()
        return self._build_out(
            no_action(reason="presentation_tool_idle", mode="PRESENTATION"),
            pointer_point=interaction_pointer,
            reason="idle",
        )

    def _interaction_pointer(self, pointer_point: CursorPoint | None, *, pinch_visible: bool = False) -> CursorPoint | None:
        if self._state != PresentationToolState.DRAW_IDLE:
            self._panel_pointer_filter.reset()
            return pointer_point
        if pointer_point is None:
            self._panel_pointer_filter.reset()
            return None
        if self._panel_open and pinch_visible:
            locked_layout = self._locked_panel_layout()
            if locked_layout is not None:
                locked_point = locked_layout.center
                self._panel_pointer_filter.update(
                    locked_point,
                    timestamp=self._clock(),
                    alpha_min=1.0,
                    alpha_max=1.0,
                )
                return locked_point
        alpha = max(0.0, min(1.0, float(self.cfg.panel_pointer_smoothing_alpha)))
        working_point = pointer_point
        previous = self._panel_pointer_filter.current
        slow_padding = max(0.0, float(getattr(self.cfg, "panel_slow_padding", 0.018)))
        if self._panel_open and previous is not None and panel_contains_point(pointer_point, padding=slow_padding, cfg=self.cfg):
            slow_scale = max(0.0, min(1.0, float(getattr(self.cfg, "panel_pointer_slow_scale", 0.48))))
            working_point = CursorPoint(
                x=previous.x + ((pointer_point.x - previous.x) * slow_scale),
                y=previous.y + ((pointer_point.y - previous.y) * slow_scale),
            )
            alpha = min(alpha, max(0.0, min(1.0, float(getattr(self.cfg, "panel_pointer_slow_alpha", 0.14)))))
        return self._panel_pointer_filter.update(
            working_point,
            timestamp=self._clock(),
            alpha_min=alpha,
            alpha_max=alpha,
        )

    def _update_panel_visibility(self, pointer_point: CursorPoint | None) -> None:
        if self._state != PresentationToolState.DRAW_IDLE:
            self._panel_open = False
            self._panel_sticky_until = 0.0
            self._panel_leave_frames = 0
            self._clear_panel_selection_lock()
            return
        if not self._panel_open:
            self._panel_sticky_until = 0.0
            self._panel_leave_frames = 0
            self._clear_panel_selection_lock()
            return
        if self._clock() < self._panel_sticky_until:
            self._panel_leave_frames = 0
            return
        leave_padding = max(0.0, float(getattr(self.cfg, "panel_leave_padding", 0.022)))
        if panel_contains_point(pointer_point, padding=leave_padding, cfg=self.cfg):
            self._panel_leave_frames = 0
            return
        self._panel_leave_frames += 1
        if self._panel_leave_frames >= max(1, int(getattr(self.cfg, "panel_leave_grace_frames", 6))):
            self._panel_open = False
            self._panel_sticky_until = 0.0
            self._panel_leave_frames = 0
            self._clear_panel_selection_lock()

    @staticmethod
    def _is_pinch_index_visible(suite_out: GestureSuiteOut) -> bool:
        raw_candidates = getattr(suite_out, "raw_candidates", None)
        if raw_candidates is not None and "PINCH_INDEX" in raw_candidates:
            return True
        candidates = getattr(suite_out, "candidates", None)
        if candidates is not None and "PINCH_INDEX" in candidates:
            return True
        return False

    def _panel_state(self, pointer_point: CursorPoint | None) -> PresentationPanelState:
        return resolve_panel_state(
            pointer_point,
            draw_mode_active=self._state in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING},
            stroke_active=self._state == PresentationToolState.DRAW_STROKING,
            selected_color_key=self._selected_color_key,
            selected_pen_key=self._selected_pen_key,
            selected_size_key=self._selected_size_key,
            panel_open=self._panel_open,
            cfg=self.cfg,
        )

    def _locked_panel_layout(self):
        option_id = self._panel_locked_option_id
        if option_id is None:
            return None
        return self._panel_layouts_by_id.get(option_id)

    def _clear_panel_selection_lock(self) -> None:
        self._panel_locked_option_id = None

    def _build_out(
        self,
        intent: ActionIntent,
        *,
        pointer_point: CursorPoint | None,
        reason: str,
        stroke_capturing: bool | None = None,
    ) -> PresentationToolOut:
        panel_state = self._panel_state(pointer_point)
        capture_active = self._state == PresentationToolState.DRAW_STROKING if stroke_capturing is None else bool(stroke_capturing)
        return PresentationToolOut(
            state=self._state,
            intent=intent,
            pointer_point=pointer_point if self._state != PresentationToolState.NONE else None,
            owns_presentation=self._state in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING},
            stroke_active=self._state == PresentationToolState.DRAW_STROKING,
            stroke_capturing=capture_active,
            selected_color_key=self._selected_color_key,
            selected_pen_key=self._selected_pen_key,
            selected_size_key=self._selected_size_key,
            panel_state=panel_state,
            reason=reason,
        )
