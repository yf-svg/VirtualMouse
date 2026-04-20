from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.config import PresentationToolConfig
from app.control.cursor_space import CursorPoint
from app.control.presentation_panel import panel_item_layouts
from app.modes.presentation_tools import PresentationToolController, PresentationToolState

_UNSET = object()


class _FakeClock:
    def __init__(self):
        self.value = 0.0

    def now(self) -> float:
        return self.value

    def set(self, value: float) -> None:
        self.value = float(value)

    def advance(self, delta: float) -> None:
        self.value += float(delta)


@dataclass
class _SuiteOut:
    chosen: str | None
    stable: str | None
    eligible: str | None
    raw_candidates: set[str]
    candidates: set[str]
    reason: str
    down: str | None
    up: str | None
    source: str
    confidence: float | None
    rule_chosen: str | None
    ml_chosen: str | None
    ml_reason: str
    feature_reason: str
    hold_frames: int
    gate_reason: str


def _suite_out(
    *,
    eligible: str | None = None,
    chosen: str | None | object = _UNSET,
    stable: str | None | object = _UNSET,
    raw_candidates: set[str] | None | object = _UNSET,
    candidates: set[str] | None | object = _UNSET,
    down: str | None = None,
    up: str | None = None,
    hold_frames: int = 2,
) -> _SuiteOut:
    resolved_candidates = ({eligible} if eligible is not None else set()) if candidates is _UNSET else (set() if candidates is None else set(candidates))
    resolved_raw_candidates = set(resolved_candidates) if raw_candidates is _UNSET else (set() if raw_candidates is None else set(raw_candidates))
    return _SuiteOut(
        chosen=eligible if chosen is _UNSET else chosen,
        stable=eligible if stable is _UNSET else stable,
        eligible=eligible,
        raw_candidates=resolved_raw_candidates,
        candidates=resolved_candidates,
        reason="test",
        down=down,
        up=up,
        source="rules",
        confidence=None,
        rule_chosen=eligible,
        ml_chosen=None,
        ml_reason="none",
        feature_reason="ok",
        hold_frames=hold_frames,
        gate_reason="stable_match",
    )


class PresentationToolControllerTests(unittest.TestCase):
    def test_l_toggles_laser_on_and_off(self):
        controller = PresentationToolController()

        enabled = controller.update(
            suite_out=_suite_out(eligible="L", down="L"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        disabled = controller.update(
            suite_out=_suite_out(eligible="L", down="L"),
            hand_present=True,
            pointer_point=CursorPoint(0.4, 0.5),
        )

        self.assertEqual(enabled.state, PresentationToolState.LASER)
        self.assertEqual(enabled.intent.action_name, "PRESENTATION_LASER_ON")
        self.assertFalse(enabled.owns_presentation)
        self.assertEqual(disabled.state, PresentationToolState.NONE)
        self.assertEqual(disabled.intent.action_name, "PRESENTATION_LASER_OFF")

    def test_bravo_toggles_draw_on_and_off(self):
        controller = PresentationToolController()

        enabled = controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        disabled = controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.4, 0.5),
        )

        self.assertEqual(enabled.state, PresentationToolState.DRAW_IDLE)
        self.assertEqual(enabled.intent.action_name, "PRESENTATION_DRAW_ON")
        self.assertTrue(enabled.owns_presentation)
        self.assertEqual(disabled.state, PresentationToolState.NONE)
        self.assertEqual(disabled.intent.action_name, "PRESENTATION_DRAW_OFF")

    def test_pinch_index_only_strokes_inside_draw_mode(self):
        controller = PresentationToolController()

        ignored = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", down="PINCH_INDEX", hold_frames=2),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        started = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", down="PINCH_INDEX", hold_frames=2),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        held = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        ended = controller.update(
            suite_out=_suite_out(eligible="BRAVO", up="PINCH_INDEX"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        self.assertEqual(ignored.state, PresentationToolState.NONE)
        self.assertEqual(ignored.intent.action_name, "NO_ACTION")
        self.assertEqual(started.state, PresentationToolState.DRAW_STROKING)
        self.assertEqual(started.intent.action_name, "PRESENTATION_DRAW_STROKE_START")
        self.assertEqual(held.state, PresentationToolState.DRAW_STROKING)
        self.assertEqual(held.reason, "draw_stroke_held")
        self.assertEqual(ended.state, PresentationToolState.DRAW_IDLE)
        self.assertEqual(ended.intent.action_name, "PRESENTATION_DRAW_STROKE_END")

    def test_switching_tools_stays_mutually_exclusive(self):
        controller = PresentationToolController()

        controller.update(
            suite_out=_suite_out(eligible="L", down="L"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        switched = controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.4, 0.5),
        )

        self.assertEqual(switched.state, PresentationToolState.DRAW_IDLE)
        self.assertEqual(switched.intent.action_name, "PRESENTATION_DRAW_ON")

    def test_hand_loss_ends_active_stroke_but_not_selected_tool(self):
        controller = PresentationToolController()
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", down="PINCH_INDEX"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        lost = controller.update(
            suite_out=None,
            hand_present=False,
            pointer_point=None,
        )

        self.assertEqual(lost.state, PresentationToolState.DRAW_IDLE)
        self.assertEqual(lost.intent.action_name, "PRESENTATION_DRAW_STROKE_END")

    def test_draw_release_grace_absorbs_brief_pinch_jitter(self):
        controller = PresentationToolController(
            cfg=PresentationToolConfig(draw_release_grace_frames=2)
        )
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", down="PINCH_INDEX"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        grace = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.21, 0.31),
        )
        ended = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.22, 0.32),
        )

        self.assertEqual(grace.state, PresentationToolState.DRAW_STROKING)
        self.assertEqual(grace.reason, "draw_release_grace")
        self.assertTrue(grace.stroke_active)
        self.assertFalse(grace.stroke_capturing)
        self.assertEqual(ended.state, PresentationToolState.DRAW_IDLE)
        self.assertEqual(ended.intent.action_name, "PRESENTATION_DRAW_STROKE_END")

    def test_draw_stroke_stops_capture_when_chosen_label_drops_before_eligible_release(self):
        controller = PresentationToolController()
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", down="PINCH_INDEX"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        pending_release = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", chosen=None, stable="PINCH_INDEX"),
            hand_present=True,
            pointer_point=CursorPoint(0.28, 0.38),
        )

        self.assertEqual(pending_release.state, PresentationToolState.DRAW_STROKING)
        self.assertEqual(pending_release.intent.action_name, "NO_ACTION")
        self.assertEqual(pending_release.reason, "draw_capture_pending_release")
        self.assertTrue(pending_release.stroke_active)
        self.assertFalse(pending_release.stroke_capturing)

    def test_draw_stroke_stops_capture_when_pinch_is_no_longer_in_current_candidates(self):
        controller = PresentationToolController()
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )
        controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", down="PINCH_INDEX"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        pending_release = controller.update(
            suite_out=_suite_out(
                eligible="PINCH_INDEX",
                chosen="PINCH_INDEX",
                stable="PINCH_INDEX",
                raw_candidates=set(),
                candidates=set(),
            ),
            hand_present=True,
            pointer_point=CursorPoint(0.28, 0.38),
        )

        self.assertEqual(pending_release.state, PresentationToolState.DRAW_STROKING)
        self.assertEqual(pending_release.intent.action_name, "NO_ACTION")
        self.assertEqual(pending_release.reason, "draw_capture_pending_release")
        self.assertTrue(pending_release.stroke_active)
        self.assertFalse(pending_release.stroke_capturing)

    def test_peace_sign_opens_tray_and_pinch_selects_color_without_starting_stroke(self):
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                panel_pointer_smoothing_alpha=1.0,
                panel_pointer_slow_alpha=1.0,
                panel_pointer_slow_scale=1.0,
            )
        )
        color_point = next(item.center for item in panel_item_layouts(1.0) if item.option_id == "color:azure")
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.40, 0.50),
        )
        opened = controller.update(
            suite_out=_suite_out(eligible="PEACE_SIGN", down="PEACE_SIGN"),
            hand_present=True,
            pointer_point=color_point,
        )

        pending = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", down="PINCH_INDEX", hold_frames=1),
            hand_present=True,
            pointer_point=color_point,
        )
        selected = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", hold_frames=2),
            hand_present=True,
            pointer_point=color_point,
        )

        self.assertEqual(opened.intent.action_name, "NO_ACTION")
        self.assertEqual(opened.reason, "panel_opened")
        self.assertTrue(opened.panel_state.visible)
        self.assertEqual(pending.intent.action_name, "NO_ACTION")
        self.assertEqual(pending.reason, "panel_selection_pending")
        self.assertEqual(selected.state, PresentationToolState.DRAW_IDLE)
        self.assertEqual(selected.intent.action_name, "PRESENTATION_DRAW_COLOR_SET")
        self.assertEqual(selected.selected_color_key, "azure")
        self.assertFalse(selected.stroke_active)

    def test_peace_sign_opens_tray_and_pinch_selects_pen_without_starting_stroke(self):
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                panel_pointer_smoothing_alpha=1.0,
                panel_pointer_slow_alpha=1.0,
                panel_pointer_slow_scale=1.0,
            )
        )
        pen_point = next(item.center for item in panel_item_layouts(1.0) if item.option_id == "pen:quill")
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.40, 0.50),
        )
        controller.update(
            suite_out=_suite_out(eligible="PEACE_SIGN", down="PEACE_SIGN"),
            hand_present=True,
            pointer_point=pen_point,
        )

        pending = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", down="PINCH_INDEX", hold_frames=1),
            hand_present=True,
            pointer_point=pen_point,
        )
        selected = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", hold_frames=2),
            hand_present=True,
            pointer_point=pen_point,
        )

        self.assertEqual(pending.intent.action_name, "NO_ACTION")
        self.assertEqual(pending.reason, "panel_selection_pending")
        self.assertEqual(selected.state, PresentationToolState.DRAW_IDLE)
        self.assertEqual(selected.intent.action_name, "PRESENTATION_DRAW_PEN_SET")
        self.assertEqual(selected.selected_pen_key, "quill")
        self.assertFalse(selected.stroke_active)

    def test_peace_sign_opens_tray_and_pinch_selects_size_without_starting_stroke(self):
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                panel_pointer_smoothing_alpha=1.0,
                panel_pointer_slow_alpha=1.0,
                panel_pointer_slow_scale=1.0,
            )
        )
        size_point = next(item.center for item in panel_item_layouts(1.0) if item.option_id == "size:20")
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.40, 0.50),
        )
        controller.update(
            suite_out=_suite_out(eligible="PEACE_SIGN", down="PEACE_SIGN"),
            hand_present=True,
            pointer_point=size_point,
        )

        pending = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", down="PINCH_INDEX", hold_frames=1),
            hand_present=True,
            pointer_point=size_point,
        )
        selected = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", hold_frames=2),
            hand_present=True,
            pointer_point=size_point,
        )

        self.assertEqual(pending.intent.action_name, "NO_ACTION")
        self.assertEqual(pending.reason, "panel_selection_pending")
        self.assertEqual(selected.state, PresentationToolState.DRAW_IDLE)
        self.assertEqual(selected.intent.action_name, "PRESENTATION_DRAW_SIZE_SET")
        self.assertEqual(selected.selected_size_key, "20")
        self.assertFalse(selected.stroke_active)

    def test_panel_selection_locks_to_hovered_item_while_pinch_is_held(self):
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                panel_pointer_smoothing_alpha=1.0,
                panel_pointer_slow_alpha=1.0,
                panel_pointer_slow_scale=1.0,
            )
        )
        target_point = next(item.center for item in panel_item_layouts(1.0) if item.option_id == "size:20")
        drift_point = CursorPoint(0.52, 0.84)
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.40, 0.50),
        )
        controller.update(
            suite_out=_suite_out(eligible="PEACE_SIGN", down="PEACE_SIGN"),
            hand_present=True,
            pointer_point=target_point,
        )

        first_pinch = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", down="PINCH_INDEX", hold_frames=1),
            hand_present=True,
            pointer_point=target_point,
        )
        selected = controller.update(
            suite_out=_suite_out(eligible="PINCH_INDEX", hold_frames=2),
            hand_present=True,
            pointer_point=drift_point,
        )

        self.assertEqual(first_pinch.reason, "panel_selection_pending")
        self.assertEqual(first_pinch.panel_state.hovered_option_id, "size:20")
        self.assertEqual(selected.intent.action_name, "PRESENTATION_DRAW_SIZE_SET")
        self.assertEqual(selected.selected_size_key, "20")
        self.assertEqual(selected.pointer_point, target_point)

    def test_tray_stays_hidden_until_peace_sign_opens_it(self):
        controller = PresentationToolController()
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.40, 0.60),
        )

        idle = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.42, 0.62),
        )
        reveal = controller.update(
            suite_out=_suite_out(eligible="PEACE_SIGN", down="PEACE_SIGN"),
            hand_present=True,
            pointer_point=CursorPoint(0.88, 0.30),
        )

        self.assertFalse(idle.panel_state.visible)
        self.assertFalse(idle.panel_state.expanded)
        self.assertTrue(reveal.panel_state.visible)
        self.assertTrue(reveal.panel_state.expanded)

    def test_open_tray_closes_after_leave_grace(self):
        controller = PresentationToolController(
            cfg=PresentationToolConfig(panel_leave_grace_frames=2, panel_min_open_ms=0)
        )
        inside_point = next(item.center for item in panel_item_layouts(1.0) if item.option_id == "color:gold")
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.40, 0.60),
        )
        controller.update(
            suite_out=_suite_out(eligible="PEACE_SIGN", down="PEACE_SIGN"),
            hand_present=True,
            pointer_point=inside_point,
        )

        still_open = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.46, 0.78),
        )
        closed = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.46, 0.78),
        )

        self.assertTrue(still_open.panel_state.visible)
        self.assertFalse(still_open.panel_state.blocked_by_panel)
        self.assertFalse(closed.panel_state.visible)

    def test_open_tray_stays_visible_for_minimum_open_period_before_leave_close(self):
        clock = _FakeClock()
        controller = PresentationToolController(
            cfg=PresentationToolConfig(panel_leave_grace_frames=2, panel_min_open_ms=5000),
            clock=clock.now,
        )
        inside_point = next(item.center for item in panel_item_layouts(1.0) if item.option_id == "color:gold")
        outside_point = CursorPoint(0.46, 0.78)
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.40, 0.60),
        )
        clock.set(0.0)
        controller.update(
            suite_out=_suite_out(eligible="PEACE_SIGN", down="PEACE_SIGN"),
            hand_present=True,
            pointer_point=inside_point,
        )

        clock.advance(1.0)
        still_sticky_a = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=outside_point,
        )
        still_sticky_b = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=outside_point,
        )

        clock.advance(4.1)
        grace = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=outside_point,
        )
        closed = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=outside_point,
        )

        self.assertTrue(still_sticky_a.panel_state.visible)
        self.assertTrue(still_sticky_b.panel_state.visible)
        self.assertTrue(grace.panel_state.visible)
        self.assertFalse(closed.panel_state.visible)

    def test_draw_idle_pointer_is_smoothed_for_panel_selection(self):
        controller = PresentationToolController(
            cfg=PresentationToolConfig(panel_pointer_smoothing_alpha=0.5)
        )
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.20, 0.30),
        )

        first = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.20, 0.30),
        )
        second = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.60, 0.70),
        )

        self.assertIsNotNone(first.pointer_point)
        self.assertIsNotNone(second.pointer_point)
        self.assertAlmostEqual(first.pointer_point.x, 0.20, places=6)
        self.assertAlmostEqual(first.pointer_point.y, 0.30, places=6)
        self.assertAlmostEqual(second.pointer_point.x, 0.40, places=6)
        self.assertAlmostEqual(second.pointer_point.y, 0.50, places=6)

    def test_open_tray_slows_pointer_motion_inside_the_panel(self):
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                panel_pointer_smoothing_alpha=1.0,
                panel_pointer_slow_alpha=1.0,
                panel_pointer_slow_scale=0.5,
            )
        )
        first_point = next(item.center for item in panel_item_layouts(1.0) if item.option_id == "color:gold")
        second_point = next(item.center for item in panel_item_layouts(1.0) if item.option_id == "color:coral")
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.40, 0.60),
        )
        controller.update(
            suite_out=_suite_out(eligible="PEACE_SIGN", down="PEACE_SIGN"),
            hand_present=True,
            pointer_point=first_point,
        )

        first = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=first_point,
        )
        slowed = controller.update(
            suite_out=_suite_out(eligible="BRAVO"),
            hand_present=True,
            pointer_point=second_point,
        )

        self.assertIsNotNone(first.pointer_point)
        self.assertIsNotNone(slowed.pointer_point)
        self.assertAlmostEqual(slowed.pointer_point.x, first_point.x + ((second_point.x - first_point.x) * 0.5), places=6)
        self.assertAlmostEqual(slowed.pointer_point.y, first_point.y + ((second_point.y - first_point.y) * 0.5), places=6)

    def test_quick_fist_release_undoes_one_stroke(self):
        clock = _FakeClock()
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                draw_undo_min_ms=60,
                draw_undo_max_ms=650,
                draw_clear_hold_ms=800,
                draw_clear_confirm_frames=2,
            ),
            clock=clock.now,
        )
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        clock.set(0.0)
        pressed = controller.update(
            suite_out=_suite_out(eligible="FIST", down="FIST", hold_frames=1),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )
        clock.advance(0.25)
        released = controller.update(
            suite_out=_suite_out(eligible="BRAVO", up="FIST", hold_frames=1),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )

        self.assertEqual(pressed.intent.action_name, "NO_ACTION")
        self.assertEqual(pressed.reason, "draw_erase_pending")
        self.assertEqual(released.state, PresentationToolState.DRAW_IDLE)
        self.assertEqual(released.intent.action_name, "PRESENTATION_DRAW_UNDO")
        self.assertEqual(released.reason, "undo_one")

    def test_short_detected_fist_release_still_undoes_cleanly(self):
        clock = _FakeClock()
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                draw_undo_min_ms=60,
                draw_undo_max_ms=650,
                draw_undo_min_detected_frames=1,
                draw_clear_hold_ms=800,
            ),
            clock=clock.now,
        )
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        clock.set(0.0)
        controller.update(
            suite_out=_suite_out(eligible="FIST", down="FIST", hold_frames=1),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )
        clock.advance(0.04)
        released = controller.update(
            suite_out=_suite_out(eligible="BRAVO", up="FIST", hold_frames=1),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )

        self.assertEqual(released.intent.action_name, "PRESENTATION_DRAW_UNDO")
        self.assertEqual(released.reason, "undo_one")

    def test_relaxed_quick_fist_window_handles_natural_release_latency(self):
        clock = _FakeClock()
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                draw_undo_min_ms=60,
                draw_undo_max_ms=650,
                draw_clear_hold_ms=800,
                draw_clear_confirm_frames=2,
            ),
            clock=clock.now,
        )
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        clock.set(0.0)
        controller.update(
            suite_out=_suite_out(eligible="FIST", down="FIST", hold_frames=1),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )
        clock.advance(0.55)
        released = controller.update(
            suite_out=_suite_out(eligible="BRAVO", up="FIST", hold_frames=1),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )

        self.assertEqual(released.intent.action_name, "PRESENTATION_DRAW_UNDO")
        self.assertEqual(released.reason, "undo_one")

    def test_long_fist_hold_clears_annotations_once(self):
        clock = _FakeClock()
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                draw_undo_min_ms=60,
                draw_undo_max_ms=650,
                draw_clear_hold_ms=800,
                draw_clear_confirm_frames=2,
            ),
            clock=clock.now,
        )
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        clock.set(0.0)
        controller.update(
            suite_out=_suite_out(eligible="FIST", down="FIST", hold_frames=1),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )
        clock.advance(0.81)
        cleared = controller.update(
            suite_out=_suite_out(eligible="FIST", hold_frames=8),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )
        clock.advance(0.15)
        held = controller.update(
            suite_out=_suite_out(eligible="FIST", hold_frames=10),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )
        released = controller.update(
            suite_out=_suite_out(eligible="BRAVO", up="FIST", hold_frames=1),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )

        self.assertEqual(cleared.intent.action_name, "PRESENTATION_DRAW_CLEAR")
        self.assertEqual(cleared.reason, "clear_all")
        self.assertEqual(held.intent.action_name, "NO_ACTION")
        self.assertEqual(held.reason, "draw_erase_pending")
        self.assertEqual(released.intent.action_name, "NO_ACTION")

    def test_quick_fist_with_stale_temporal_fist_labels_still_undoes(self):
        clock = _FakeClock()
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                draw_undo_min_ms=60,
                draw_undo_max_ms=650,
                draw_clear_hold_ms=800,
            ),
            clock=clock.now,
        )
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        clock.set(0.0)
        controller.update(
            suite_out=_suite_out(eligible="FIST", down="FIST", raw_candidates={"FIST"}, candidates={"FIST"}),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )
        clock.advance(0.25)
        released = controller.update(
            suite_out=_suite_out(
                eligible="FIST",
                stable="FIST",
                chosen=None,
                raw_candidates=set(),
                candidates=set(),
            ),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )

        self.assertEqual(released.intent.action_name, "PRESENTATION_DRAW_UNDO")
        self.assertEqual(released.reason, "undo_one")

    def test_stale_temporal_fist_labels_do_not_trigger_clear_all(self):
        clock = _FakeClock()
        controller = PresentationToolController(
            cfg=PresentationToolConfig(
                draw_undo_min_ms=60,
                draw_undo_max_ms=650,
                draw_clear_hold_ms=800,
                draw_clear_confirm_frames=3,
            ),
            clock=clock.now,
        )
        controller.update(
            suite_out=_suite_out(eligible="BRAVO", down="BRAVO"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        clock.set(0.0)
        controller.update(
            suite_out=_suite_out(eligible="FIST", down="FIST", raw_candidates={"FIST"}, candidates={"FIST"}),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )
        clock.advance(0.81)
        released = controller.update(
            suite_out=_suite_out(
                eligible="FIST",
                stable="FIST",
                chosen=None,
                raw_candidates=set(),
                candidates=set(),
            ),
            hand_present=True,
            pointer_point=CursorPoint(0.25, 0.35),
        )

        self.assertNotEqual(released.intent.action_name, "PRESENTATION_DRAW_CLEAR")
        self.assertEqual(released.intent.action_name, "NO_ACTION")

    def test_reset_returns_controller_to_none(self):
        controller = PresentationToolController()
        controller.update(
            suite_out=_suite_out(eligible="L", down="L"),
            hand_present=True,
            pointer_point=CursorPoint(0.2, 0.3),
        )

        controller.reset()

        self.assertEqual(controller.state, PresentationToolState.NONE)


if __name__ == "__main__":
    unittest.main()
