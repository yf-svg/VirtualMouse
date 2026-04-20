from __future__ import annotations

import unittest

import app.control.presentation_overlay as presentation_overlay
from app.config import PresentationToolConfig
from app.control.cursor_space import CursorPoint
from app.control.mouse import ScreenPoint
from app.control.presentation_panel import (
    default_draw_color_key,
    default_draw_pen_key,
    default_draw_size_key,
    resolve_panel_state,
)
from app.control.presentation_overlay import NoOpPresentationOverlayBackend, map_pointer_to_window
from app.control.presentation_tool_execution import PresentationToolExecutor
from app.control.window_watch import PresentationAppKind, PresentationContext
from app.modes.presentation_tools import PresentationToolOut, PresentationToolState


def _context(
    *,
    allowed: bool = True,
    confident: bool = True,
    window_rect: tuple[int, int, int, int] | None = (100, 200, 500, 700),
) -> PresentationContext:
    return PresentationContext(
        allowed=allowed,
        confident=confident,
        kind=PresentationAppKind.POWERPOINT if allowed else PresentationAppKind.UNSUPPORTED,
        process_name="powerpnt.exe" if allowed else "notepad.exe",
        window_title="Deck - PowerPoint" if allowed else "notes",
        window_rect=window_rect,
        screen_size=(1920, 1080),
        fullscreen_like=bool(allowed),
        navigation_allowed=bool(allowed),
        supports_start=bool(allowed),
        supports_exit=bool(allowed),
        reason="powerpoint_foreground" if allowed else "unsupported_foreground_app",
    )


def _tool_out(
    *,
    state: PresentationToolState,
    pointer: CursorPoint | None,
    action_name: str = "NO_ACTION",
    selected_color_key: str | None = None,
    selected_pen_key: str | None = None,
    selected_size_key: str | None = None,
    panel_open: bool = False,
    stroke_capturing: bool | None = None,
) -> PresentationToolOut:
    color_key = selected_color_key or default_draw_color_key()
    pen_key = selected_pen_key or default_draw_pen_key()
    size_key = selected_size_key or default_draw_size_key()
    capture_active = state == PresentationToolState.DRAW_STROKING if stroke_capturing is None else bool(stroke_capturing)
    return PresentationToolOut(
        state=state,
        intent=type("Intent", (), {"action_name": action_name})(),
        pointer_point=pointer,
        owns_presentation=state in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING},
        stroke_active=state == PresentationToolState.DRAW_STROKING,
        stroke_capturing=capture_active,
        selected_color_key=color_key,
        selected_pen_key=pen_key,
        selected_size_key=size_key,
        panel_state=resolve_panel_state(
            pointer,
            draw_mode_active=state in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING},
            stroke_active=state == PresentationToolState.DRAW_STROKING,
            selected_color_key=color_key,
            selected_pen_key=pen_key,
            selected_size_key=size_key,
            panel_open=panel_open,
        ),
        reason="test",
    )


class PresentationOverlayTests(unittest.TestCase):
    def test_overlay_module_defines_winproc_return_type_fallback(self):
        self.assertIsNotNone(presentation_overlay._LRESULT)

    def test_map_pointer_to_window_clamps_to_rect(self):
        point = map_pointer_to_window(CursorPoint(-0.25, 1.25), (10, 20, 110, 220))

        self.assertEqual(point, ScreenPoint(10, 219))

    def test_executor_shows_laser_inside_presentation_rect(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            overlay_backend=backend,
        )

        report = executor.apply(
            _tool_out(state=PresentationToolState.LASER, pointer=CursorPoint(0.25, 0.5)),
            _context(),
        )

        self.assertTrue(report.performed)
        self.assertTrue(report.visible)
        self.assertEqual(report.reason, "laser_visible")
        self.assertEqual(report.laser_point, ScreenPoint(200, 450))
        self.assertEqual(report.stroke_count, 0)
        self.assertTrue(backend.states[-1].visible)
        self.assertEqual(backend.states[-1].laser_point, ScreenPoint(200, 450))

    def test_executor_shows_draw_cursor_inside_presentation_rect(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            overlay_backend=backend,
        )

        report = executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(0.25, 0.5)),
            _context(),
        )

        self.assertTrue(report.performed)
        self.assertTrue(report.visible)
        self.assertEqual(report.reason, "draw_ready_visible")
        self.assertEqual(report.draw_point, ScreenPoint(200, 450))
        self.assertEqual(backend.states[-1].draw_point, ScreenPoint(200, 450))
        self.assertEqual(backend.states[-1].draw_cursor_style, "pen")
        self.assertIsNone(backend.states[-1].panel)

    def test_executor_renders_draw_tray_when_open(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            overlay_backend=backend,
        )

        report = executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_IDLE,
                pointer=CursorPoint(0.84, 0.26),
                panel_open=True,
            ),
            _context(),
        )

        self.assertTrue(report.performed)
        self.assertTrue(report.visible)
        self.assertIsNotNone(backend.states[-1].panel)
        self.assertTrue(backend.states[-1].panel.visible)
        self.assertGreater(len(backend.states[-1].panel.items), 0)

    def test_executor_applies_selected_color_and_pen_to_strokes(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(
                draw_smoothing_alpha=1.0,
                draw_stroke_smoothing_alpha_min=1.0,
                draw_stroke_smoothing_alpha_max=1.0,
            ),
            overlay_backend=backend,
        )

        executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_STROKING,
                pointer=CursorPoint(0.10, 0.10),
                selected_color_key="azure",
                selected_pen_key="marker",
                selected_size_key="10",
            ),
            _context(),
        )
        report = executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_IDLE,
                pointer=CursorPoint(0.20, 0.20),
                selected_color_key="azure",
                selected_pen_key="marker",
                selected_size_key="10",
            ),
            _context(),
        )

        self.assertTrue(report.performed)
        self.assertEqual(report.stroke_count, 1)
        stroke = backend.states[-1].strokes[0]
        self.assertEqual(stroke.radius, 5)
        self.assertEqual(stroke.pen_kind, "marker")
        self.assertEqual(stroke.color_argb & 0x00FFFFFF, 0x5ED4FF)

    def test_executor_applies_selected_size_to_strokes(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(
                draw_smoothing_alpha=1.0,
                draw_stroke_smoothing_alpha_min=1.0,
                draw_stroke_smoothing_alpha_max=1.0,
            ),
            overlay_backend=backend,
        )

        executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_STROKING,
                pointer=CursorPoint(0.10, 0.10),
                selected_pen_key="pen",
                selected_size_key="05",
            ),
            _context(),
        )
        small_report = executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_IDLE,
                pointer=CursorPoint(0.20, 0.20),
                selected_pen_key="pen",
                selected_size_key="05",
            ),
            _context(),
        )

        executor.reset()

        executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_STROKING,
                pointer=CursorPoint(0.10, 0.10),
                selected_pen_key="pen",
                selected_size_key="20",
            ),
            _context(),
        )
        large_report = executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_IDLE,
                pointer=CursorPoint(0.20, 0.20),
                selected_pen_key="pen",
                selected_size_key="20",
            ),
            _context(),
        )

        self.assertEqual(small_report.stroke_count, 1)
        self.assertEqual(large_report.stroke_count, 1)
        small_stroke = backend.states[1].strokes[0]
        large_stroke = backend.states[-1].strokes[0]
        self.assertLess(small_stroke.radius, large_stroke.radius)

    def test_executor_does_not_append_release_grace_pointer_to_stroke(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(
                draw_smoothing_alpha=1.0,
                draw_stroke_smoothing_alpha_min=1.0,
                draw_stroke_smoothing_alpha_max=1.0,
            ),
            overlay_backend=backend,
        )

        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.10, 0.10)),
            _context(),
        )
        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.20, 0.20)),
            _context(),
        )
        report = executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_STROKING,
                pointer=CursorPoint(0.30, 0.30),
                stroke_capturing=False,
            ),
            _context(),
        )

        self.assertTrue(report.performed)
        self.assertTrue(report.visible)
        self.assertEqual(report.stroke_count, 1)
        self.assertEqual(
            backend.states[-1].strokes[0].points,
            (ScreenPoint(140, 250), ScreenPoint(180, 300)),
        )

    def test_executor_hides_when_context_has_no_window_rect(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            overlay_backend=backend,
        )

        report = executor.apply(
            _tool_out(state=PresentationToolState.LASER, pointer=CursorPoint(0.25, 0.5)),
            _context(window_rect=None),
        )

        self.assertFalse(report.performed)
        self.assertFalse(report.visible)
        self.assertEqual(report.reason, "presentation_tool_missing_window_rect")
        self.assertEqual(backend.reset_calls, 1)

    def test_executor_persists_draw_annotations_after_tool_is_released(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(
                draw_smoothing_alpha=1.0,
                draw_stroke_smoothing_alpha_min=1.0,
                draw_stroke_smoothing_alpha_max=1.0,
            ),
            overlay_backend=backend,
        )

        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.10, 0.10)),
            _context(),
        )
        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.20, 0.20)),
            _context(),
        )
        report = executor.apply(
            _tool_out(state=PresentationToolState.NONE, pointer=None),
            _context(),
        )

        self.assertTrue(report.performed)
        self.assertTrue(report.visible)
        self.assertEqual(report.reason, "annotations_visible")
        self.assertEqual(report.stroke_count, 1)
        self.assertIsNone(report.draw_point)
        self.assertEqual(len(backend.states[-1].strokes), 1)
        self.assertEqual(
            backend.states[-1].strokes[0].points,
            (ScreenPoint(140, 250), ScreenPoint(180, 300)),
        )

    def test_executor_reuses_committed_stroke_render_points_until_window_changes(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(
                draw_smoothing_alpha=1.0,
                draw_stroke_smoothing_alpha_min=1.0,
                draw_stroke_smoothing_alpha_max=1.0,
            ),
            overlay_backend=backend,
        )

        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.10, 0.10)),
            _context(),
        )
        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.20, 0.20)),
            _context(),
        )
        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(0.20, 0.20)),
            _context(),
        )
        cached_points = backend.states[-1].strokes[0].points

        executor.apply(
            _tool_out(state=PresentationToolState.NONE, pointer=None),
            _context(),
        )
        reused_points = backend.states[-1].strokes[0].points

        executor.apply(
            _tool_out(state=PresentationToolState.NONE, pointer=None),
            _context(window_rect=(0, 0, 800, 800)),
        )
        remapped_points = backend.states[-1].strokes[0].points

        self.assertIs(cached_points, reused_points)
        self.assertIsNot(cached_points, remapped_points)
        self.assertEqual(remapped_points, (ScreenPoint(80, 80), ScreenPoint(160, 160)))

    def test_executor_hides_when_no_tool_and_no_annotations(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            overlay_backend=backend,
        )

        report = executor.apply(
            _tool_out(state=PresentationToolState.NONE, pointer=None),
            _context(),
        )

        self.assertFalse(report.performed)
        self.assertFalse(report.visible)
        self.assertEqual(report.reason, "presentation_tool_not_visible")
        self.assertEqual(backend.reset_calls, 1)

    def test_executor_respects_live_policy_disable(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=False,
            overlay_backend=backend,
        )

        report = executor.apply(
            _tool_out(state=PresentationToolState.LASER, pointer=CursorPoint(0.25, 0.5)),
            _context(),
        )

        self.assertFalse(report.performed)
        self.assertFalse(report.visible)
        self.assertEqual(report.reason, "presentation_tools_disabled_by_policy")

    def test_executor_respects_tool_feature_flag_disable(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(enable_live_presentation_tools=False),
            overlay_backend=backend,
        )

        report = executor.apply(
            _tool_out(state=PresentationToolState.LASER, pointer=CursorPoint(0.25, 0.5)),
            _context(),
        )

        self.assertFalse(report.performed)
        self.assertFalse(report.visible)
        self.assertEqual(report.reason, "presentation_tools_disabled_by_policy")

    def test_executor_smooths_laser_pointer_motion(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(laser_smoothing_alpha=0.5),
            overlay_backend=backend,
        )

        first = executor.apply(
            _tool_out(state=PresentationToolState.LASER, pointer=CursorPoint(0.0, 0.5)),
            _context(),
        )
        second = executor.apply(
            _tool_out(state=PresentationToolState.LASER, pointer=CursorPoint(1.0, 0.5)),
            _context(),
        )

        self.assertEqual(first.laser_point, ScreenPoint(100, 450))
        self.assertEqual(second.laser_point, ScreenPoint(300, 450))

    def test_executor_holds_last_laser_point_through_brief_pointer_loss(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(laser_smoothing_alpha=1.0, laser_hold_last_frames=2),
            overlay_backend=backend,
        )

        first = executor.apply(
            _tool_out(state=PresentationToolState.LASER, pointer=CursorPoint(0.25, 0.9)),
            _context(),
        )
        held = executor.apply(
            _tool_out(state=PresentationToolState.LASER, pointer=None),
            _context(),
        )
        expired = executor.apply(
            _tool_out(state=PresentationToolState.LASER, pointer=None),
            _context(),
        )
        gone = executor.apply(
            _tool_out(state=PresentationToolState.LASER, pointer=None),
            _context(),
        )

        self.assertEqual(first.laser_point, ScreenPoint(200, 649))
        self.assertTrue(held.visible)
        self.assertEqual(held.laser_point, ScreenPoint(200, 649))
        self.assertTrue(expired.visible)
        self.assertEqual(expired.laser_point, ScreenPoint(200, 649))
        self.assertFalse(gone.visible)
        self.assertEqual(gone.reason, "presentation_tool_not_visible")

    def test_executor_holds_last_draw_point_through_brief_pointer_loss(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(draw_smoothing_alpha=1.0, draw_hold_last_frames=2),
            overlay_backend=backend,
        )

        first = executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(0.25, 0.90)),
            _context(),
        )
        held = executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=None),
            _context(),
        )
        expired = executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=None),
            _context(),
        )
        gone = executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=None),
            _context(),
        )

        self.assertEqual(first.draw_point, ScreenPoint(200, 649))
        self.assertTrue(held.visible)
        self.assertEqual(held.draw_point, ScreenPoint(200, 649))
        self.assertTrue(expired.visible)
        self.assertEqual(expired.draw_point, ScreenPoint(200, 649))
        self.assertFalse(gone.visible)
        self.assertEqual(gone.reason, "presentation_tool_not_visible")

    def test_executor_uses_faster_smoothing_for_active_stroke_than_draw_idle(self):
        idle_executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(
                draw_smoothing_alpha=0.40,
                draw_stroke_smoothing_alpha_min=0.80,
                draw_stroke_smoothing_alpha_max=0.80,
            ),
            overlay_backend=NoOpPresentationOverlayBackend(),
        )
        stroke_executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(
                draw_smoothing_alpha=0.40,
                draw_stroke_smoothing_alpha_min=0.80,
                draw_stroke_smoothing_alpha_max=0.80,
            ),
            overlay_backend=NoOpPresentationOverlayBackend(),
        )

        idle_executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(0.0, 0.5)),
            _context(),
        )
        idle_second = idle_executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(1.0, 0.5)),
            _context(),
        )

        stroke_executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.0, 0.5)),
            _context(),
        )
        stroke_second = stroke_executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(1.0, 0.5)),
            _context(),
        )

        self.assertIsNotNone(idle_second.draw_point)
        self.assertIsNotNone(stroke_second.draw_point)
        self.assertGreater(stroke_second.draw_point.x, idle_second.draw_point.x)

    def test_executor_resets_idle_smoothing_when_stroke_begins(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(
                draw_smoothing_alpha=0.20,
                draw_stroke_smoothing_alpha_min=1.0,
                draw_stroke_smoothing_alpha_max=1.0,
            ),
            overlay_backend=backend,
        )

        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(0.0, 0.5)),
            _context(),
        )
        idle = executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(1.0, 0.5)),
            _context(),
        )
        stroke = executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(1.0, 0.5)),
            _context(),
        )

        self.assertIsNotNone(idle.draw_point)
        self.assertIsNotNone(stroke.draw_point)
        self.assertLess(idle.draw_point.x, 499)
        self.assertEqual(stroke.draw_point, ScreenPoint(499, 450))

    def test_executor_clears_annotations_when_draw_clear_action_is_emitted(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(draw_smoothing_alpha=1.0),
            overlay_backend=backend,
        )

        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.10, 0.10)),
            _context(),
        )
        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.20, 0.20)),
            _context(),
        )

        report = executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_IDLE,
                pointer=CursorPoint(0.30, 0.30),
                action_name="PRESENTATION_DRAW_CLEAR",
            ),
            _context(),
        )

        self.assertTrue(report.performed)
        self.assertTrue(report.visible)
        self.assertEqual(report.reason, "draw_cleared_visible")
        self.assertEqual(report.stroke_count, 0)
        self.assertEqual(report.draw_point, ScreenPoint(220, 350))
        self.assertEqual(backend.states[-1].strokes, ())
        self.assertEqual(backend.states[-1].draw_cursor_style, "eraser")

    def test_executor_undoes_only_one_stroke_for_draw_undo_action(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(draw_smoothing_alpha=1.0),
            overlay_backend=backend,
        )

        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.10, 0.10)),
            _context(),
        )
        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(0.20, 0.20)),
            _context(),
        )
        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_STROKING, pointer=CursorPoint(0.30, 0.30)),
            _context(),
        )
        executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(0.40, 0.40)),
            _context(),
        )

        report = executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_IDLE,
                pointer=CursorPoint(0.45, 0.45),
                action_name="PRESENTATION_DRAW_UNDO",
            ),
            _context(),
        )

        self.assertTrue(report.performed)
        self.assertTrue(report.visible)
        self.assertEqual(report.reason, "draw_undo_visible")
        self.assertEqual(report.stroke_count, 1)
        self.assertEqual(backend.states[-1].draw_cursor_style, "eraser")

    def test_executor_reverts_to_pen_cursor_after_eraser_feedback_expires(self):
        backend = NoOpPresentationOverlayBackend()
        executor = PresentationToolExecutor(
            live_enabled=True,
            cfg=PresentationToolConfig(draw_smoothing_alpha=1.0, draw_clear_feedback_frames=2),
            overlay_backend=backend,
        )

        executor.apply(
            _tool_out(
                state=PresentationToolState.DRAW_IDLE,
                pointer=CursorPoint(0.30, 0.30),
                action_name="PRESENTATION_DRAW_CLEAR",
            ),
            _context(),
        )
        flash = executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(0.31, 0.31)),
            _context(),
        )
        settled = executor.apply(
            _tool_out(state=PresentationToolState.DRAW_IDLE, pointer=CursorPoint(0.32, 0.32)),
            _context(),
        )

        self.assertTrue(flash.visible)
        self.assertEqual(backend.states[-2].draw_cursor_style, "eraser")
        self.assertTrue(settled.visible)
        self.assertEqual(backend.states[-1].draw_cursor_style, "pen")


if __name__ == "__main__":
    unittest.main()
