from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.config import (
    ClutchInteractionConfig,
    CursorPolicyConfig,
    CursorPreviewConfig,
    ExecutionConfig,
    PrimaryInteractionConfig,
    ScrollInteractionConfig,
    SecondaryInteractionConfig,
)
from app.control.clutch import ClutchController
from app.control.cursor_policy import CursorPolicy
from app.control.cursor_preview import CursorPreviewController
from app.control.cursor_space import CursorPoint
from app.control.execution import OSActionExecutor
from app.control.execution_safety import ExecutionSafetyGate
from app.control.mouse import NoOpMouseBackend, ScreenPoint
from app.control.primary_interaction import PrimaryInteractionController
from app.control.scroll_mode import ScrollModeController
from app.control.secondary_interaction import SecondaryInteractionController
from app.lifecycle.runtime_status import _format_execution_policy_status
from app.modes.general import resolve_general_action


def pt(x: float, y: float) -> CursorPoint:
    return CursorPoint(x=x, y=y)


def _policy_ctx(executor):
    return type(
        "Ctx",
        (),
        {
            "executor": executor,
            "override_policy": type("OverridePolicy", (), {"status_text": lambda self: "EXEC:INHERIT|ROUTE:AUTO"})(),
        },
    )()


@dataclass
class _SuiteOut:
    chosen: str | None
    stable: str | None
    eligible: str | None
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


class Phase6OsExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = NoOpMouseBackend(width=100, height=200)
        self.clutch = ClutchController(cfg=ClutchInteractionConfig(hand_loss_grace_s=0.10))
        self.scroll = ScrollModeController(
            cfg=ScrollInteractionConfig(
                dead_zone=0.010,
                axis_dominance_margin=0.004,
                pause_reset_s=0.20,
                pause_motion_epsilon=0.003,
                scroll_gain=1.0,
                hand_loss_grace_s=0.10,
            )
        )
        self.primary = PrimaryInteractionController(
            cfg=PrimaryInteractionConfig(
                drag_start_distance=0.040,
                click_release_tolerance=0.020,
                double_click_window_s=0.30,
                hand_loss_grace_s=0.10,
            )
        )
        self.secondary = SecondaryInteractionController(
            cfg=SecondaryInteractionConfig(
                release_tolerance=0.025,
                hand_loss_grace_s=0.10,
            )
        )
        self.cursor = CursorPreviewController(
            policy=CursorPolicy(
                cfg=CursorPolicyConfig(
                    allowed_gestures=("L",),
                    provisional=True,
                )
            ),
            cfg=CursorPreviewConfig(
                move_epsilon=0.002,
                gain=1.0,
            ),
        )

    def _executor(self, *, live_os: bool = True) -> OSActionExecutor:
        return OSActionExecutor(
            cfg=ExecutionConfig(
                profile="live",
                enable_live_os=live_os,
                enable_live_cursor=True,
                enable_live_primary=True,
                enable_live_secondary=True,
                enable_live_scroll=True,
                scroll_units_per_motion=1000.0,
            ),
            mouse_backend=self.backend,
        )

    def _resolve(self, gesture_label: str | None, cursor_point: CursorPoint | None, now: float):
        return resolve_general_action(
            gesture_label=gesture_label,
            cursor_point=cursor_point,
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=now,
        )

    @staticmethod
    def _suite_out(
        *,
        stable: str | None,
        eligible: str | None,
        feature_reason: str = "ok",
        gate_reason: str = "stable_match",
    ) -> _SuiteOut:
        return _SuiteOut(
            chosen=eligible,
            stable=stable,
            eligible=eligible,
            candidates={stable} if stable is not None else set(),
            reason="test",
            down=None,
            up=None,
            source="rules",
            confidence=None,
            rule_chosen=stable,
            ml_chosen=None,
            ml_reason="none",
            feature_reason=feature_reason,
            hold_frames=2,
            gate_reason=gate_reason,
        )

    def test_execution_stays_globally_disabled(self):
        executor = OSActionExecutor(
            cfg=ExecutionConfig(
                profile="dry_run",
                enable_live_os=False,
                enable_live_cursor=True,
                enable_live_primary=True,
                enable_live_secondary=True,
                enable_live_scroll=True,
            ),
            mouse_backend=self.backend,
        )

        out = self._resolve("L", pt(0.20, 0.20), 0.00)
        report = executor.apply_general_mode(out)

        self.assertFalse(report.cursor.performed)
        self.assertEqual(report.cursor.reason, "dry_run_profile")
        self.assertEqual(self.backend.moves, [])
        self.assertEqual(self.backend.left_clicks, 0)
        self.assertEqual(self.backend.right_clicks, 0)

    def test_live_cursor_move_uses_preview_output_as_source_of_truth(self):
        executor = self._executor()
        self._resolve("L", pt(0.20, 0.20), 0.00)
        out = self._resolve("L", pt(0.30, 0.40), 0.05)

        report = executor.apply_general_mode(
            out,
            safety=ExecutionSafetyGate().evaluate(
                suite_out=self._suite_out(stable="L", eligible="L"),
                general_out=out,
                hand_present=True,
            ),
        )

        self.assertTrue(report.cursor.performed)
        self.assertEqual(report.cursor.reason, "cursor_moved")
        self.assertEqual(report.cursor.target, ScreenPoint(30, 80))
        self.assertEqual(self.backend.moves, [ScreenPoint(30, 80)])

    def test_live_primary_single_click_emits_once_after_pending_window(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("PINCH_INDEX", pt(0.20, 0.20), 0.00)
        self._resolve("L", pt(0.20, 0.20), 0.05)
        out = self._resolve("L", pt(0.20, 0.20), 0.40)

        report = executor.apply_general_mode(
            out,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="L", eligible="L"),
                general_out=out,
                hand_present=True,
            ),
        )

        self.assertTrue(report.primary.performed)
        self.assertEqual(report.primary.reason, "primary_click_emitted")
        self.assertEqual(self.backend.left_clicks, 1)
        self.assertEqual(self.backend.double_left_clicks, 0)

    def test_live_primary_double_click_does_not_emit_single_click_first(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("PINCH_INDEX", pt(0.20, 0.20), 0.00)
        self._resolve("L", pt(0.20, 0.20), 0.05)
        self._resolve("PINCH_INDEX", pt(0.20, 0.20), 0.15)
        out = self._resolve("L", pt(0.20, 0.20), 0.20)

        report = executor.apply_general_mode(
            out,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="L", eligible="L"),
                general_out=out,
                hand_present=True,
            ),
        )

        self.assertTrue(report.primary.performed)
        self.assertEqual(report.primary.reason, "primary_double_click_emitted")
        self.assertEqual(self.backend.left_clicks, 0)
        self.assertEqual(self.backend.double_left_clicks, 1)

    def test_live_drag_press_hold_release_is_faithful_to_primary_outputs(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("PINCH_INDEX", pt(0.20, 0.20), 0.00)
        start = self._resolve("PINCH_INDEX", pt(0.30, 0.20), 0.05)
        hold = self._resolve("PINCH_INDEX", pt(0.34, 0.24), 0.10)
        end = self._resolve("L", pt(0.36, 0.25), 0.15)

        start_report = executor.apply_general_mode(
            start,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="PINCH_INDEX", eligible="PINCH_INDEX"),
                general_out=start,
                hand_present=True,
            ),
        )
        hold_report = executor.apply_general_mode(
            hold,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="PINCH_INDEX", eligible="PINCH_INDEX"),
                general_out=hold,
                hand_present=True,
            ),
        )
        end_report = executor.apply_general_mode(
            end,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="L", eligible="L"),
                general_out=end,
                hand_present=True,
            ),
        )

        self.assertTrue(start_report.primary.performed)
        self.assertEqual(start_report.primary.reason, "primary_drag_started")
        self.assertEqual(self.backend.left_downs, 1)
        self.assertEqual(self.backend.left_ups, 1)
        self.assertTrue(hold_report.primary.performed)
        self.assertEqual(hold_report.primary.reason, "primary_drag_moved")
        self.assertTrue(end_report.primary.performed)
        self.assertEqual(end_report.primary.reason, "primary_drag_ended")
        self.assertGreaterEqual(len(self.backend.moves), 2)

    def test_drag_is_released_safely_when_clutch_cancels_primary_ownership(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("PINCH_INDEX", pt(0.20, 0.20), 0.00)
        drag = self._resolve("PINCH_INDEX", pt(0.30, 0.20), 0.05)
        clutch = self._resolve("FIST", pt(0.30, 0.20), 0.10)

        executor.apply_general_mode(
            drag,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="PINCH_INDEX", eligible="PINCH_INDEX"),
                general_out=drag,
                hand_present=True,
            ),
        )
        report = executor.apply_general_mode(
            clutch,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="FIST", eligible="FIST"),
                general_out=clutch,
                hand_present=True,
            ),
        )

        self.assertTrue(report.primary.performed)
        self.assertEqual(report.primary.reason, "primary_drag_cancelled")
        self.assertEqual(self.backend.left_downs, 1)
        self.assertEqual(self.backend.left_ups, 1)
        self.assertEqual(self.backend.left_clicks, 0)

    def test_live_secondary_right_click_emits_once_on_valid_release(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("PINCH_MIDDLE", pt(0.40, 0.30), 0.00)
        out = self._resolve("L", pt(0.40, 0.30), 0.05)
        idle = self._resolve("L", pt(0.40, 0.30), 0.10)

        report = executor.apply_general_mode(
            out,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="L", eligible="L"),
                general_out=out,
                hand_present=True,
            ),
        )
        idle_report = executor.apply_general_mode(
            idle,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="L", eligible="L"),
                general_out=idle,
                hand_present=True,
            ),
        )

        self.assertTrue(report.secondary.performed)
        self.assertEqual(report.secondary.reason, "secondary_right_click_emitted")
        self.assertFalse(idle_report.secondary.performed)
        self.assertEqual(self.backend.right_clicks, 1)

    def test_live_vertical_scroll_uses_controller_locked_axis(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("SHAKA", pt(0.50, 0.50), 0.00)
        self._resolve(None, pt(0.50, 0.56), 0.05)
        out = self._resolve(None, pt(0.50, 0.62), 0.10)

        report = executor.apply_general_mode(
            out,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable=None, eligible=None),
                general_out=out,
                hand_present=True,
            ),
        )

        self.assertTrue(report.scroll.performed)
        self.assertEqual(report.scroll.reason, "scroll_vertical_emitted")
        self.assertEqual(self.backend.vertical_scrolls, [60])
        self.assertEqual(self.backend.horizontal_scrolls, [])

    def test_live_horizontal_scroll_uses_controller_locked_axis(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("SHAKA", pt(0.50, 0.50), 0.00)
        self._resolve(None, pt(0.56, 0.50), 0.05)
        out = self._resolve(None, pt(0.62, 0.50), 0.10)

        report = executor.apply_general_mode(
            out,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable=None, eligible=None),
                general_out=out,
                hand_present=True,
            ),
        )

        self.assertTrue(report.scroll.performed)
        self.assertEqual(report.scroll.reason, "scroll_horizontal_emitted")
        self.assertEqual(self.backend.horizontal_scrolls, [60])
        self.assertEqual(self.backend.vertical_scrolls, [])

    def test_canceled_secondary_and_idle_scroll_emit_nothing(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("PINCH_MIDDLE", pt(0.10, 0.10), 0.00)
        cancelled = self._resolve("PINCH_MIDDLE", pt(0.20, 0.20), 0.05)
        self._resolve("SHAKA", pt(0.50, 0.50), 0.10)
        idle_scroll = self._resolve(None, pt(0.505, 0.505), 0.15)

        cancelled_report = executor.apply_general_mode(
            cancelled,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="PINCH_MIDDLE", eligible="PINCH_MIDDLE"),
                general_out=cancelled,
                hand_present=True,
            ),
        )
        scroll_report = executor.apply_general_mode(
            idle_scroll,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable=None, eligible=None),
                general_out=idle_scroll,
                hand_present=True,
            ),
        )

        self.assertFalse(cancelled_report.secondary.performed)
        self.assertEqual(cancelled_report.secondary.reason, "secondary_no_action")
        self.assertFalse(scroll_report.scroll.performed)
        self.assertIn(scroll_report.scroll.reason, {"scroll_no_action", "scroll_units_zero"})
        self.assertEqual(self.backend.right_clicks, 0)
        self.assertEqual(self.backend.vertical_scrolls, [])
        self.assertEqual(self.backend.horizontal_scrolls, [])

    def test_invalid_profile_fails_safe_even_if_subsystems_are_enabled(self):
        executor = OSActionExecutor(
            cfg=ExecutionConfig(
                profile="unsafe",
                enable_live_os=True,
                enable_live_cursor=True,
                enable_live_primary=True,
                enable_live_secondary=True,
                enable_live_scroll=True,
            ),
            mouse_backend=self.backend,
        )

        out = self._resolve("L", pt(0.30, 0.40), 0.05)
        report = executor.apply_general_mode(out)

        self.assertEqual(executor.policy.reason, "invalid_execution_profile")
        self.assertFalse(report.cursor.performed)
        self.assertEqual(report.cursor.reason, "invalid_execution_profile")

    def test_live_profile_requires_master_enable(self):
        executor = OSActionExecutor(
            cfg=ExecutionConfig(
                profile="live",
                enable_live_os=False,
                enable_live_cursor=True,
                enable_live_primary=True,
            ),
            mouse_backend=self.backend,
        )

        out = self._resolve("L", pt(0.30, 0.40), 0.05)
        report = executor.apply_general_mode(out)

        self.assertEqual(executor.policy.reason, "live_profile_missing_master_enable")
        self.assertFalse(report.cursor.performed)

    def test_no_cursor_live_move_when_cursor_subsystem_is_disabled(self):
        executor = OSActionExecutor(
            cfg=ExecutionConfig(
                profile="live",
                enable_live_os=True,
                enable_live_cursor=False,
                enable_live_primary=True,
            ),
            mouse_backend=self.backend,
        )

        self._resolve("L", pt(0.20, 0.20), 0.00)
        out = self._resolve("L", pt(0.30, 0.40), 0.05)
        report = executor.apply_general_mode(
            out,
            safety=ExecutionSafetyGate().evaluate(
                suite_out=self._suite_out(stable="L", eligible="L"),
                general_out=out,
                hand_present=True,
            ),
        )

        self.assertFalse(report.cursor.performed)
        self.assertEqual(report.cursor.reason, "cursor_disabled_by_policy")
        self.assertEqual(self.backend.moves, [])

    def test_tainted_primary_click_is_suppressed_after_instability(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("PINCH_INDEX", pt(0.20, 0.20), 0.00)
        pending = self._resolve("L", pt(0.20, 0.20), 0.05)
        gate.evaluate(
            suite_out=self._suite_out(stable="PINCH_INDEX", eligible=None, feature_reason="unstable"),
            general_out=pending,
            hand_present=True,
        )
        click = self._resolve("L", pt(0.20, 0.20), 0.40)

        report = executor.apply_general_mode(
            click,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="L", eligible="L"),
                general_out=click,
                hand_present=True,
            ),
        )

        self.assertFalse(report.primary.performed)
        self.assertEqual(report.primary.reason, "suppressed_tainted_click")
        self.assertEqual(self.backend.left_clicks, 0)

    def test_primary_drag_releases_on_hand_loss_suppression(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("PINCH_INDEX", pt(0.20, 0.20), 0.00)
        drag = self._resolve("PINCH_INDEX", pt(0.30, 0.20), 0.05)
        lost = self._resolve(None, None, 0.06)

        executor.apply_general_mode(
            drag,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="PINCH_INDEX", eligible="PINCH_INDEX"),
                general_out=drag,
                hand_present=True,
            ),
        )
        report = executor.apply_general_mode(
            lost,
            safety=gate.evaluate(
                suite_out=None,
                general_out=lost,
                hand_present=False,
            ),
        )

        self.assertTrue(report.primary.performed)
        self.assertEqual(report.primary.reason, "primary_drag_safety_release")
        self.assertEqual(self.backend.left_downs, 1)
        self.assertEqual(self.backend.left_ups, 1)

    def test_no_right_click_on_unstable_secondary_transition(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("PINCH_MIDDLE", pt(0.40, 0.30), 0.00)
        out = self._resolve("L", pt(0.40, 0.30), 0.05)

        report = executor.apply_general_mode(
            out,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable="L", eligible="L", feature_reason="unstable"),
                general_out=out,
                hand_present=True,
            ),
        )

        self.assertFalse(report.secondary.performed)
        self.assertEqual(report.secondary.reason, "suppressed_unstable_prediction")
        self.assertEqual(self.backend.right_clicks, 0)

    def test_no_scroll_on_unstable_scroll_output(self):
        executor = self._executor()
        gate = ExecutionSafetyGate()
        self._resolve("SHAKA", pt(0.50, 0.50), 0.00)
        self._resolve(None, pt(0.50, 0.56), 0.05)
        out = self._resolve(None, pt(0.50, 0.62), 0.10)

        report = executor.apply_general_mode(
            out,
            safety=gate.evaluate(
                suite_out=self._suite_out(stable=None, eligible=None, feature_reason="unstable"),
                general_out=out,
                hand_present=True,
            ),
        )

        self.assertFalse(report.scroll.performed)
        self.assertEqual(report.scroll.reason, "suppressed_unstable_prediction")
        self.assertEqual(self.backend.vertical_scrolls, [])

    def test_policy_status_is_visible_for_overlay(self):
        executor = self._executor()
        safety = ExecutionSafetyGate().evaluate(
            suite_out=self._suite_out(stable="L", eligible="L"),
            general_out=self._resolve("L", pt(0.30, 0.40), 0.05),
            hand_present=True,
        )

        status = _format_execution_policy_status(
            _policy_ctx(executor),
            safety.summary(),
        )

        self.assertIn("OVR:EXEC:INHERIT|ROUTE:AUTO", status)
        self.assertIn("XPOL:LIVE:cur,pri,sec,scr", status)
        self.assertIn("SAFE:ok", status)


if __name__ == "__main__":
    unittest.main()
