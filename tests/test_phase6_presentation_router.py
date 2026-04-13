from __future__ import annotations

import unittest

from app.config import ExecutionConfig, PresentationContextConfig
from app.constants import AppState
from app.control.execution import OSActionExecutor
from app.control.execution_safety import ExecutionSafetyGate
from app.control.keyboard import NoOpKeyboardBackend
from app.control.mouse import NoOpMouseBackend
from app.control.window_watch import (
    ForegroundWindowSnapshot,
    PresentationAppKind,
    StubForegroundWindowBackend,
    WindowWatch,
)
from app.gestures.suite import GestureSuiteOut
from app.lifecycle.runtime_loop import _format_execution_policy_status, _format_presentation_context
from app.modes.presentation import map_gesture_to_action as map_presentation_action
from app.modes.presentation import resolve_presentation_action
from app.modes.router import ModeRouter
from app.security.auth import GestureAuth, GestureAuthCfg


def _suite_out(*, eligible: str | None, feature_reason: str = "ok") -> GestureSuiteOut:
    return GestureSuiteOut(
        chosen=eligible,
        stable=eligible,
        eligible=eligible,
        candidates={eligible} if eligible is not None else set(),
        reason="test",
        down=None,
        up=None,
        source="rules",
        confidence=None,
        rule_chosen=eligible,
        ml_chosen=None,
        ml_reason="none",
        feature_reason=feature_reason,
        hold_frames=2,
        gate_reason="stable_match",
    )


class Phase6PresentationRouterTests(unittest.TestCase):
    def test_window_watch_defaults_to_not_allowed(self):
        watch = WindowWatch(
            backend=StubForegroundWindowBackend(),
            cfg=PresentationContextConfig(),
        )
        context = watch.presentation_context()
        self.assertFalse(context.allowed)
        self.assertEqual(context.kind, PresentationAppKind.NONE)

    def test_window_watch_detects_powerpoint_context(self):
        watch = WindowWatch(
            backend=StubForegroundWindowBackend(
                ForegroundWindowSnapshot(
                    process_name="powerpnt.exe",
                    window_title="Quarterly Review - PowerPoint",
                    window_rect=(50, 50, 1050, 750),
                    screen_size=(1920, 1080),
                    valid=True,
                    reason="ok",
                )
            ),
            cfg=PresentationContextConfig(),
        )

        context = watch.presentation_context()

        self.assertTrue(context.allowed)
        self.assertTrue(context.confident)
        self.assertEqual(context.kind, PresentationAppKind.POWERPOINT)
        self.assertTrue(context.supports_start)
        self.assertFalse(context.navigation_allowed)

    def test_window_watch_detects_fullscreen_browser_presentation_context(self):
        watch = WindowWatch(
            backend=StubForegroundWindowBackend(
                ForegroundWindowSnapshot(
                    process_name="msedge.exe",
                    window_title="Demo Deck - Google Slides",
                    window_rect=(0, 0, 1919, 1079),
                    screen_size=(1920, 1080),
                    valid=True,
                    reason="ok",
                )
            ),
            cfg=PresentationContextConfig(),
        )

        context = watch.presentation_context()

        self.assertTrue(context.allowed)
        self.assertEqual(context.kind, PresentationAppKind.BROWSER_PRESENTATION)
        self.assertTrue(context.navigation_allowed)
        self.assertTrue(context.supports_start)
        self.assertTrue(context.supports_exit)

    def test_window_watch_rejects_ambiguous_browser_context(self):
        watch = WindowWatch(
            backend=StubForegroundWindowBackend(
                ForegroundWindowSnapshot(
                    process_name="chrome.exe",
                    window_title="Inbox - Gmail",
                    window_rect=(0, 0, 1919, 1079),
                    screen_size=(1920, 1080),
                    valid=True,
                    reason="ok",
                )
            ),
            cfg=PresentationContextConfig(),
        )

        context = watch.presentation_context()

        self.assertFalse(context.allowed)
        self.assertFalse(context.confident)
        self.assertEqual(context.kind, PresentationAppKind.UNSUPPORTED)

    def test_router_enters_presentation_only_from_active_general(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))
        router.route_auth_edge("ONE", now=1.0)
        router.route_auth_edge("TWO", now=2.0)

        out = router.sync_presentation_permission(True)

        self.assertEqual(out.state, AppState.ACTIVE_PRESENTATION)
        self.assertEqual(out.suite_key, "ops")
        self.assertEqual(out.auth_progress_text, "Presentation active")

    def test_router_returns_to_general_when_presentation_permission_is_removed(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))
        router.route_auth_edge("ONE", now=1.0)
        router.route_auth_edge("TWO", now=2.0)
        router.sync_presentation_permission(True)

        out = router.sync_presentation_permission(False)

        self.assertEqual(out.state, AppState.ACTIVE_GENERAL)
        self.assertEqual(out.suite_key, "ops")

    def test_router_does_not_enter_presentation_while_locked(self):
        router = ModeRouter()

        out = router.sync_presentation_permission(True)

        self.assertEqual(out.state, AppState.IDLE_LOCKED)
        self.assertEqual(out.suite_key, "auth")

    def test_presentation_mode_dry_run_defaults_to_no_action_without_context(self):
        out = map_presentation_action("POINT_RIGHT")

        self.assertEqual(out.action_name, "NO_ACTION")
        self.assertEqual(out.mode, "PRESENTATION")
        self.assertFalse(out.executable)

    def test_resolve_presentation_action_maps_reserved_gestures_conservatively(self):
        watch = WindowWatch(
            backend=StubForegroundWindowBackend(
                ForegroundWindowSnapshot(
                    process_name="powerpnt.exe",
                    window_title="Demo Deck - PowerPoint",
                    window_rect=(10, 10, 1000, 700),
                    screen_size=(1920, 1080),
                    valid=True,
                    reason="ok",
                )
            )
        )
        context = watch.presentation_context()

        start = resolve_presentation_action(gesture_label="OPEN_PALM", context=context)
        next_slide = resolve_presentation_action(gesture_label="POINT_RIGHT", context=context)

        self.assertEqual(start.intent.action_name, "PRESENT_START")
        self.assertEqual(next_slide.intent.action_name, "NO_ACTION")
        self.assertIn("navigation_blocked", next_slide.intent.reason)

    def test_resolve_presentation_action_rejects_unsupported_context(self):
        watch = WindowWatch(
            backend=StubForegroundWindowBackend(
                ForegroundWindowSnapshot(
                    process_name="notepad.exe",
                    window_title="notes.txt - Notepad",
                    window_rect=(0, 0, 800, 600),
                    screen_size=(1920, 1080),
                    valid=True,
                    reason="ok",
                )
            )
        )
        context = watch.presentation_context()

        out = resolve_presentation_action(gesture_label="POINT_RIGHT", context=context)

        self.assertEqual(out.intent.action_name, "NO_ACTION")
        self.assertIn("context_not_allowed", out.intent.reason)

    def test_presentation_mode_live_execution_routes_through_executor(self):
        watch = WindowWatch(
            backend=StubForegroundWindowBackend(
                ForegroundWindowSnapshot(
                    process_name="msedge.exe",
                    window_title="Deck - Google Slides",
                    window_rect=(0, 0, 1919, 1079),
                    screen_size=(1920, 1080),
                    valid=True,
                    reason="ok",
                )
            )
        )
        context = watch.presentation_context()
        out = resolve_presentation_action(gesture_label="POINT_RIGHT", context=context)
        safety = ExecutionSafetyGate().evaluate_presentation(
            suite_out=_suite_out(eligible="POINT_RIGHT"),
            presentation_out=out,
            hand_present=True,
        )
        keyboard = NoOpKeyboardBackend()
        executor = OSActionExecutor(
            cfg=ExecutionConfig(
                profile="live",
                enable_live_os=True,
                enable_live_presentation=True,
            ),
            mouse_backend=NoOpMouseBackend(),
            keyboard_backend=keyboard,
        )

        report = executor.apply_presentation_mode(out, allow=safety.allow, suppress_reason=safety.reason)

        self.assertTrue(report.performed)
        self.assertEqual(report.reason, "presentation_right_emitted")
        self.assertEqual(keyboard.keys, ["RIGHT"])

    def test_presentation_mode_is_suppressed_on_unstable_prediction(self):
        watch = WindowWatch(
            backend=StubForegroundWindowBackend(
                ForegroundWindowSnapshot(
                    process_name="msedge.exe",
                    window_title="Deck - Google Slides",
                    window_rect=(0, 0, 1919, 1079),
                    screen_size=(1920, 1080),
                    valid=True,
                    reason="ok",
                )
            )
        )
        context = watch.presentation_context()
        out = resolve_presentation_action(gesture_label="POINT_RIGHT", context=context)
        safety = ExecutionSafetyGate().evaluate_presentation(
            suite_out=_suite_out(eligible="POINT_RIGHT", feature_reason="unstable"),
            presentation_out=out,
            hand_present=True,
        )
        keyboard = NoOpKeyboardBackend()
        executor = OSActionExecutor(
            cfg=ExecutionConfig(
                profile="live",
                enable_live_os=True,
                enable_live_presentation=True,
            ),
            mouse_backend=NoOpMouseBackend(),
            keyboard_backend=keyboard,
        )

        report = executor.apply_presentation_mode(out, allow=safety.allow, suppress_reason=safety.reason)

        self.assertFalse(report.performed)
        self.assertEqual(report.reason, "suppressed_unstable_prediction")
        self.assertEqual(keyboard.keys, [])

    def test_overlay_helpers_show_presentation_context_and_execution_policy(self):
        watch = WindowWatch(
            backend=StubForegroundWindowBackend(
                ForegroundWindowSnapshot(
                    process_name="powerpnt.exe",
                    window_title="Quarterly Review - PowerPoint",
                    window_rect=(50, 50, 1050, 750),
                    screen_size=(1920, 1080),
                    valid=True,
                    reason="ok",
                )
            )
        )
        context = watch.presentation_context()
        executor = OSActionExecutor(
            cfg=ExecutionConfig(
                profile="live",
                enable_live_os=True,
                enable_live_presentation=True,
            ),
            mouse_backend=NoOpMouseBackend(),
            keyboard_backend=NoOpKeyboardBackend(),
        )

        ctx_text = _format_presentation_context(context)
        policy_text = _format_execution_policy_status(
            type("Ctx", (), {"executor": executor})(),
            "ok",
        )

        self.assertIn("PCTX:POWERPOINT:powerpnt.exe", ctx_text)
        self.assertIn("XPOL:LIVE:prs", policy_text)


if __name__ == "__main__":
    unittest.main()
