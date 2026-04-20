from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.config import ExecutionConfig, OperatorLifecycleConfig, OperatorOverrideConfig
from app.constants import AppState
from app.control.execution import OSActionExecutor
from app.control.keyboard import NoOpKeyboardBackend
from app.control.mouse import NoOpMouseBackend
from app.control.window_watch import PresentationAppKind, PresentationContext
from app.lifecycle.operator_lifecycle import OperatorLifecycleController, neutralize_runtime_ownership
from app.lifecycle.operator_policy import resolve_operator_override_policy


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


def _suite_out(*, down: str | None, hold_frames: int = 2) -> _SuiteOut:
    return _SuiteOut(
        chosen=down,
        stable=down,
        eligible=down,
        candidates={down} if down is not None else set(),
        reason="test",
        down=down,
        up=None,
        source="rules",
        confidence=None,
        rule_chosen=down,
        ml_chosen=None,
        ml_reason="none",
        feature_reason="ok",
        hold_frames=hold_frames,
        gate_reason="stable_match",
    )


def _general_out(*, scroll_owns: bool = False, primary_owns: bool = False, secondary_owns: bool = False, clutch_owns: bool = False):
    def _component(owns_state: bool, action_name: str = "NO_ACTION"):
        return type(
            "Component",
            (),
            {
                "owns_state": owns_state,
                "intent": type("Intent", (), {"action_name": action_name})(),
            },
        )()

    return type(
        "GeneralOut",
        (),
        {
            "clutch": _component(clutch_owns),
            "scroll": _component(scroll_owns, "SCROLL_VERTICAL" if scroll_owns else "NO_ACTION"),
            "primary": _component(primary_owns, "PRIMARY_DRAG_HOLD" if primary_owns else "NO_ACTION"),
            "secondary": _component(secondary_owns, "SECONDARY_RIGHT_CLICK" if secondary_owns else "NO_ACTION"),
        },
    )()


class _ResetSpy:
    def __init__(self):
        self.calls = 0

    def reset(self) -> None:
        self.calls += 1


class Phase6OperatorLifecycleTests(unittest.TestCase):
    def test_manual_exit_request_is_explicit_and_auditable(self):
        controller = OperatorLifecycleController(
            cfg=OperatorLifecycleConfig(
                manual_exit_keys=("ESC", "Q"),
                enable_gesture_exit=True,
                gesture_exit_label="THUMBS_DOWN",
                gesture_exit_min_hold_frames=2,
            )
        )

        request = controller.request_from_key(27)

        self.assertIsNotNone(request)
        self.assertEqual(request.source, "manual")
        self.assertEqual(request.reason, "manual_exit_key")
        self.assertEqual(request.trigger, "ESC")
        self.assertIn("manual:ESC", controller.status_text(request=request))

    def test_gesture_exit_uses_eligible_edge_only_in_active_modes(self):
        controller = OperatorLifecycleController(
            cfg=OperatorLifecycleConfig(
                enable_gesture_exit=True,
                gesture_exit_label="THUMBS_DOWN",
                gesture_exit_min_hold_frames=2,
            )
        )

        request = controller.request_from_suite_out(
            suite_out=_suite_out(down="THUMBS_DOWN", hold_frames=2),
            router_state=AppState.ACTIVE_GENERAL,
            general_out=_general_out(),
        )

        self.assertIsNotNone(request)
        self.assertEqual(request.source, "gesture")
        self.assertEqual(request.trigger, "THUMBS_DOWN")
        self.assertEqual(request.effect, "exit_app")
        presentation_request = controller.request_from_suite_out(
            suite_out=_suite_out(down="THUMBS_DOWN", hold_frames=2),
            router_state=AppState.ACTIVE_PRESENTATION,
        )
        self.assertIsNotNone(presentation_request)
        self.assertEqual(presentation_request.effect, "exit_presentation")
        self.assertIsNone(
            controller.request_from_suite_out(
                suite_out=_suite_out(down="THUMBS_DOWN", hold_frames=2),
                router_state=AppState.AUTHENTICATING,
            )
        )

    def test_presentation_gesture_exit_requires_release_before_general_exit_can_rearm(self):
        controller = OperatorLifecycleController(
            cfg=OperatorLifecycleConfig(
                enable_gesture_exit=True,
                gesture_exit_label="THUMBS_DOWN",
                gesture_exit_min_hold_frames=2,
            )
        )

        presentation_request = controller.request_from_suite_out(
            suite_out=_suite_out(down="THUMBS_DOWN", hold_frames=2),
            router_state=AppState.ACTIVE_PRESENTATION,
        )
        suppressed_general = controller.request_from_suite_out(
            suite_out=_suite_out(down="THUMBS_DOWN", hold_frames=2),
            router_state=AppState.ACTIVE_GENERAL,
            general_out=_general_out(),
        )
        controller.request_from_suite_out(
            suite_out=_suite_out(down=None, hold_frames=0),
            router_state=AppState.ACTIVE_GENERAL,
            general_out=_general_out(),
        )
        rearmed_general = controller.request_from_suite_out(
            suite_out=_suite_out(down="THUMBS_DOWN", hold_frames=2),
            router_state=AppState.ACTIVE_GENERAL,
            general_out=_general_out(),
        )

        self.assertIsNotNone(presentation_request)
        self.assertEqual(presentation_request.effect, "exit_presentation")
        self.assertIsNone(suppressed_general)
        self.assertIsNotNone(rearmed_general)
        self.assertEqual(rearmed_general.effect, "exit_app")

    def test_general_gesture_exit_is_suppressed_while_scroll_or_other_owners_are_active(self):
        controller = OperatorLifecycleController(
            cfg=OperatorLifecycleConfig(
                enable_gesture_exit=True,
                gesture_exit_label="THUMBS_DOWN",
                gesture_exit_min_hold_frames=2,
            )
        )

        self.assertIsNone(
            controller.request_from_suite_out(
                suite_out=_suite_out(down="THUMBS_DOWN", hold_frames=2),
                router_state=AppState.ACTIVE_GENERAL,
                general_out=_general_out(scroll_owns=True),
            )
        )
        self.assertIsNone(
            controller.request_from_suite_out(
                suite_out=_suite_out(down="THUMBS_DOWN", hold_frames=2),
                router_state=AppState.ACTIVE_GENERAL,
                general_out=_general_out(primary_owns=True),
            )
        )

    def test_neutralize_runtime_releases_drag_and_resets_owned_state(self):
        mouse = NoOpMouseBackend()
        keyboard = NoOpKeyboardBackend()
        executor = OSActionExecutor(
            cfg=ExecutionConfig(
                profile="live",
                enable_live_os=True,
                enable_live_primary=True,
            ),
            mouse_backend=mouse,
            keyboard_backend=keyboard,
        )
        executor._primary_drag_down = True
        ctx = type(
            "Ctx",
            (),
            {
                "auth_suite": _ResetSpy(),
                "ops_suite": _ResetSpy(),
                "clutch": _ResetSpy(),
                "scroll_mode": _ResetSpy(),
                "primary_interaction": _ResetSpy(),
                "secondary_interaction": _ResetSpy(),
                "cursor_preview": _ResetSpy(),
                "execution_safety": _ResetSpy(),
                "smoother": _ResetSpy(),
                "executor": executor,
            },
        )()

        report = neutralize_runtime_ownership(ctx, reason="manual_exit_key")

        self.assertEqual(mouse.left_ups, 1)
        self.assertEqual(keyboard.release_all_calls, 1)
        self.assertTrue(report.released_primary_drag)
        self.assertEqual(report.summary(), "drag_up,key_release,owners_reset,safety_reset")
        self.assertEqual(ctx.primary_interaction.calls, 1)
        self.assertEqual(ctx.execution_safety.calls, 1)

    def test_override_policy_force_dry_run_disables_live_execution(self):
        policy = resolve_operator_override_policy(
            override_cfg=OperatorOverrideConfig(execution_override="dry_run"),
            execution_cfg=ExecutionConfig(
                profile="live",
                enable_live_os=True,
                enable_live_cursor=True,
                enable_live_primary=True,
            ),
        )

        self.assertTrue(policy.valid)
        self.assertEqual(policy.effective_execution.profile, "dry_run")
        self.assertFalse(policy.effective_execution.enable_live_os)
        self.assertIn("EXEC:DRY_RUN", policy.status_text())

    def test_override_policy_force_live_does_not_bypass_master_enable(self):
        policy = resolve_operator_override_policy(
            override_cfg=OperatorOverrideConfig(execution_override="live"),
            execution_cfg=ExecutionConfig(
                profile="dry_run",
                enable_live_os=False,
                enable_live_cursor=True,
            ),
        )
        executor = OSActionExecutor(
            cfg=policy.effective_execution,
            mouse_backend=NoOpMouseBackend(),
            keyboard_backend=NoOpKeyboardBackend(),
        )

        self.assertEqual(policy.effective_execution.profile, "live")
        self.assertFalse(executor.policy.live_master_enabled)
        self.assertEqual(executor.policy.reason, "live_profile_missing_master_enable")

    def test_override_policy_cursor_test_enables_only_live_cursor(self):
        policy = resolve_operator_override_policy(
            override_cfg=OperatorOverrideConfig(execution_override="cursor_test"),
            execution_cfg=ExecutionConfig(profile="dry_run"),
        )
        executor = OSActionExecutor(
            cfg=policy.effective_execution,
            mouse_backend=NoOpMouseBackend(),
            keyboard_backend=NoOpKeyboardBackend(),
        )

        self.assertTrue(policy.valid)
        self.assertEqual(policy.effective_execution.profile, "live")
        self.assertTrue(policy.effective_execution.enable_live_os)
        self.assertTrue(policy.effective_execution.enable_live_cursor)
        self.assertFalse(policy.effective_execution.enable_live_primary)
        self.assertFalse(policy.effective_execution.enable_live_secondary)
        self.assertFalse(policy.effective_execution.enable_live_scroll)
        self.assertFalse(policy.effective_execution.enable_live_presentation)
        self.assertTrue(executor.policy.live_master_enabled)
        self.assertTrue(executor.policy.cursor_enabled)
        self.assertFalse(executor.policy.primary_enabled)
        self.assertEqual(executor.policy.status_text(), "LIVE:cur")

    def test_override_policy_fallback_live_enables_full_rule_runtime(self):
        policy = resolve_operator_override_policy(
            override_cfg=OperatorOverrideConfig(execution_override="fallback_live"),
            execution_cfg=ExecutionConfig(profile="dry_run"),
        )
        executor = OSActionExecutor(
            cfg=policy.effective_execution,
            mouse_backend=NoOpMouseBackend(),
            keyboard_backend=NoOpKeyboardBackend(),
        )

        self.assertTrue(policy.valid)
        self.assertEqual(policy.effective_execution.profile, "live")
        self.assertTrue(policy.effective_execution.enable_live_os)
        self.assertTrue(policy.effective_execution.enable_live_cursor)
        self.assertTrue(policy.effective_execution.enable_live_primary)
        self.assertTrue(policy.effective_execution.enable_live_secondary)
        self.assertTrue(policy.effective_execution.enable_live_scroll)
        self.assertTrue(policy.effective_execution.enable_live_presentation)
        self.assertEqual(executor.policy.status_text(), "LIVE:cur,pri,sec,scr,prs")

    def test_default_override_policy_brings_up_fallback_live_runtime(self):
        policy = resolve_operator_override_policy()

        self.assertTrue(policy.valid)
        self.assertEqual(policy.execution_override, "fallback_live")
        self.assertEqual(policy.effective_execution.profile, "live")
        self.assertTrue(policy.effective_execution.enable_live_cursor)
        self.assertTrue(policy.effective_execution.enable_live_primary)
        self.assertTrue(policy.effective_execution.enable_live_secondary)
        self.assertTrue(policy.effective_execution.enable_live_scroll)
        self.assertTrue(policy.effective_execution.enable_live_presentation)

    def test_force_presentation_override_still_requires_safe_context(self):
        policy = resolve_operator_override_policy(
            override_cfg=OperatorOverrideConfig(routing_override="force_presentation")
        )
        context = PresentationContext(
            allowed=False,
            confident=False,
            kind=PresentationAppKind.UNSUPPORTED,
            process_name="notepad.exe",
            window_title="notes",
            window_rect=(0, 0, 800, 600),
            screen_size=(1920, 1080),
            fullscreen_like=False,
            navigation_allowed=False,
            supports_start=False,
            supports_exit=False,
            reason="unsupported_foreground_app",
        )

        route = policy.route_presentation(context)

        self.assertFalse(route.presentation_allowed)
        self.assertIn("blocked", route.reason)

    def test_force_general_override_keeps_router_on_general_path(self):
        policy = resolve_operator_override_policy(
            override_cfg=OperatorOverrideConfig(routing_override="force_general")
        )
        context = PresentationContext(
            allowed=True,
            confident=True,
            kind=PresentationAppKind.POWERPOINT,
            process_name="powerpnt.exe",
            window_title="Deck - PowerPoint",
            window_rect=(0, 0, 1919, 1079),
            screen_size=(1920, 1080),
            fullscreen_like=True,
            navigation_allowed=True,
            supports_start=True,
            supports_exit=True,
            reason="powerpoint_foreground",
        )

        route = policy.route_presentation(context)

        self.assertFalse(route.presentation_allowed)
        self.assertEqual(route.reason, "forced_general")

    def test_invalid_override_fails_safe(self):
        policy = resolve_operator_override_policy(
            override_cfg=OperatorOverrideConfig(
                execution_override="unsafe",
                routing_override="sideways",
            ),
            execution_cfg=ExecutionConfig(
                profile="live",
                enable_live_os=True,
                enable_live_cursor=True,
            ),
        )

        self.assertFalse(policy.valid)
        self.assertEqual(policy.execution_override, "dry_run")
        self.assertEqual(policy.routing_override, "auto")
        self.assertFalse(policy.effective_execution.enable_live_os)
        self.assertIn("FAILSAFE:invalid_execution_override,invalid_routing_override", policy.status_text())


if __name__ == "__main__":
    unittest.main()
