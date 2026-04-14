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
        )

        self.assertIsNotNone(request)
        self.assertEqual(request.source, "gesture")
        self.assertEqual(request.trigger, "THUMBS_DOWN")
        self.assertIsNone(
            controller.request_from_suite_out(
                suite_out=_suite_out(down="THUMBS_DOWN", hold_frames=2),
                router_state=AppState.AUTHENTICATING,
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
