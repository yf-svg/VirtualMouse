from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.config import ExecutionConfig
from app.control.execution import OSActionExecutor
from app.control.execution_safety import ExecutionSafetyGate
from app.control.keyboard import NoOpKeyboardBackend
from app.control.mouse import NoOpMouseBackend
from app.control.presentation_panel import (
    default_draw_color_key,
    default_draw_pen_key,
    default_draw_size_key,
    resolve_panel_state,
)
from app.control.window_watch import (
    ForegroundWindowSnapshot,
    StubForegroundWindowBackend,
    WindowWatch,
)
from app.modes.presentation import resolve_presentation_action
from app.modes.presentation_runtime import PresentationGestureInterpreter, update_presentation_playback_signal
from app.modes.presentation_tools import PresentationToolOut, PresentationToolState


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


def _suite_out(*, eligible: str | None, feature_reason: str = "ok") -> _SuiteOut:
    return _SuiteOut(
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


def _presentation_context():
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
    return watch.presentation_context()


def _tool_out(*, state: PresentationToolState) -> PresentationToolOut:
    color_key = default_draw_color_key()
    pen_key = default_draw_pen_key()
    size_key = default_draw_size_key()
    return PresentationToolOut(
        state=state,
        intent=type("Intent", (), {"action_name": "NO_ACTION"})(),
        pointer_point=None,
        owns_presentation=state in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING},
        stroke_active=state == PresentationToolState.DRAW_STROKING,
        stroke_capturing=state == PresentationToolState.DRAW_STROKING,
        selected_color_key=color_key,
        selected_pen_key=pen_key,
        selected_size_key=size_key,
        panel_state=resolve_panel_state(
            None,
            draw_mode_active=state in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING},
            stroke_active=state == PresentationToolState.DRAW_STROKING,
            selected_color_key=color_key,
            selected_pen_key=pen_key,
            selected_size_key=size_key,
        ),
        reason="test",
    )


class PresentationRuntimeTests(unittest.TestCase):
    def test_navigation_emits_once_for_held_label_until_release(self):
        interpreter = PresentationGestureInterpreter()

        first = interpreter.update(suite_out=_suite_out(eligible="POINT_RIGHT"), hand_present=True)
        second = interpreter.update(suite_out=_suite_out(eligible="POINT_RIGHT"), hand_present=True)

        self.assertEqual(first.event_label, "POINT_RIGHT")
        self.assertEqual(first.reason, "emitted")
        self.assertIsNone(second.event_label)
        self.assertEqual(second.reason, "held_after_emit")

    def test_session_control_requires_extra_confirm_frames(self):
        interpreter = PresentationGestureInterpreter()

        first = interpreter.update(suite_out=_suite_out(eligible="OPEN_PALM"), hand_present=True)
        second = interpreter.update(suite_out=_suite_out(eligible="OPEN_PALM"), hand_present=True)

        self.assertIsNone(first.event_label)
        self.assertEqual(first.reason, "pending_confirm")
        self.assertEqual(first.threshold_frames, 2)
        self.assertEqual(second.event_label, "OPEN_PALM")
        self.assertEqual(second.reason, "emitted")

    def test_peace_sign_is_ignored_by_playback_interpreter(self):
        interpreter = PresentationGestureInterpreter()

        first = interpreter.update(suite_out=_suite_out(eligible="PEACE_SIGN"), hand_present=True)

        self.assertIsNone(first.event_label)
        self.assertEqual(first.reason, "idle")

    def test_non_playback_gesture_is_ignored(self):
        interpreter = PresentationGestureInterpreter()

        signal = interpreter.update(suite_out=_suite_out(eligible="BRAVO"), hand_present=True)

        self.assertIsNone(signal.gesture_label)
        self.assertIsNone(signal.event_label)
        self.assertEqual(signal.reason, "idle")

    def test_brief_release_gap_does_not_reemit_same_navigation_hold(self):
        interpreter = PresentationGestureInterpreter()

        interpreter.update(suite_out=_suite_out(eligible="POINT_RIGHT"), hand_present=True)
        gap = interpreter.update(suite_out=None, hand_present=False)
        held_again = interpreter.update(suite_out=_suite_out(eligible="POINT_RIGHT"), hand_present=True)
        release = interpreter.update(suite_out=None, hand_present=False)
        released = interpreter.update(suite_out=None, hand_present=False)
        rearmed = interpreter.update(suite_out=_suite_out(eligible="POINT_RIGHT"), hand_present=True)

        self.assertEqual(gap.reason, "release_grace")
        self.assertIsNone(held_again.event_label)
        self.assertEqual(held_again.reason, "held_after_emit")
        self.assertEqual(release.reason, "release_grace")
        self.assertEqual(released.reason, "released")
        self.assertEqual(rearmed.event_label, "POINT_RIGHT")

    def test_live_executor_receives_one_key_for_held_navigation_gesture(self):
        interpreter = PresentationGestureInterpreter()
        context = _presentation_context()
        safety_gate = ExecutionSafetyGate()
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

        for _ in range(3):
            signal = interpreter.update(suite_out=_suite_out(eligible="POINT_RIGHT"), hand_present=True)
            out = resolve_presentation_action(gesture_label=signal.event_label, context=context)
            safety = safety_gate.evaluate_presentation(
                suite_out=_suite_out(eligible="POINT_RIGHT"),
                presentation_out=out,
                hand_present=True,
            )
            executor.apply_presentation_mode(out, allow=safety.allow, suppress_reason=safety.reason)

        self.assertEqual(keyboard.keys, ["RIGHT"])

    def test_draw_mode_suppresses_playback_and_resets_pending_navigation(self):
        interpreter = PresentationGestureInterpreter()

        pending = interpreter.update(suite_out=_suite_out(eligible="OPEN_PALM"), hand_present=True)
        suppressed = update_presentation_playback_signal(
            interpreter=interpreter,
            suite_out=_suite_out(eligible="POINT_RIGHT"),
            hand_present=True,
            tool_out=_tool_out(state=PresentationToolState.DRAW_IDLE),
        )
        rearmed = update_presentation_playback_signal(
            interpreter=interpreter,
            suite_out=_suite_out(eligible="POINT_RIGHT"),
            hand_present=True,
            tool_out=_tool_out(state=PresentationToolState.NONE),
        )

        self.assertEqual(pending.reason, "pending_confirm")
        self.assertIsNone(suppressed.event_label)
        self.assertEqual(suppressed.reason, "tool_owned:draw_idle")
        self.assertEqual(rearmed.event_label, "POINT_RIGHT")
        self.assertEqual(rearmed.active_frames, 1)

    def test_laser_mode_does_not_suppress_playback_navigation(self):
        interpreter = PresentationGestureInterpreter()

        signal = update_presentation_playback_signal(
            interpreter=interpreter,
            suite_out=_suite_out(eligible="POINT_RIGHT"),
            hand_present=True,
            tool_out=_tool_out(state=PresentationToolState.LASER),
        )

        self.assertEqual(signal.event_label, "POINT_RIGHT")
        self.assertEqual(signal.reason, "emitted")

    def test_annotations_do_not_suppress_playback_after_tool_is_off(self):
        interpreter = PresentationGestureInterpreter()

        signal = update_presentation_playback_signal(
            interpreter=interpreter,
            suite_out=_suite_out(eligible="POINT_LEFT"),
            hand_present=True,
            tool_out=_tool_out(state=PresentationToolState.NONE),
        )

        self.assertEqual(signal.event_label, "POINT_LEFT")
        self.assertEqual(signal.reason, "emitted")


if __name__ == "__main__":
    unittest.main()
