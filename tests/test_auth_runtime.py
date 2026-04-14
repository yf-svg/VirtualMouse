from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.security.auth import GestureAuthCfg
from app.security.auth_runtime import AuthGestureInterpreter


@dataclass(frozen=True)
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


def _suite_out(
    *,
    chosen: str | None = None,
    stable: str | None = None,
    eligible: str | None = None,
) -> _SuiteOut:
    return _SuiteOut(
        chosen=chosen,
        stable=stable,
        eligible=eligible,
        candidates=set(),
        reason="test",
        down=eligible,
        up=None,
        source="rules",
        confidence=None,
        rule_chosen=chosen,
        ml_chosen=None,
        ml_reason="none",
        feature_reason="ok",
        hold_frames=2,
        gate_reason="test",
    )


class AuthGestureInterpreterTests(unittest.TestCase):
    def test_detected_label_is_auth_only(self):
        interpreter = AuthGestureInterpreter()

        out = interpreter.update(
            suite_out=_suite_out(chosen="PEACE_SIGN", stable="PEACE_SIGN", eligible=None),
            expected_next="TWO",
        )

        self.assertIsNone(out.detected_gesture)
        self.assertIsNone(out.event_label)

    def test_expected_digit_emits_on_first_eligible_auth_frame(self):
        interpreter = AuthGestureInterpreter()

        first = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            expected_next="TWO",
        )
        out = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            expected_next="TWO",
        )

        self.assertEqual(first.detected_gesture, "TWO")
        self.assertIsNone(first.event_label)
        self.assertEqual(out.event_label, "TWO")

    def test_expected_next_digit_requires_release_after_previous_digit_commit(self):
        interpreter = AuthGestureInterpreter()

        interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            expected_next="TWO",
        )
        first = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            expected_next="TWO",
        )
        chained1 = interpreter.update(
            suite_out=_suite_out(chosen="THREE", stable="THREE", eligible="THREE"),
            expected_next="THREE",
        )
        chained2 = interpreter.update(
            suite_out=_suite_out(chosen="THREE", stable="THREE", eligible="THREE"),
            expected_next="THREE",
        )
        release1 = interpreter.update(
            suite_out=_suite_out(chosen=None, stable=None, eligible=None),
            expected_next="THREE",
        )
        release2 = interpreter.update(
            suite_out=_suite_out(chosen=None, stable=None, eligible=None),
            expected_next="THREE",
        )
        warm = interpreter.update(
            suite_out=_suite_out(chosen="THREE", stable="THREE", eligible="THREE"),
            expected_next="THREE",
        )
        next_digit = interpreter.update(
            suite_out=_suite_out(chosen="THREE", stable="THREE", eligible="THREE"),
            expected_next="THREE",
        )

        self.assertEqual(first.event_label, "TWO")
        self.assertIsNone(chained1.event_label)
        self.assertIsNone(chained2.event_label)
        self.assertIsNone(release1.event_label)
        self.assertIsNone(release2.event_label)
        self.assertIsNone(warm.event_label)
        self.assertEqual(next_digit.event_label, "THREE")

    def test_transient_thumbs_down_requires_extra_confirmation(self):
        interpreter = AuthGestureInterpreter()

        first = interpreter.update(
            suite_out=_suite_out(chosen="THUMBS_DOWN", stable="THUMBS_DOWN", eligible="THUMBS_DOWN"),
            expected_next="THREE",
        )
        second = interpreter.update(
            suite_out=_suite_out(chosen="THUMBS_DOWN", stable="THUMBS_DOWN", eligible="THUMBS_DOWN"),
            expected_next="THREE",
        )

        self.assertEqual(first.detected_gesture, "THUMBS_DOWN")
        self.assertIsNone(first.event_label)
        self.assertEqual(second.event_label, "THUMBS_DOWN")

    def test_wrong_digit_requires_extra_confirmation_before_reset_event(self):
        interpreter = AuthGestureInterpreter()

        first = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            expected_next="THREE",
        )
        second = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            expected_next="THREE",
        )
        third = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            expected_next="THREE",
        )

        self.assertIsNone(first.event_label)
        self.assertIsNone(second.event_label)
        self.assertEqual(third.event_label, "TWO")

    def test_waiting_for_bravo_ignores_non_control_auth_events(self):
        interpreter = AuthGestureInterpreter()

        digit = interpreter.update(
            suite_out=_suite_out(chosen="THREE", stable="THREE", eligible="THREE"),
            expected_next="BRAVO",
        )
        reset = interpreter.update(
            suite_out=_suite_out(chosen="FIST", stable="FIST", eligible="FIST"),
            expected_next="BRAVO",
        )
        reset_confirmed = interpreter.update(
            suite_out=_suite_out(chosen="FIST", stable="FIST", eligible="FIST"),
            expected_next="BRAVO",
        )

        self.assertEqual(digit.detected_gesture, "THREE")
        self.assertIsNone(digit.event_label)
        self.assertEqual(reset.detected_gesture, "FIST")
        self.assertIsNone(reset.event_label)
        self.assertEqual(reset_confirmed.event_label, "FIST")

    def test_same_held_auth_pose_only_emits_once_until_released(self):
        interpreter = AuthGestureInterpreter()

        interpreter.update(
            suite_out=_suite_out(chosen="BRAVO", stable="BRAVO", eligible="BRAVO"),
            expected_next="BRAVO",
        )
        first = interpreter.update(
            suite_out=_suite_out(chosen="BRAVO", stable="BRAVO", eligible="BRAVO"),
            expected_next="BRAVO",
        )
        held = interpreter.update(
            suite_out=_suite_out(chosen="BRAVO", stable="BRAVO", eligible="BRAVO"),
            expected_next="BRAVO",
        )
        released1 = interpreter.update(
            suite_out=_suite_out(chosen=None, stable=None, eligible=None),
            expected_next="BRAVO",
        )
        released2 = interpreter.update(
            suite_out=_suite_out(chosen=None, stable=None, eligible=None),
            expected_next="BRAVO",
        )
        reenter_warm = interpreter.update(
            suite_out=_suite_out(chosen="BRAVO", stable="BRAVO", eligible="BRAVO"),
            expected_next="BRAVO",
        )
        reenter = interpreter.update(
            suite_out=_suite_out(chosen="BRAVO", stable="BRAVO", eligible="BRAVO"),
            expected_next="BRAVO",
        )

        self.assertEqual(first.event_label, "BRAVO")
        self.assertIsNone(held.event_label)
        self.assertIsNone(released1.event_label)
        self.assertIsNone(released2.event_label)
        self.assertIsNone(reenter_warm.event_label)
        self.assertEqual(reenter.event_label, "BRAVO")

    def test_brief_missing_hand_does_not_rearm_next_digit(self):
        interpreter = AuthGestureInterpreter()

        interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            expected_next="TWO",
            hand_present=True,
        )
        first = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            expected_next="TWO",
            hand_present=True,
        )
        lost = interpreter.update(
            suite_out=None,
            expected_next="THREE",
            hand_present=False,
        )
        reacquired = interpreter.update(
            suite_out=_suite_out(chosen="THREE", stable="THREE", eligible="THREE"),
            expected_next="THREE",
            hand_present=True,
        )

        self.assertEqual(first.event_label, "TWO")
        self.assertIsNone(lost.event_label)
        self.assertIsNone(reacquired.event_label)

    def test_long_missing_hand_releases_digit_latch(self):
        interpreter = AuthGestureInterpreter()

        interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            expected_next="TWO",
            hand_present=True,
        )
        interpreter.update(
            suite_out=None,
            expected_next="THREE",
            hand_present=False,
        )
        interpreter.update(
            suite_out=None,
            expected_next="THREE",
            hand_present=False,
        )
        interpreter.update(
            suite_out=None,
            expected_next="THREE",
            hand_present=False,
        )
        next_digit = interpreter.update(
            suite_out=_suite_out(chosen="THREE", stable="THREE", eligible="THREE"),
            expected_next="THREE",
            hand_present=True,
        )
        confirmed = interpreter.update(
            suite_out=_suite_out(chosen="THREE", stable="THREE", eligible="THREE"),
            expected_next="THREE",
            hand_present=True,
        )

        self.assertIsNone(next_digit.event_label)
        self.assertEqual(confirmed.event_label, "THREE")

    def test_runtime_roles_follow_auth_cfg(self):
        cfg = GestureAuthCfg(
            sequence=("ONE", "TWO"),
            reset_gestures=("CLOSED_PALM",),
            approve_gestures=("OPEN_PALM",),
            back_gestures=("POINT_LEFT",),
        )
        interpreter = AuthGestureInterpreter(auth_cfg=cfg)

        non_auth = interpreter.update(
            suite_out=_suite_out(chosen="FIST", stable="FIST", eligible="FIST"),
            expected_next="ONE",
        )
        interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            expected_next="ONE",
        )
        expected = interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            expected_next="ONE",
        )
        back_first = interpreter.update(
            suite_out=_suite_out(chosen="POINT_LEFT", stable="POINT_LEFT", eligible="POINT_LEFT"),
            expected_next="TWO",
        )
        back_second = interpreter.update(
            suite_out=_suite_out(chosen="POINT_LEFT", stable="POINT_LEFT", eligible="POINT_LEFT"),
            expected_next="TWO",
        )

        self.assertIsNone(non_auth.detected_gesture)
        self.assertIsNone(non_auth.event_label)
        self.assertEqual(expected.event_label, "ONE")
        self.assertIsNone(back_first.event_label)
        self.assertEqual(back_second.event_label, "POINT_LEFT")


if __name__ == "__main__":
    unittest.main()
