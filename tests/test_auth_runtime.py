from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.gestures.sets.auth_set import auth_allowed_for_cfg, auth_priority_for_cfg
from app.security.auth import AuthInputState, GestureAuthCfg
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


def _auth_state(
    *,
    committed_sequence: tuple[str, ...] = (),
    max_length: int = 3,
    locked_out: bool = False,
) -> AuthInputState:
    return AuthInputState(
        committed_sequence=committed_sequence,
        max_length=max_length,
        accepting_digits=len(committed_sequence) < max_length,
        ready_to_submit=len(committed_sequence) == max_length,
        locked_out=locked_out,
    )


class AuthGestureInterpreterTests(unittest.TestCase):
    def test_peace_sign_aliases_to_two_in_auth_mode(self):
        interpreter = AuthGestureInterpreter()

        first = interpreter.update(
            suite_out=_suite_out(chosen="PEACE_SIGN", stable="PEACE_SIGN", eligible=None),
            auth_state=_auth_state(),
        )
        second = interpreter.update(
            suite_out=_suite_out(chosen="PEACE_SIGN", stable="PEACE_SIGN", eligible="PEACE_SIGN"),
            auth_state=_auth_state(),
        )
        third = interpreter.update(
            suite_out=_suite_out(chosen="PEACE_SIGN", stable="PEACE_SIGN", eligible="PEACE_SIGN"),
            auth_state=_auth_state(),
        )

        self.assertEqual(first.detected_gesture, "TWO")
        self.assertEqual(second.detected_gesture, "TWO")
        self.assertIsNone(second.event_label)
        self.assertEqual(third.event_label, "TWO")

    def test_non_auth_gesture_still_stays_hidden(self):
        interpreter = AuthGestureInterpreter()

        out = interpreter.update(
            suite_out=_suite_out(chosen="SHAKA", stable="SHAKA", eligible="SHAKA"),
            auth_state=_auth_state(),
        )

        self.assertIsNone(out.detected_gesture)
        self.assertIsNone(out.event_label)

    def test_digit_emits_after_confirmation_frames(self):
        interpreter = AuthGestureInterpreter()

        first = interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(),
        )
        second = interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(),
        )

        self.assertEqual(first.detected_gesture, "ONE")
        self.assertIsNone(first.event_label)
        self.assertEqual(second.event_label, "ONE")

    def test_next_digit_requires_release_between_commits(self):
        interpreter = AuthGestureInterpreter()

        interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(),
        )
        committed = interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(),
        )
        blocked = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            auth_state=_auth_state(committed_sequence=("ONE",)),
        )
        release_1 = interpreter.update(
            suite_out=_suite_out(chosen=None, stable=None, eligible=None),
            auth_state=_auth_state(committed_sequence=("ONE",)),
        )
        release_2 = interpreter.update(
            suite_out=_suite_out(chosen=None, stable=None, eligible=None),
            auth_state=_auth_state(committed_sequence=("ONE",)),
        )
        warm = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            auth_state=_auth_state(committed_sequence=("ONE",)),
        )
        next_digit = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            auth_state=_auth_state(committed_sequence=("ONE",)),
        )

        self.assertEqual(committed.event_label, "ONE")
        self.assertIsNone(blocked.event_label)
        self.assertIsNone(release_1.event_label)
        self.assertIsNone(release_2.event_label)
        self.assertIsNone(warm.event_label)
        self.assertEqual(next_digit.event_label, "TWO")

    def test_same_held_digit_only_emits_once_until_released(self):
        interpreter = AuthGestureInterpreter()

        interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(),
        )
        first = interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(),
        )
        held = interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(committed_sequence=("ONE",)),
        )

        self.assertEqual(first.event_label, "ONE")
        self.assertIsNone(held.event_label)

    def test_back_requires_confirmation_and_emits_without_digit_release(self):
        interpreter = AuthGestureInterpreter()

        first = interpreter.update(
            suite_out=_suite_out(chosen="THUMBS_DOWN", stable="THUMBS_DOWN", eligible="THUMBS_DOWN"),
            auth_state=_auth_state(committed_sequence=("ONE", "TWO")),
        )
        second = interpreter.update(
            suite_out=_suite_out(chosen="THUMBS_DOWN", stable="THUMBS_DOWN", eligible="THUMBS_DOWN"),
            auth_state=_auth_state(committed_sequence=("ONE", "TWO")),
        )

        self.assertEqual(first.detected_gesture, "THUMBS_DOWN")
        self.assertIsNone(first.event_label)
        self.assertEqual(second.event_label, "THUMBS_DOWN")

    def test_bravo_is_ignored_until_buffer_is_full(self):
        interpreter = AuthGestureInterpreter()

        out = interpreter.update(
            suite_out=_suite_out(chosen="BRAVO", stable="BRAVO", eligible="BRAVO"),
            auth_state=_auth_state(committed_sequence=("ONE",), max_length=3),
        )

        self.assertEqual(out.detected_gesture, "BRAVO")
        self.assertIsNone(out.event_label)

    def test_buffer_full_blocks_extra_digits_and_allows_bravo(self):
        interpreter = AuthGestureInterpreter()
        full_state = _auth_state(committed_sequence=("ONE", "TWO", "THREE"), max_length=3)

        ignored = interpreter.update(
            suite_out=_suite_out(chosen="THREE", stable="THREE", eligible="THREE"),
            auth_state=full_state,
        )
        interpreter.update(
            suite_out=_suite_out(chosen="BRAVO", stable="BRAVO", eligible="BRAVO"),
            auth_state=full_state,
        )
        bravo = interpreter.update(
            suite_out=_suite_out(chosen="BRAVO", stable="BRAVO", eligible="BRAVO"),
            auth_state=full_state,
        )

        self.assertIsNone(ignored.event_label)
        self.assertEqual(bravo.event_label, "BRAVO")

    def test_open_palm_aliases_to_five_when_five_is_in_auth_sequence(self):
        cfg = GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE"))
        interpreter = AuthGestureInterpreter(auth_cfg=cfg)
        state = _auth_state(committed_sequence=("ONE", "TWO", "THREE", "FOUR"), max_length=5)

        first = interpreter.update(
            suite_out=_suite_out(chosen="OPEN_PALM", stable="OPEN_PALM", eligible="OPEN_PALM"),
            auth_state=state,
        )
        second = interpreter.update(
            suite_out=_suite_out(chosen="OPEN_PALM", stable="OPEN_PALM", eligible="OPEN_PALM"),
            auth_state=state,
        )

        self.assertEqual(first.detected_gesture, "FIVE")
        self.assertIsNone(first.event_label)
        self.assertEqual(second.event_label, "FIVE")

    def test_brief_missing_hand_does_not_rearm_next_digit(self):
        interpreter = AuthGestureInterpreter()

        interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(),
            hand_present=True,
        )
        committed = interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(),
            hand_present=True,
        )
        lost = interpreter.update(
            suite_out=None,
            auth_state=_auth_state(committed_sequence=("ONE",)),
            hand_present=False,
        )
        reacquired = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            auth_state=_auth_state(committed_sequence=("ONE",)),
            hand_present=True,
        )

        self.assertEqual(committed.event_label, "ONE")
        self.assertIsNone(lost.event_label)
        self.assertIsNone(reacquired.event_label)

    def test_long_missing_hand_releases_digit_latch(self):
        interpreter = AuthGestureInterpreter()

        interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(),
            hand_present=True,
        )
        interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(),
            hand_present=True,
        )
        interpreter.update(suite_out=None, auth_state=_auth_state(committed_sequence=("ONE",)), hand_present=False)
        interpreter.update(suite_out=None, auth_state=_auth_state(committed_sequence=("ONE",)), hand_present=False)
        interpreter.update(suite_out=None, auth_state=_auth_state(committed_sequence=("ONE",)), hand_present=False)
        warm = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            auth_state=_auth_state(committed_sequence=("ONE",)),
            hand_present=True,
        )
        confirmed = interpreter.update(
            suite_out=_suite_out(chosen="TWO", stable="TWO", eligible="TWO"),
            auth_state=_auth_state(committed_sequence=("ONE",)),
            hand_present=True,
        )

        self.assertIsNone(warm.event_label)
        self.assertEqual(confirmed.event_label, "TWO")

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
            auth_state=_auth_state(max_length=2),
        )
        interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(max_length=2),
        )
        digit = interpreter.update(
            suite_out=_suite_out(chosen="ONE", stable="ONE", eligible="ONE"),
            auth_state=_auth_state(max_length=2),
        )
        back_first = interpreter.update(
            suite_out=_suite_out(chosen="POINT_LEFT", stable="POINT_LEFT", eligible="POINT_LEFT"),
            auth_state=_auth_state(committed_sequence=("ONE",), max_length=2),
        )
        back_second = interpreter.update(
            suite_out=_suite_out(chosen="POINT_LEFT", stable="POINT_LEFT", eligible="POINT_LEFT"),
            auth_state=_auth_state(committed_sequence=("ONE",), max_length=2),
        )

        self.assertIsNone(non_auth.detected_gesture)
        self.assertIsNone(non_auth.event_label)
        self.assertEqual(digit.event_label, "ONE")
        self.assertIsNone(back_first.event_label)
        self.assertEqual(back_second.event_label, "POINT_LEFT")

    def test_auth_set_helpers_include_mode_alias_labels(self):
        cfg = GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE"))

        allowed = auth_allowed_for_cfg(cfg)
        priority = auth_priority_for_cfg(cfg)

        self.assertIn("PEACE_SIGN", allowed)
        self.assertIn("OPEN_PALM", allowed)
        self.assertLess(priority.index("FIVE"), priority.index("OPEN_PALM"))


if __name__ == "__main__":
    unittest.main()
