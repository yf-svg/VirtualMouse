from __future__ import annotations

import unittest

from app.security.auth import GestureAuthCfg, GestureAuthOut
from app.ui.auth_overlay_state import AuthOverlayStateStore, build_auth_overlay_state


def _auth_out(
    *,
    status: str,
    matched_steps: int,
    expected_next: str | None,
    retry_after_s: float | None = None,
) -> GestureAuthOut:
    return GestureAuthOut(
        authenticated=status == "success",
        status=status,
        matched_steps=matched_steps,
        total_steps=6,
        expected_next=expected_next,
        consumed_label=None,
        failed_attempts=0,
        max_failures=5,
        retry_after_s=retry_after_s,
    )


class AuthOverlayStateTests(unittest.TestCase):
    def test_progress_state_shows_accepted_digits_and_waiting_status(self):
        state = build_auth_overlay_state(
            cfg=GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE")),
            auth_out=_auth_out(status="progress", matched_steps=3, expected_next="FOUR"),
            auth_status="progress",
            detected_gesture="THREE",
        )

        self.assertEqual(state.current_sequence, ("ONE", "TWO", "THREE"))
        self.assertEqual(state.display_digits, ("1", "2", "3"))
        self.assertEqual(state.total_required, 5)
        self.assertEqual(state.status_text, "Waiting for: FOUR")
        self.assertEqual(state.detected_gesture, "THREE")

    def test_persists_committed_sequence_across_live_gesture_changes(self):
        store = AuthOverlayStateStore()
        cfg = GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE"))

        store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="progress", matched_steps=3, expected_next="FOUR"),
            auth_status="progress",
            detected_gesture="THREE",
        )
        state = store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="progress", matched_steps=3, expected_next="FOUR"),
            auth_status="progress",
            detected_gesture="THUMBS_DOWN",
        )

        self.assertEqual(state.current_sequence, ("ONE", "TWO", "THREE"))
        self.assertEqual(state.detected_gesture, "THUMBS_DOWN")

    def test_step_back_removes_only_one_committed_step(self):
        store = AuthOverlayStateStore()
        cfg = GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE"))

        store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="progress", matched_steps=3, expected_next="FOUR"),
            auth_status="progress",
            detected_gesture="THREE",
        )
        state = store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="step_back", matched_steps=2, expected_next="THREE"),
            auth_status="step_back",
            detected_gesture="THUMBS_DOWN",
        )

        self.assertEqual(state.current_sequence, ("ONE", "TWO"))
        self.assertEqual(state.status_text, "Waiting for: THREE")

    def test_success_state_shows_full_sequence_after_bravo(self):
        store = AuthOverlayStateStore()
        cfg = GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE"))

        store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="progress", matched_steps=5, expected_next="BRAVO"),
            auth_status="progress",
            detected_gesture="FIVE",
        )
        state = store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="success", matched_steps=0, expected_next=None),
            auth_status="success",
            detected_gesture="BRAVO",
        )

        self.assertEqual(state.current_sequence, ("ONE", "TWO", "THREE", "FOUR", "FIVE"))
        self.assertEqual(state.status_text, "Approved")

    def test_locked_state_requests_sequence_freeze_and_countdown_text(self):
        store = AuthOverlayStateStore()
        cfg = GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE"))

        store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="progress", matched_steps=4, expected_next="FIVE"),
            auth_status="progress",
            detected_gesture="FOUR",
        )
        state = store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="locked_out", matched_steps=0, expected_next="ONE", retry_after_s=9.2),
            auth_status="locked_out",
            detected_gesture=None,
        )

        self.assertTrue(state.freeze_sequence)
        self.assertEqual(state.current_sequence, ("ONE", "TWO", "THREE", "FOUR"))
        self.assertEqual(state.status_text, "Locked (10s)")

    def test_reset_statuses_clear_committed_sequence_after_explicit_reset_event(self):
        store = AuthOverlayStateStore()
        cfg = GestureAuthCfg()

        store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="progress", matched_steps=3, expected_next="FOUR"),
            auth_status="progress",
            detected_gesture="THREE",
        )
        wrong = store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="reset_wrong", matched_steps=0, expected_next="ONE"),
            auth_status="reset_wrong",
            detected_gesture="BRAVO",
        )
        cleared = store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="idle", matched_steps=0, expected_next="ONE"),
            auth_status="idle",
            detected_gesture=None,
        )
        timeout = build_auth_overlay_state(
            cfg=cfg,
            auth_out=_auth_out(status="reset_timeout", matched_steps=0, expected_next="ONE"),
            auth_status="reset_timeout",
            detected_gesture=None,
        )

        self.assertEqual(wrong.status_text, "Wrong input - restarting")
        self.assertEqual(wrong.current_sequence, ("ONE", "TWO", "THREE"))
        self.assertEqual(cleared.current_sequence, ())
        self.assertEqual(timeout.status_text, "Timeout - try again")


if __name__ == "__main__":
    unittest.main()
