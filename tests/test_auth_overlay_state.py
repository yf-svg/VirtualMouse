from __future__ import annotations

import unittest

from app.security.auth import GestureAuthCfg, GestureAuthOut
from app.ui.auth_overlay_state import AuthOverlayStateStore, build_auth_overlay_state


def _auth_out(
    *,
    status: str,
    committed_sequence: tuple[str, ...],
    expected_next: str | None,
    retry_after_s: float | None = None,
) -> GestureAuthOut:
    cfg = GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE"))
    return GestureAuthOut(
        authenticated=status == "success",
        status=status,
        matched_steps=len(committed_sequence),
        total_steps=len(cfg.sequence),
        expected_next=expected_next,
        consumed_label=None,
        failed_attempts=0,
        max_failures=5,
        retry_after_s=retry_after_s,
        committed_sequence=committed_sequence,
        buffer_full=len(committed_sequence) == len(cfg.sequence),
    )


class AuthOverlayStateTests(unittest.TestCase):
    def test_progress_state_shows_committed_digits_and_waiting_status(self):
        state = build_auth_overlay_state(
            cfg=GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE")),
            auth_out=_auth_out(status="progress", committed_sequence=("ONE", "TWO", "THREE"), expected_next="FOUR"),
            auth_status="progress",
            detected_gesture="THREE",
        )

        self.assertEqual(state.current_sequence, ("ONE", "TWO", "THREE"))
        self.assertEqual(state.display_digits, ("1", "2", "3"))
        self.assertEqual(state.total_required, 5)
        self.assertEqual(state.status_text, "Waiting for: FOUR")
        self.assertEqual(state.detected_gesture, "THREE")

    def test_committed_sequence_persists_when_live_detection_changes_or_disappears(self):
        store = AuthOverlayStateStore()
        cfg = GestureAuthCfg(sequence=("ONE", "TWO", "THREE"))

        store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="started", committed_sequence=("ONE",), expected_next="TWO"),
            auth_status="started",
            detected_gesture="ONE",
        )
        changed = store.build_state(
            cfg=cfg,
            auth_out=GestureAuthOut(
                authenticated=False,
                status="progress",
                matched_steps=1,
                total_steps=3,
                expected_next="TWO",
                consumed_label=None,
                failed_attempts=0,
                max_failures=5,
                retry_after_s=None,
                committed_sequence=("ONE",),
                buffer_full=False,
            ),
            auth_status="progress",
            detected_gesture="THUMBS_DOWN",
        )
        gone = store.build_state(
            cfg=cfg,
            auth_out=GestureAuthOut(
                authenticated=False,
                status="progress",
                matched_steps=1,
                total_steps=3,
                expected_next="TWO",
                consumed_label=None,
                failed_attempts=0,
                max_failures=5,
                retry_after_s=None,
                committed_sequence=("ONE",),
                buffer_full=False,
            ),
            auth_status="progress",
            detected_gesture=None,
        )

        self.assertEqual(changed.current_sequence, ("ONE",))
        self.assertEqual(gone.current_sequence, ("ONE",))
        self.assertIsNone(gone.detected_gesture)

    def test_step_back_removes_only_one_committed_step(self):
        state = build_auth_overlay_state(
            cfg=GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE")),
            auth_out=_auth_out(status="step_back", committed_sequence=("ONE", "TWO"), expected_next="THREE"),
            auth_status="step_back",
            detected_gesture="THUMBS_DOWN",
        )

        self.assertEqual(state.current_sequence, ("ONE", "TWO"))
        self.assertEqual(state.status_text, "Waiting for: THREE")

    def test_ready_to_submit_shows_committed_buffer_and_ok_prompt(self):
        state = build_auth_overlay_state(
            cfg=GestureAuthCfg(sequence=("ONE", "TWO", "THREE")),
            auth_out=GestureAuthOut(
                authenticated=False,
                status="ready_to_submit",
                matched_steps=3,
                total_steps=3,
                expected_next="BRAVO",
                consumed_label=None,
                failed_attempts=0,
                max_failures=5,
                retry_after_s=None,
                committed_sequence=("ONE", "TWO", "THREE"),
                buffer_full=True,
            ),
            auth_status="ready_to_submit",
            detected_gesture="THREE",
        )

        self.assertEqual(state.current_sequence, ("ONE", "TWO", "THREE"))
        self.assertEqual(state.status_text, "Waiting for: BRAVO")

    def test_success_state_shows_full_sequence_after_bravo(self):
        state = build_auth_overlay_state(
            cfg=GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE")),
            auth_out=_auth_out(status="success", committed_sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE"), expected_next=None),
            auth_status="success",
            detected_gesture="BRAVO",
        )

        self.assertEqual(state.current_sequence, ("ONE", "TWO", "THREE", "FOUR", "FIVE"))
        self.assertEqual(state.status_text, "Approved")

    def test_locked_state_freezes_last_visible_sequence_and_countdown_text(self):
        store = AuthOverlayStateStore()
        cfg = GestureAuthCfg(sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE"))

        store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="ready_to_submit", committed_sequence=("ONE", "TWO", "THREE", "FOUR", "FIVE"), expected_next="BRAVO"),
            auth_status="ready_to_submit",
            detected_gesture="FIVE",
        )
        state = store.build_state(
            cfg=cfg,
            auth_out=_auth_out(status="locked_out", committed_sequence=(), expected_next="ONE", retry_after_s=9.2),
            auth_status="locked_out",
            detected_gesture=None,
        )

        self.assertTrue(state.freeze_sequence)
        self.assertEqual(state.current_sequence, ("ONE", "TWO", "THREE", "FOUR", "FIVE"))
        self.assertEqual(state.status_text, "Locked (10s)")

    def test_reset_statuses_clear_committed_sequence_after_explicit_reset_event(self):
        state = build_auth_overlay_state(
            cfg=GestureAuthCfg(sequence=("ONE", "TWO", "THREE")),
            auth_out=GestureAuthOut(
                authenticated=False,
                status="reset_cancel",
                matched_steps=0,
                total_steps=3,
                expected_next="ONE",
                consumed_label="FIST",
                failed_attempts=0,
                max_failures=5,
                retry_after_s=None,
                committed_sequence=(),
                buffer_full=False,
            ),
            auth_status="reset_cancel",
            detected_gesture="FIST",
        )
        wrong = build_auth_overlay_state(
            cfg=GestureAuthCfg(sequence=("ONE", "TWO", "THREE")),
            auth_out=GestureAuthOut(
                authenticated=False,
                status="reset_wrong",
                matched_steps=0,
                total_steps=3,
                expected_next="ONE",
                consumed_label="BRAVO",
                failed_attempts=1,
                max_failures=5,
                retry_after_s=None,
                committed_sequence=(),
                buffer_full=False,
            ),
            auth_status="reset_wrong",
            detected_gesture="BRAVO",
        )

        self.assertEqual(state.current_sequence, ())
        self.assertEqual(state.status_text, "Reset")
        self.assertEqual(wrong.current_sequence, ())
        self.assertEqual(wrong.status_text, "Wrong input - restarting")


if __name__ == "__main__":
    unittest.main()
