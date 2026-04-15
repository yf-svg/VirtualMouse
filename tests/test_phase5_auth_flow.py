from __future__ import annotations

import unittest

from app.gestures.sets.auth_set import AUTH_ALLOWED
from app.security.auth import GestureAuth, GestureAuthCfg


class Phase5AuthFlowTests(unittest.TestCase):
    def test_auth_behaves_like_latched_keypad_entry(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE")))

        first = auth.update("ONE", now=1.0)
        pause_1 = auth.update(None, now=5.0)
        second = auth.update("TWO", now=6.0)
        pause_2 = auth.update(None, now=10.0)
        third = auth.update("THREE", now=11.0)
        approved = auth.update("BRAVO", now=12.0)

        self.assertEqual(first.status, "started")
        self.assertEqual(first.committed_sequence, ("ONE",))
        self.assertEqual(pause_1.committed_sequence, ("ONE",))
        self.assertEqual(second.committed_sequence, ("ONE", "TWO"))
        self.assertEqual(pause_2.committed_sequence, ("ONE", "TWO"))
        self.assertEqual(third.status, "ready_to_submit")
        self.assertEqual(third.committed_sequence, ("ONE", "TWO", "THREE"))
        self.assertEqual(approved.status, "success")
        self.assertTrue(approved.authenticated)
        self.assertEqual(approved.committed_sequence, ("ONE", "TWO", "THREE"))

    def test_auth_back_removes_only_last_committed_digit(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE")))
        auth.update("ONE", now=1.0)
        auth.update("TWO", now=2.0)
        auth.update("THREE", now=3.0)

        out = auth.update("THUMBS_DOWN", now=4.0)

        self.assertEqual(out.status, "step_back")
        self.assertEqual(out.committed_sequence, ("ONE", "TWO"))
        self.assertEqual(out.expected_next, "THREE")

    def test_auth_reset_clears_committed_buffer(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE")))
        auth.update("ONE", now=1.0)
        auth.update("TWO", now=2.0)

        out = auth.update("FIST", now=3.0)

        self.assertEqual(out.status, "reset_cancel")
        self.assertEqual(out.committed_sequence, ())
        self.assertEqual(out.expected_next, "ONE")

    def test_auth_bravo_before_buffer_full_is_ignored(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE")))
        auth.update("ONE", now=1.0)

        out = auth.update("BRAVO", now=2.0)

        self.assertEqual(out.status, "progress")
        self.assertEqual(out.committed_sequence, ("ONE",))
        self.assertEqual(out.expected_next, "TWO")

    def test_auth_ok_validates_committed_buffer_not_live_progression(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE")))
        auth.update("ONE", now=1.0)
        auth.update("THREE", now=2.0)
        auth.update("TWO", now=3.0)

        out = auth.update("BRAVO", now=4.0)

        self.assertEqual(out.status, "reset_wrong")
        self.assertEqual(out.committed_sequence, ())
        self.assertEqual(out.failed_attempts, 1)

    def test_auth_buffer_full_ignores_extra_digits_until_submit_or_edit(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE")))
        auth.update("ONE", now=1.0)
        auth.update("TWO", now=2.0)
        auth.update("THREE", now=3.0)

        digit = auth.update("THREE", now=4.0)

        self.assertEqual(digit.status, "ready_to_submit")
        self.assertEqual(digit.committed_sequence, ("ONE", "TWO", "THREE"))
        self.assertTrue(digit.buffer_full)
        self.assertEqual(digit.expected_next, "BRAVO")

    def test_auth_pause_does_not_timeout_or_clear_buffer(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), step_timeout_s=0.5))
        auth.update("ONE", now=1.0)

        out = auth.update(None, now=99.0)

        self.assertEqual(out.status, "progress")
        self.assertEqual(out.committed_sequence, ("ONE",))
        self.assertEqual(out.failed_attempts, 0)

    def test_auth_enters_lockout_after_wrong_submit_reaches_limit(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), max_failures=1, cooldown_s=3.0))
        auth.update("TWO", now=1.0)
        auth.update("ONE", now=2.0)

        out = auth.update("BRAVO", now=3.0)

        self.assertEqual(out.status, "locked_out")
        self.assertEqual(out.committed_sequence, ())
        self.assertEqual(out.failed_attempts, 1)
        self.assertGreater(out.retry_after_s or 0.0, 0.0)

    def test_auth_recovers_after_lockout_window(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), max_failures=1, cooldown_s=3.0))
        auth.update("TWO", now=1.0)
        auth.update("ONE", now=2.0)
        auth.update("BRAVO", now=3.0)

        during = auth.update("ONE", now=4.0)
        after = auth.update("ONE", now=7.0)

        self.assertEqual(during.status, "locked_out")
        self.assertEqual(after.status, "started")
        self.assertEqual(after.committed_sequence, ("ONE",))
        self.assertEqual(after.failed_attempts, 0)

    def test_auth_set_is_restricted_to_explicit_runtime_auth_gestures(self):
        self.assertEqual(
            AUTH_ALLOWED,
            {"FIST", "BRAVO", "THUMBS_DOWN", "ONE", "TWO", "THREE", "PEACE_SIGN"},
        )


if __name__ == "__main__":
    unittest.main()
