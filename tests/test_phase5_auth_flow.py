from __future__ import annotations

import unittest

from app.gestures.sets.auth_set import AUTH_ALLOWED
from app.security.auth import GestureAuth, GestureAuthCfg


class Phase5AuthFlowTests(unittest.TestCase):
    def test_auth_sequence_requires_explicit_bravo_approval(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"), step_timeout_s=4.0))

        first = auth.update("ONE", now=1.0)
        second = auth.update("TWO", now=2.0)
        third = auth.update("THREE", now=3.0)
        approved = auth.update("BRAVO", now=4.0)

        self.assertEqual(first.status, "started")
        self.assertEqual(second.status, "progress")
        self.assertEqual(third.status, "progress")
        self.assertEqual(third.expected_next, "BRAVO")
        self.assertEqual(third.matched_steps, 3)
        self.assertEqual(third.total_steps, 4)
        self.assertEqual(approved.status, "success")
        self.assertTrue(approved.authenticated)

    def test_auth_resets_on_wrong_gesture(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"), step_timeout_s=4.0))
        auth.update("ONE", now=1.0)

        out = auth.update("FOUR", now=2.0)

        self.assertEqual(out.status, "reset_wrong")
        self.assertEqual(out.matched_steps, 0)
        self.assertEqual(out.expected_next, "ONE")
        self.assertEqual(out.failed_attempts, 1)

    def test_auth_bravo_before_sequence_complete_is_ignored(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"), step_timeout_s=4.0))
        auth.update("ONE", now=1.0)

        out = auth.update("BRAVO", now=2.0)

        self.assertEqual(out.status, "progress")
        self.assertEqual(out.matched_steps, 1)
        self.assertEqual(out.expected_next, "TWO")

    def test_auth_ignores_number_jitter_while_waiting_for_bravo(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"), step_timeout_s=4.0))
        auth.update("ONE", now=1.0)
        auth.update("TWO", now=2.0)
        auth.update("THREE", now=3.0)

        out = auth.update("THREE", now=3.4)

        self.assertEqual(out.status, "progress")
        self.assertEqual(out.matched_steps, 3)
        self.assertEqual(out.expected_next, "BRAVO")
        self.assertEqual(out.failed_attempts, 0)

    def test_auth_full_sequence_allows_explicit_reset(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"), step_timeout_s=4.0))
        auth.update("ONE", now=1.0)
        auth.update("TWO", now=2.0)
        auth.update("THREE", now=3.0)

        fist = auth.update("FIST", now=3.4)

        self.assertEqual(fist.status, "reset_cancel")
        self.assertEqual(fist.matched_steps, 0)
        self.assertEqual(fist.expected_next, "ONE")

    def test_auth_full_sequence_ignores_extra_digits_while_waiting_for_bravo(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"), step_timeout_s=4.0))
        auth.update("ONE", now=1.0)
        auth.update("TWO", now=2.0)
        auth.update("THREE", now=3.0)

        digit = auth.update("THREE", now=3.8)

        self.assertEqual(digit.status, "progress")
        self.assertEqual(digit.matched_steps, 3)
        self.assertEqual(digit.expected_next, "BRAVO")

    def test_auth_resets_on_timeout(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), step_timeout_s=2.0))
        auth.update("ONE", now=1.0)

        out = auth.update(None, now=4.5)

        self.assertEqual(out.status, "reset_timeout")
        self.assertEqual(out.matched_steps, 0)
        self.assertEqual(out.expected_next, "ONE")
        self.assertEqual(out.failed_attempts, 1)

    def test_auth_thumbs_down_steps_back_one_gesture(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"), step_timeout_s=4.0))
        auth.update("ONE", now=1.0)
        auth.update("TWO", now=2.0)

        out = auth.update("THUMBS_DOWN", now=3.0)

        self.assertEqual(out.status, "step_back")
        self.assertEqual(out.matched_steps, 1)
        self.assertEqual(out.expected_next, "TWO")

    def test_auth_thumbs_down_from_approval_stage_returns_to_previous_number(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), step_timeout_s=4.0))
        auth.update("ONE", now=1.0)
        auth.update("TWO", now=2.0)

        out = auth.update("THUMBS_DOWN", now=3.0)

        self.assertEqual(out.status, "step_back")
        self.assertEqual(out.matched_steps, 1)
        self.assertEqual(out.expected_next, "TWO")

    def test_auth_set_is_restricted_to_explicit_runtime_auth_gestures(self):
        self.assertEqual(
            AUTH_ALLOWED,
            {"FIST", "BRAVO", "THUMBS_DOWN", "ONE", "TWO", "THREE"},
        )

    def test_auth_enters_lockout_after_max_failures(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), max_failures=5, cooldown_s=10.0))

        auth.update("FOUR", now=1.0)
        auth.update("FOUR", now=2.0)
        auth.update("FOUR", now=3.0)
        auth.update("FOUR", now=4.0)
        out = auth.update("FOUR", now=5.0)

        self.assertEqual(out.status, "locked_out")
        self.assertEqual(out.failed_attempts, 5)
        self.assertGreater(out.retry_after_s or 0.0, 0.0)

    def test_auth_recovers_after_lockout_window(self):
        auth = GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), max_failures=1, cooldown_s=3.0))

        locked = auth.update("FOUR", now=1.0)
        during = auth.update("ONE", now=2.0)
        after = auth.update("ONE", now=4.5)

        self.assertEqual(locked.status, "locked_out")
        self.assertEqual(during.status, "locked_out")
        self.assertEqual(after.status, "started")
        self.assertEqual(after.failed_attempts, 0)


if __name__ == "__main__":
    unittest.main()
