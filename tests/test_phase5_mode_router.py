from __future__ import annotations

import unittest

from app.constants import AppState
from app.modes.router import ModeRouter
from app.security.auth import GestureAuth, GestureAuthCfg


class Phase5ModeRouterTests(unittest.TestCase):
    def test_router_moves_from_locked_to_authenticating_on_first_step(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO", "THREE"))))

        self.assertEqual(router.current_auth_expected_next, "ONE")
        out = router.route_auth_edge("ONE", now=1.0)

        self.assertEqual(out.state, AppState.AUTHENTICATING)
        self.assertEqual(out.suite_key, "auth")
        self.assertEqual(out.auth_status, "started")
        self.assertEqual(router.current_auth_expected_next, "TWO")

    def test_router_unlocks_to_active_general_on_sequence_success(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))

        router.route_auth_edge("ONE", now=1.0)
        router.route_auth_edge("TWO", now=2.0)
        out = router.route_auth_edge("BRAVO", now=3.0)

        self.assertEqual(out.state, AppState.ACTIVE_GENERAL)
        self.assertEqual(out.suite_key, "ops")
        self.assertEqual(out.auth_status, "success")
        self.assertEqual(out.auth_progress_text, "Auth complete")

    def test_router_stays_authenticating_on_timeout_reset(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), step_timeout_s=2.0)))

        router.route_auth_edge("ONE", now=1.0)
        out = router.route_auth_edge(None, now=4.5)

        self.assertEqual(out.state, AppState.AUTHENTICATING)
        self.assertEqual(out.auth_status, "reset_timeout")
        self.assertEqual(out.suite_key, "auth")

    def test_router_stays_authenticating_on_wrong_gesture_reset(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))

        router.route_auth_edge("ONE", now=1.0)
        out = router.route_auth_edge("FIVE", now=2.0)

        self.assertEqual(out.state, AppState.AUTHENTICATING)
        self.assertEqual(out.auth_status, "reset_wrong")
        self.assertEqual(out.suite_key, "auth")

    def test_router_stays_authenticating_on_explicit_reset(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))

        router.route_auth_edge("ONE", now=1.0)
        out = router.route_auth_edge("FIST", now=2.0)

        self.assertEqual(out.state, AppState.AUTHENTICATING)
        self.assertEqual(out.auth_status, "reset_cancel")
        self.assertEqual(out.suite_key, "auth")

    def test_router_ignores_bravo_before_sequence_complete(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))

        router.route_auth_edge("ONE", now=1.0)
        out = router.route_auth_edge("BRAVO", now=2.0)

        self.assertEqual(out.state, AppState.AUTHENTICATING)
        self.assertEqual(out.auth_status, "progress")
        self.assertEqual(out.auth_out.matched_steps, 1)

    def test_router_allows_reset_from_bravo_wait_state(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))

        router.route_auth_edge("ONE", now=1.0)
        router.route_auth_edge("TWO", now=2.0)
        out = router.route_auth_edge("FIST", now=3.0)

        self.assertEqual(out.state, AppState.AUTHENTICATING)
        self.assertEqual(out.auth_status, "reset_cancel")
        self.assertEqual(out.auth_out.matched_steps, 0)

    def test_router_lock_resets_back_to_idle_locked(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))
        router.route_auth_edge("ONE", now=1.0)
        router.route_auth_edge("TWO", now=2.0)
        router.route_auth_edge("BRAVO", now=3.0)

        out = router.lock()

        self.assertEqual(out.state, AppState.IDLE_LOCKED)
        self.assertEqual(out.auth_status, "idle")
        self.assertEqual(out.suite_key, "auth")

    def test_router_returns_locked_out_status_after_retry_limit(self):
        router = ModeRouter(
            auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"), max_failures=1, cooldown_s=4.0))
        )

        out = router.route_auth_edge("FIVE", now=1.0)

        self.assertEqual(out.state, AppState.IDLE_LOCKED)
        self.assertEqual(out.auth_status, "locked_out")
        self.assertIn("locked", out.auth_progress_text.lower())

    def test_router_can_put_active_general_into_sleep(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))
        router.route_auth_edge("ONE", now=1.0)
        router.route_auth_edge("TWO", now=2.0)
        router.route_auth_edge("BRAVO", now=3.0)

        out = router.request_sleep()

        self.assertEqual(out.state, AppState.SLEEP)
        self.assertEqual(out.suite_key, "sleep")
        self.assertEqual(out.auth_status, "sleep")

    def test_router_wake_from_sleep_returns_to_authenticating(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))
        router.route_auth_edge("ONE", now=1.0)
        router.route_auth_edge("TWO", now=2.0)
        router.route_auth_edge("BRAVO", now=3.0)
        router.request_sleep()

        out = router.wake_for_auth()

        self.assertEqual(out.state, AppState.AUTHENTICATING)
        self.assertEqual(out.suite_key, "auth")
        self.assertEqual(out.auth_status, "idle")

    def test_router_lock_from_sleep_returns_to_idle_locked(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))
        router.route_auth_edge("ONE", now=1.0)
        router.route_auth_edge("TWO", now=2.0)
        router.route_auth_edge("BRAVO", now=3.0)
        router.request_sleep()

        out = router.lock()

        self.assertEqual(out.state, AppState.IDLE_LOCKED)
        self.assertEqual(out.suite_key, "auth")
        self.assertEqual(out.auth_status, "idle")

    def test_router_request_exit_moves_to_exiting_state(self):
        router = ModeRouter(auth=GestureAuth(GestureAuthCfg(sequence=("ONE", "TWO"))))
        router.route_auth_edge("ONE", now=1.0)
        router.route_auth_edge("TWO", now=2.0)
        router.route_auth_edge("BRAVO", now=3.0)

        out = router.request_exit(source="manual", reason="ESC")

        self.assertEqual(out.state, AppState.EXITING)
        self.assertEqual(out.suite_key, "exit")
        self.assertEqual(out.auth_status, "exiting")
        self.assertIn("manual:ESC", out.auth_progress_text)


if __name__ == "__main__":
    unittest.main()
