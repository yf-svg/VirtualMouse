from __future__ import annotations

import unittest

from app.config import OperatorLifecycleConfig
from app.lifecycle.operator_lifecycle import OperatorLifecycleController
from app.lifecycle.runtime_loop import RuntimeState, _consume_exit_request, _manual_exit_armed


class Phase6RuntimeLoopStartupGuardTests(unittest.TestCase):
    def _ctx(self, *, guard_s: float = 0.75):
        controller = OperatorLifecycleController(
            cfg=OperatorLifecycleConfig(
                manual_exit_keys=("ESC", "Q"),
                startup_manual_exit_guard_s=guard_s,
            )
        )
        return type("Ctx", (), {"operator_lifecycle": controller})()

    def test_manual_exit_not_armed_before_first_frame(self):
        ctx = self._ctx(guard_s=0.75)
        state = RuntimeState(startup_t0=10.0, first_frame_presented=False)

        self.assertFalse(_manual_exit_armed(ctx, state, now=11.0))
        self.assertIsNone(_consume_exit_request(ctx, state, key=27, now=11.0))

    def test_manual_exit_not_armed_during_startup_guard_even_after_first_frame(self):
        ctx = self._ctx(guard_s=0.75)
        state = RuntimeState(startup_t0=10.0, first_frame_presented=True)

        self.assertFalse(_manual_exit_armed(ctx, state, now=10.40))
        self.assertIsNone(_consume_exit_request(ctx, state, key=27, now=10.40))

    def test_manual_exit_arms_after_first_frame_and_guard_delay(self):
        ctx = self._ctx(guard_s=0.75)
        state = RuntimeState(startup_t0=10.0, first_frame_presented=True)

        self.assertTrue(_manual_exit_armed(ctx, state, now=10.80))
        request = _consume_exit_request(ctx, state, key=27, now=10.80)
        self.assertIsNotNone(request)
        self.assertEqual(request.source, "manual")
        self.assertEqual(request.trigger, "ESC")


if __name__ == "__main__":
    unittest.main()
