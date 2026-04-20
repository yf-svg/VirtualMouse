from __future__ import annotations

import unittest

from app.control.cursor_space import CursorPoint
from app.control.pointer_filters import AdaptiveEmaFilter, DropoutHold


class PointerFilterTests(unittest.TestCase):
    def test_adaptive_ema_filter_uses_more_smoothing_for_small_motion(self):
        filter_ = AdaptiveEmaFilter(
            history_size=4,
            alpha_min=0.2,
            alpha_max=0.8,
            speed_low=0.10,
            speed_high=0.50,
        )

        filter_.update(CursorPoint(0.0, 0.0), timestamp=0.0)
        slow = filter_.update(CursorPoint(0.05, 0.0), timestamp=1.0)

        self.assertAlmostEqual(slow.x, 0.01, places=6)
        self.assertAlmostEqual(slow.y, 0.0, places=6)

    def test_adaptive_ema_filter_uses_less_smoothing_for_large_motion(self):
        filter_ = AdaptiveEmaFilter(
            history_size=4,
            alpha_min=0.2,
            alpha_max=0.8,
            speed_low=0.10,
            speed_high=0.50,
        )

        filter_.update(CursorPoint(0.0, 0.0), timestamp=0.0)
        fast = filter_.update(CursorPoint(0.60, 0.0), timestamp=1.0)

        self.assertAlmostEqual(fast.x, 0.48, places=6)
        self.assertAlmostEqual(fast.y, 0.0, places=6)

    def test_dropout_hold_reuses_last_point_for_configured_frames(self):
        hold = DropoutHold(hold_frames=2)
        point = CursorPoint(0.25, 0.75)

        first = hold.apply(point)
        second = hold.apply(None)
        third = hold.apply(None)
        fourth = hold.apply(None)

        self.assertEqual(first, point)
        self.assertEqual(second, point)
        self.assertEqual(third, point)
        self.assertIsNone(fourth)


if __name__ == "__main__":
    unittest.main()
