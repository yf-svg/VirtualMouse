from __future__ import annotations

import unittest

from app.config import ClutchInteractionConfig
from app.control.clutch import ClutchController
from app.control.cursor_preview import CursorPreviewController
from app.control.cursor_space import CursorPoint
from app.control.primary_interaction import PrimaryInteractionController, PrimaryInteractionState
from app.control.scroll_mode import ScrollModeController, ScrollState
from app.control.secondary_interaction import SecondaryInteractionController
from app.modes.general import resolve_general_action


def pt(x: float, y: float) -> CursorPoint:
    return CursorPoint(x=x, y=y)


class Phase6GeneralRuntimeHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clutch = ClutchController(cfg=ClutchInteractionConfig(hand_loss_grace_s=0.10))
        self.scroll = ScrollModeController()
        self.primary = PrimaryInteractionController()
        self.secondary = SecondaryInteractionController()
        self.cursor = CursorPreviewController()

    def test_primary_uses_stable_signal_even_before_cursor_label_is_eligible(self):
        out = resolve_general_action(
            gesture_label="PINCH_INDEX",
            cursor_gesture_label=None,
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        self.assertEqual(out.primary.state, PrimaryInteractionState.PRIMARY_PINCH_CANDIDATE)
        self.assertTrue(out.primary.owns_state)

    def test_scroll_toggle_uses_stable_signal_even_before_cursor_label_is_eligible(self):
        out = resolve_general_action(
            gesture_label="SHAKA",
            cursor_gesture_label=None,
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        self.assertEqual(out.scroll.state, ScrollState.SCROLL_MODE_ACTIVE)
        self.assertEqual(out.intent.action_name, "SCROLL_MODE_ENTER")

    def test_cursor_can_still_use_hold_qualified_signal_independent_of_general_signal(self):
        out = resolve_general_action(
            gesture_label=None,
            cursor_gesture_label="CLOSED_PALM",
            cursor_point=pt(0.25, 0.25),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        self.assertEqual(out.cursor.intent.action_name, "CURSOR_PREVIEW_READY")
        self.assertTrue(out.cursor.policy.eligible)


if __name__ == "__main__":
    unittest.main()
