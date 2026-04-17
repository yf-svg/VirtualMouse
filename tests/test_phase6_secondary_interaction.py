from __future__ import annotations

import unittest

from app.config import (
    ClutchInteractionConfig,
    PrimaryInteractionConfig,
    SecondaryInteractionConfig,
)
from app.control.clutch import ClutchController
from app.control.cursor_preview import CursorPreviewController
from app.control.cursor_space import CursorPoint
from app.control.primary_interaction import PrimaryInteractionController, PrimaryInteractionState
from app.control.scroll_mode import ScrollModeController
from app.control.secondary_interaction import (
    SecondaryInteractionController,
    SecondaryInteractionState,
)
from app.modes.general import resolve_general_action


def pt(x: float, y: float) -> CursorPoint:
    return CursorPoint(x=x, y=y)


class Phase6SecondaryInteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clutch = ClutchController(
            cfg=ClutchInteractionConfig(hand_loss_grace_s=0.10)
        )
        self.scroll = ScrollModeController()
        self.primary = PrimaryInteractionController(
            cfg=PrimaryInteractionConfig(
                drag_start_distance=0.05,
                click_release_tolerance=0.02,
                double_click_window_s=0.30,
                hand_loss_grace_s=0.10,
                enable_double_click=True,
            )
        )
        self.secondary = SecondaryInteractionController(
            cfg=SecondaryInteractionConfig(
                release_tolerance=0.03,
                hand_loss_grace_s=0.10,
            )
        )
        self.cursor = CursorPreviewController()

    def test_secondary_press_does_not_emit_right_click(self):
        out = resolve_general_action(
            gesture_label="PINCH_MIDDLE",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        self.assertEqual(out.secondary.state, SecondaryInteractionState.SECONDARY_PINCH_CANDIDATE)
        self.assertEqual(out.intent.action_name, "NO_ACTION")

    def test_secondary_release_emits_one_right_click(self):
        resolve_general_action(
            gesture_label="PINCH_MIDDLE",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        clicked = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.22, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(clicked.intent.action_name, "SECONDARY_RIGHT_CLICK")
        self.assertEqual(clicked.secondary.state, SecondaryInteractionState.NEUTRAL)

        after = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.22, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.10,
        )
        self.assertEqual(after.intent.action_name, "NO_ACTION")

    def test_secondary_movement_beyond_tolerance_cancels(self):
        resolve_general_action(
            gesture_label="PINCH_MIDDLE",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        cancelled = resolve_general_action(
            gesture_label="PINCH_MIDDLE",
            cursor_point=pt(0.25, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.02,
        )
        self.assertEqual(cancelled.secondary.state, SecondaryInteractionState.NEUTRAL)
        self.assertEqual(cancelled.intent.action_name, "NO_ACTION")

    def test_secondary_hand_loss_fails_safe_without_click(self):
        resolve_general_action(
            gesture_label="PINCH_MIDDLE",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        lost = resolve_general_action(
            gesture_label=None,
            cursor_point=None,
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.03,
        )
        self.assertEqual(lost.secondary.state, SecondaryInteractionState.HAND_LOST_SAFE)
        self.assertEqual(lost.intent.action_name, "NO_ACTION")

        cancelled = resolve_general_action(
            gesture_label=None,
            cursor_point=None,
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.20,
        )
        self.assertEqual(cancelled.secondary.state, SecondaryInteractionState.NEUTRAL)
        self.assertEqual(cancelled.intent.action_name, "NO_ACTION")

    def test_clutch_cancels_secondary_candidate(self):
        resolve_general_action(
            gesture_label="PINCH_MIDDLE",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        clutched = resolve_general_action(
            gesture_label="FIST",
            cursor_point=pt(0.21, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(clutched.intent.action_name, "CLUTCH_HOLD")
        self.assertEqual(clutched.secondary.state, SecondaryInteractionState.NEUTRAL)

    def test_secondary_does_not_interfere_with_primary_pending(self):
        resolve_general_action(
            gesture_label="PINCH_INDEX",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )
        pending = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.21, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(pending.primary.state, PrimaryInteractionState.CLICK_PENDING)

        suppressed = resolve_general_action(
            gesture_label="PINCH_MIDDLE",
            cursor_point=pt(0.21, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.10,
        )
        self.assertEqual(suppressed.primary.state, PrimaryInteractionState.CLICK_PENDING)
        self.assertEqual(suppressed.secondary.state, SecondaryInteractionState.NEUTRAL)
        self.assertEqual(suppressed.intent.action_name, "NO_ACTION")


if __name__ == "__main__":
    unittest.main()
