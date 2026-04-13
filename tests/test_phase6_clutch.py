from __future__ import annotations

import unittest

from app.config import ClutchInteractionConfig, PrimaryInteractionConfig
from app.control.clutch import ClutchController, ClutchState
from app.control.cursor_preview import CursorPreviewController
from app.control.cursor_space import CursorPoint
from app.control.primary_interaction import PrimaryInteractionController, PrimaryInteractionState
from app.control.scroll_mode import ScrollModeController
from app.control.secondary_interaction import SecondaryInteractionController
from app.modes.general import resolve_general_action


def pt(x: float, y: float) -> CursorPoint:
    return CursorPoint(x=x, y=y)


class Phase6ClutchTests(unittest.TestCase):
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
            )
        )
        self.secondary = SecondaryInteractionController()
        self.cursor = CursorPreviewController()

    def test_fist_owns_general_mode_and_emits_clutch_hold(self):
        out = resolve_general_action(
            gesture_label="FIST",
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        self.assertEqual(out.clutch.state, ClutchState.CLUTCH_ACTIVE)
        self.assertEqual(out.intent.action_name, "CLUTCH_HOLD")
        self.assertEqual(out.primary.state, PrimaryInteractionState.NEUTRAL)

    def test_clutch_cancels_primary_pending_click(self):
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

        clutched = resolve_general_action(
            gesture_label="FIST",
            cursor_point=pt(0.22, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.10,
        )
        self.assertEqual(clutched.intent.action_name, "CLUTCH_HOLD")
        self.assertEqual(clutched.primary.state, PrimaryInteractionState.NEUTRAL)

        after_window = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.22, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.50,
        )
        self.assertEqual(after_window.intent.action_name, "CLUTCH_RELEASE")

        next_frame = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.22, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.55,
        )
        self.assertEqual(next_frame.intent.action_name, "NO_ACTION")

    def test_clutch_release_suppresses_immediate_primary_reentry_same_frame(self):
        resolve_general_action(
            gesture_label="FIST",
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        released = resolve_general_action(
            gesture_label="PINCH_INDEX",
            cursor_point=pt(0.31, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(released.intent.action_name, "CLUTCH_RELEASE")
        self.assertEqual(released.primary.state, PrimaryInteractionState.NEUTRAL)

        next_frame = resolve_general_action(
            gesture_label="PINCH_INDEX",
            cursor_point=pt(0.31, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.08,
        )
        self.assertEqual(next_frame.primary.state, PrimaryInteractionState.PRIMARY_PINCH_CANDIDATE)

    def test_clutch_hand_loss_exits_safely_after_grace(self):
        resolve_general_action(
            gesture_label="FIST",
            cursor_point=pt(0.30, 0.30),
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
        self.assertEqual(lost.clutch.state, ClutchState.HAND_LOST_SAFE)
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
        self.assertEqual(cancelled.clutch.state, ClutchState.NEUTRAL)
        self.assertEqual(cancelled.intent.action_name, "NO_ACTION")


if __name__ == "__main__":
    unittest.main()
