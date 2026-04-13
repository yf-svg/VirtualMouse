from __future__ import annotations

import unittest

from app.config import ClutchInteractionConfig, PrimaryInteractionConfig
from app.control.clutch import ClutchController
from app.control.cursor_preview import CursorPreviewController
from app.control.cursor_space import CursorPoint
from app.control.primary_interaction import (
    PrimaryInteractionController,
    PrimaryInteractionState,
)
from app.control.scroll_mode import ScrollModeController
from app.control.secondary_interaction import SecondaryInteractionController
from app.modes.general import resolve_general_action


def pt(x: float, y: float) -> CursorPoint:
    return CursorPoint(x=x, y=y)


class Phase6PrimaryInteractionTests(unittest.TestCase):
    def setUp(self) -> None:
        cfg = PrimaryInteractionConfig(
            drag_start_distance=0.05,
            click_release_tolerance=0.02,
            double_click_window_s=0.30,
            hand_loss_grace_s=0.10,
        )
        self.clutch = ClutchController(cfg=ClutchInteractionConfig(hand_loss_grace_s=0.10))
        self.scroll = ScrollModeController()
        self.controller = PrimaryInteractionController(cfg=cfg)
        self.secondary = SecondaryInteractionController()
        self.cursor = CursorPreviewController()

    def test_primary_pinch_release_enters_click_pending_then_emits_single_click(self):
        first = self.controller.update(gesture_label="PINCH_INDEX", cursor_point=pt(0.20, 0.20), now=0.00)
        self.assertEqual(first.state, PrimaryInteractionState.PRIMARY_PINCH_CANDIDATE)
        self.assertTrue(first.owns_state)

        released = self.controller.update(gesture_label=None, cursor_point=pt(0.21, 0.20), now=0.05)
        self.assertEqual(released.state, PrimaryInteractionState.CLICK_PENDING)
        self.assertEqual(released.intent.action_name, "NO_ACTION")

        waiting = self.controller.update(gesture_label=None, cursor_point=pt(0.21, 0.20), now=0.20)
        self.assertEqual(waiting.state, PrimaryInteractionState.CLICK_PENDING)
        self.assertEqual(waiting.intent.action_name, "NO_ACTION")

        clicked = self.controller.update(gesture_label=None, cursor_point=pt(0.21, 0.20), now=0.40)
        self.assertEqual(clicked.state, PrimaryInteractionState.NEUTRAL)
        self.assertEqual(clicked.intent.action_name, "PRIMARY_CLICK")

    def test_double_click_resolves_without_single_click_first(self):
        self.controller.update(gesture_label="PINCH_INDEX", cursor_point=pt(0.20, 0.20), now=0.00)
        first_release = self.controller.update(gesture_label=None, cursor_point=pt(0.21, 0.20), now=0.05)
        self.assertEqual(first_release.state, PrimaryInteractionState.CLICK_PENDING)
        self.assertEqual(first_release.intent.action_name, "NO_ACTION")

        second_press = self.controller.update(gesture_label="PINCH_INDEX", cursor_point=pt(0.20, 0.20), now=0.12)
        self.assertEqual(second_press.state, PrimaryInteractionState.PRIMARY_PINCH_CANDIDATE)

        double_clicked = self.controller.update(gesture_label=None, cursor_point=pt(0.21, 0.20), now=0.18)
        self.assertEqual(double_clicked.state, PrimaryInteractionState.NEUTRAL)
        self.assertEqual(double_clicked.intent.action_name, "PRIMARY_DOUBLE_CLICK")

        after = self.controller.update(gesture_label=None, cursor_point=pt(0.21, 0.20), now=0.45)
        self.assertEqual(after.intent.action_name, "NO_ACTION")

    def test_drag_requires_movement_not_time(self):
        self.controller.update(gesture_label="PINCH_INDEX", cursor_point=pt(0.10, 0.10), now=0.00)

        holding = self.controller.update(gesture_label="PINCH_INDEX", cursor_point=pt(0.11, 0.10), now=1.50)
        self.assertEqual(holding.state, PrimaryInteractionState.PRIMARY_PINCH_CANDIDATE)
        self.assertEqual(holding.intent.action_name, "NO_ACTION")

        drag_started = self.controller.update(gesture_label="PINCH_INDEX", cursor_point=pt(0.17, 0.10), now=1.55)
        self.assertEqual(drag_started.state, PrimaryInteractionState.DRAG_ACTIVE)
        self.assertEqual(drag_started.intent.action_name, "PRIMARY_DRAG_START")

    def test_drag_release_emits_drag_end(self):
        self.controller.update(gesture_label="PINCH_INDEX", cursor_point=pt(0.10, 0.10), now=0.00)
        self.controller.update(gesture_label="PINCH_INDEX", cursor_point=pt(0.17, 0.10), now=0.05)

        released = self.controller.update(gesture_label=None, cursor_point=pt(0.18, 0.10), now=0.08)
        self.assertEqual(released.state, PrimaryInteractionState.NEUTRAL)
        self.assertEqual(released.intent.action_name, "PRIMARY_DRAG_END")

    def test_hand_loss_during_candidate_cancels_without_click(self):
        self.controller.update(gesture_label="PINCH_INDEX", cursor_point=pt(0.20, 0.20), now=0.00)

        lost = self.controller.update(gesture_label=None, cursor_point=None, now=0.03)
        self.assertEqual(lost.state, PrimaryInteractionState.HAND_LOST_SAFE)
        self.assertEqual(lost.intent.action_name, "NO_ACTION")

        cancelled = self.controller.update(gesture_label=None, cursor_point=None, now=0.20)
        self.assertEqual(cancelled.state, PrimaryInteractionState.NEUTRAL)
        self.assertEqual(cancelled.intent.action_name, "NO_ACTION")

    def test_general_mode_falls_back_only_when_primary_and_clutch_do_not_own_state(self):
        out = resolve_general_action(
            gesture_label="BRAVO",
            cursor_point=pt(0.25, 0.25),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.controller,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        self.assertEqual(out.primary.state, PrimaryInteractionState.NEUTRAL)
        self.assertFalse(out.primary.owns_state)
        self.assertEqual(out.intent.action_name, "NO_ACTION")
        self.assertEqual(out.intent.reason, "unmapped_gesture:BRAVO")


if __name__ == "__main__":
    unittest.main()
