from __future__ import annotations

import unittest

from app.config import (
    ClutchInteractionConfig,
    PrimaryInteractionConfig,
    ScrollInteractionConfig,
)
from app.control.clutch import ClutchController
from app.control.cursor_preview import CursorPreviewController
from app.control.cursor_space import CursorPoint
from app.control.primary_interaction import PrimaryInteractionController, PrimaryInteractionState
from app.control.scroll_mode import ScrollAxis, ScrollModeController, ScrollState
from app.control.secondary_interaction import SecondaryInteractionController, SecondaryInteractionState
from app.modes.general import resolve_general_action


def pt(x: float, y: float) -> CursorPoint:
    return CursorPoint(x=x, y=y)


class Phase6ScrollModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clutch = ClutchController(
            cfg=ClutchInteractionConfig(hand_loss_grace_s=0.10)
        )
        self.scroll = ScrollModeController(
            cfg=ScrollInteractionConfig(
                dead_zone=0.03,
                axis_dominance_margin=0.01,
                pause_reset_s=0.20,
                pause_motion_epsilon=0.005,
                scroll_gain=1.0,
                hand_loss_grace_s=0.10,
            )
        )
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

    def test_toggle_is_edge_triggered_and_requires_rearm(self):
        entered = resolve_general_action(
            gesture_label="SHAKA",
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )
        self.assertEqual(entered.intent.action_name, "SCROLL_MODE_ENTER")
        self.assertEqual(entered.scroll.state, ScrollState.SCROLL_MODE_ACTIVE)

        held = resolve_general_action(
            gesture_label="SHAKA",
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(held.intent.action_name, "NO_ACTION")
        self.assertEqual(held.scroll.state, ScrollState.SCROLL_MODE_ACTIVE)

        release = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.08,
        )
        self.assertEqual(release.intent.action_name, "NO_ACTION")

        exited = resolve_general_action(
            gesture_label="SHAKA",
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.10,
        )
        self.assertEqual(exited.intent.action_name, "SCROLL_MODE_EXIT")
        self.assertEqual(exited.scroll.state, ScrollState.NEUTRAL)

    def test_dead_zone_prevents_scroll_on_entry_jitter(self):
        resolve_general_action(
            gesture_label="SHAKA",
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        jitter = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.315, 0.31),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(jitter.intent.action_name, "NO_ACTION")
        self.assertEqual(jitter.scroll.axis, ScrollAxis.NONE)

    def test_axis_locks_then_scrolls_on_one_axis_only(self):
        resolve_general_action(
            gesture_label="SHAKA",
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        locked = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.305, 0.35),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(locked.scroll.axis, ScrollAxis.VERTICAL)
        self.assertEqual(locked.intent.action_name, "NO_ACTION")

        scrolled = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.33, 0.39),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.08,
        )
        self.assertEqual(scrolled.scroll.axis, ScrollAxis.VERTICAL)
        self.assertEqual(scrolled.intent.action_name, "SCROLL_VERTICAL")

    def test_pause_resets_axis_lock(self):
        resolve_general_action(
            gesture_label="SHAKA",
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )
        resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.305, 0.35),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )

        wait1 = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.305, 0.351),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.10,
        )
        self.assertEqual(wait1.scroll.axis, ScrollAxis.VERTICAL)

        reset = resolve_general_action(
            gesture_label=None,
            cursor_point=pt(0.305, 0.351),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.35,
        )
        self.assertEqual(reset.scroll.axis, ScrollAxis.NONE)
        self.assertEqual(reset.intent.action_name, "NO_ACTION")

    def test_scroll_mode_suppresses_primary_and_secondary_progression(self):
        resolve_general_action(
            gesture_label="SHAKA",
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        suppressed = resolve_general_action(
            gesture_label="PINCH_INDEX",
            cursor_point=pt(0.31, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(suppressed.scroll.state, ScrollState.SCROLL_MODE_ACTIVE)
        self.assertEqual(suppressed.primary.state, PrimaryInteractionState.NEUTRAL)
        self.assertEqual(suppressed.secondary.state, SecondaryInteractionState.NEUTRAL)

    def test_clutch_cancels_scroll_immediately(self):
        resolve_general_action(
            gesture_label="SHAKA",
            cursor_point=pt(0.30, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        clutched = resolve_general_action(
            gesture_label="FIST",
            cursor_point=pt(0.31, 0.30),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(clutched.intent.action_name, "CLUTCH_HOLD")
        self.assertEqual(clutched.scroll.state, ScrollState.NEUTRAL)

    def test_prolonged_hand_loss_exits_scroll_mode_safely(self):
        resolve_general_action(
            gesture_label="SHAKA",
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
        self.assertEqual(lost.scroll.state, ScrollState.HAND_LOST_SAFE)

        exited = resolve_general_action(
            gesture_label=None,
            cursor_point=None,
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.20,
        )
        self.assertEqual(exited.scroll.state, ScrollState.NEUTRAL)
        self.assertEqual(exited.intent.action_name, "SCROLL_MODE_EXIT")


if __name__ == "__main__":
    unittest.main()
