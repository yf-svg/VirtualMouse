from __future__ import annotations

import unittest

from app.config import (
    ClutchInteractionConfig,
    CursorPolicyConfig,
    CursorPreviewConfig,
    PrimaryInteractionConfig,
)
from app.control.clutch import ClutchController
from app.control.cursor_policy import CursorPolicy
from app.control.cursor_preview import CursorPreviewController, CursorPreviewState
from app.control.cursor_space import CursorPoint
from app.control.primary_interaction import PrimaryInteractionController
from app.control.scroll_mode import ScrollModeController
from app.control.secondary_interaction import SecondaryInteractionController
from app.modes.general import resolve_general_action


def pt(x: float, y: float) -> CursorPoint:
    return CursorPoint(x=x, y=y)


class Phase6CursorPreviewTests(unittest.TestCase):
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
        self.cursor = CursorPreviewController(
            policy=CursorPolicy(
                cfg=CursorPolicyConfig(
                    allowed_gestures=("L",),
                    provisional=True,
                )
            ),
            cfg=CursorPreviewConfig(
                move_epsilon=0.002,
                gain=1.0,
            ),
        )

    def test_cursor_policy_is_isolated_and_provisional(self):
        out = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )

        self.assertTrue(out.cursor.policy.eligible)
        self.assertTrue(out.cursor.policy.provisional)
        self.assertEqual(out.cursor.policy.gesture_label, "L")

    def test_default_cursor_policy_now_targets_closed_palm(self):
        cursor = CursorPreviewController(cfg=CursorPreviewConfig(move_epsilon=0.002, gain=1.0))

        rejected = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=cursor,
            now=0.00,
        )
        self.assertFalse(rejected.cursor.policy.eligible)

        accepted = resolve_general_action(
            gesture_label="CLOSED_PALM",
            cursor_point=pt(0.22, 0.22),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=cursor,
            now=0.05,
        )
        self.assertTrue(accepted.cursor.policy.eligible)
        self.assertEqual(accepted.cursor.policy.gesture_label, "CLOSED_PALM")
        self.assertEqual(accepted.intent.action_name, "CURSOR_PREVIEW_READY")

    def test_cursor_preview_moves_when_no_higher_priority_owner_exists(self):
        ready = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )
        self.assertEqual(ready.cursor.state, CursorPreviewState.CURSOR_ACTIVE)
        self.assertEqual(ready.intent.action_name, "CURSOR_PREVIEW_READY")

        moved = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.25, 0.22),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(moved.intent.action_name, "CURSOR_PREVIEW_MOVE")
        self.assertIsNotNone(moved.cursor.preview_point)
        self.assertAlmostEqual(moved.cursor.preview_point.x, 0.25, places=3)
        self.assertAlmostEqual(moved.cursor.preview_point.y, 0.22, places=3)

    def test_cursor_preview_can_be_seeded_from_existing_os_cursor_without_jump(self):
        self.cursor.seed_preview_point(pt(0.60, 0.40))

        ready = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )
        self.assertEqual(ready.intent.action_name, "CURSOR_PREVIEW_READY")
        self.assertIsNotNone(ready.cursor.preview_point)
        self.assertAlmostEqual(ready.cursor.preview_point.x, 0.60, places=3)
        self.assertAlmostEqual(ready.cursor.preview_point.y, 0.40, places=3)

        moved = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.25, 0.22),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        self.assertEqual(moved.intent.action_name, "CURSOR_PREVIEW_MOVE")
        self.assertIsNotNone(moved.cursor.preview_point)
        self.assertAlmostEqual(moved.cursor.preview_point.x, 0.65, places=3)
        self.assertAlmostEqual(moved.cursor.preview_point.y, 0.42, places=3)

    def test_cursor_is_suppressed_by_primary_ownership(self):
        suppressed = resolve_general_action(
            gesture_label="PINCH_INDEX",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )
        self.assertEqual(suppressed.intent.action_name, "NO_ACTION")
        self.assertEqual(suppressed.cursor.state, CursorPreviewState.NEUTRAL)

    def test_cursor_reanchors_after_clutch_without_jump(self):
        resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.00,
        )
        moved = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.24, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.03,
        )
        before_clutch = moved.cursor.preview_point

        resolve_general_action(
            gesture_label="FIST",
            cursor_point=pt(0.70, 0.70),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.05,
        )
        released = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.80, 0.80),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.08,
        )
        self.assertEqual(released.intent.action_name, "CLUTCH_RELEASE")
        self.assertAlmostEqual(released.cursor.preview_point.x, before_clutch.x, places=3)
        self.assertAlmostEqual(released.cursor.preview_point.y, before_clutch.y, places=3)

        reanchored = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.80, 0.80),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.10,
        )
        self.assertEqual(reanchored.intent.action_name, "CURSOR_PREVIEW_READY")
        self.assertAlmostEqual(reanchored.cursor.preview_point.x, before_clutch.x, places=3)
        self.assertAlmostEqual(reanchored.cursor.preview_point.y, before_clutch.y, places=3)

        resumed = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.82, 0.83),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=self.cursor,
            now=0.12,
        )
        self.assertEqual(resumed.intent.action_name, "CURSOR_PREVIEW_MOVE")

    def test_cursor_preview_can_swap_policy_without_touching_ownership_chain(self):
        cursor = CursorPreviewController(
            policy=CursorPolicy(
                cfg=CursorPolicyConfig(
                    allowed_gestures=("OPEN_PALM",),
                    provisional=True,
                )
            ),
            cfg=CursorPreviewConfig(move_epsilon=0.002, gain=1.0),
        )

        rejected = resolve_general_action(
            gesture_label="L",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=cursor,
            now=0.00,
        )
        self.assertFalse(rejected.cursor.policy.eligible)

        accepted = resolve_general_action(
            gesture_label="OPEN_PALM",
            cursor_point=pt(0.20, 0.20),
            clutch_controller=self.clutch,
            scroll_controller=self.scroll,
            primary_controller=self.primary,
            secondary_controller=self.secondary,
            cursor_controller=cursor,
            now=0.05,
        )
        self.assertTrue(accepted.cursor.policy.eligible)
        self.assertEqual(accepted.intent.action_name, "CURSOR_PREVIEW_READY")


if __name__ == "__main__":
    unittest.main()
