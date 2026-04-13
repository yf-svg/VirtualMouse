from __future__ import annotations

import unittest

from app.modes.general import GENERAL_DRY_RUN_BINDINGS, map_gesture_to_action


class Phase6GeneralModeDryRunTests(unittest.TestCase):
    def test_general_mode_returns_no_action_when_no_eligible_gesture(self):
        out = map_gesture_to_action(None)

        self.assertEqual(out.action_name, "NO_ACTION")
        self.assertFalse(out.executable)
        self.assertEqual(out.reason, "no_eligible_gesture")

    def test_general_mode_marks_unapproved_gesture_as_pending(self):
        out = map_gesture_to_action("FIST")

        self.assertEqual(out.action_name, "NO_ACTION")
        self.assertFalse(out.executable)
        self.assertEqual(out.gesture_label, "FIST")
        self.assertEqual(out.reason, "unmapped_gesture:FIST")

    def test_general_mode_bindings_preserve_one_gesture_one_action(self):
        action_names = list(GENERAL_DRY_RUN_BINDINGS.values())
        self.assertEqual(len(action_names), len(set(action_names)))


if __name__ == "__main__":
    unittest.main()
