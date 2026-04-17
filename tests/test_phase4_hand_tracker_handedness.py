from __future__ import annotations

import unittest

from app.perception.hand_tracker import normalize_mediapipe_handedness


class HandTrackerHandednessTests(unittest.TestCase):
    def test_unmirrored_input_swaps_mediapipe_handedness_to_physical_hand(self):
        self.assertEqual(
            normalize_mediapipe_handedness("Left", input_is_mirrored=False),
            "Right",
        )
        self.assertEqual(
            normalize_mediapipe_handedness("Right", input_is_mirrored=False),
            "Left",
        )

    def test_mirrored_input_preserves_mediapipe_handedness(self):
        self.assertEqual(
            normalize_mediapipe_handedness("Left", input_is_mirrored=True),
            "Left",
        )
        self.assertEqual(
            normalize_mediapipe_handedness("Right", input_is_mirrored=True),
            "Right",
        )

    def test_unknown_handedness_is_left_untouched(self):
        self.assertIsNone(normalize_mediapipe_handedness(None, input_is_mirrored=False))
        self.assertEqual(
            normalize_mediapipe_handedness("Unknown", input_is_mirrored=False),
            "Unknown",
        )


if __name__ == "__main__":
    unittest.main()
