from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.gestures.pinch import detect_pinch_type, reset_pinch
from app.gestures.hand_gestures import HandGestures
from app.gestures.fist import detect_fist
from app.gestures.bravo import detect_bravo


@dataclass
class GestureSnapshot:
    """
    Raw gesture candidates detected in the current frame.
    These are NOT actions. They are signals.
    """
    pinch: Optional[str]        # PINCH_* or None
    fist: bool
    open_palm: bool
    number: Optional[str]       # ONE..FIVE or None
    l_gesture: bool
    bravo: bool


class GestureRegistry:
    """
    Single entry point for ALL gesture recognition.

    Responsibilities:
    - Run all gesture detectors on raw landmarks
    - Return a GestureSnapshot
    - Reset stateful detectors when hand disappears
    """

    def __init__(self):
        self.hand = HandGestures()

    def reset(self) -> None:
        """
        Reset stateful gesture detectors.
        Must be called when NO HAND is detected.
        """
        try:
            reset_pinch()
        except Exception:
            pass

    def detect(self, hand_landmarks: Any) -> GestureSnapshot:
        """
        Run all gesture recognizers on RAW landmarks.
        """
        fist = detect_fist(hand_landmarks)
        if fist:
            return GestureSnapshot(
                pinch=None,
                fist=True,
                open_palm=False,
                number=None,
                l_gesture=False,
                bravo=False,
            )

        pinch = detect_pinch_type(hand_landmarks)

        open_palm = self.hand.detect_open_palm(hand_landmarks)
        number = self.hand.detect_numbers_1_to_5(hand_landmarks)
        l_gesture = self.hand.detect_L(hand_landmarks)

        # ✅ BRAVO is now handled by a dedicated detector file
        bravo = detect_bravo(hand_landmarks)

        return GestureSnapshot(
            pinch=pinch,
            fist=fist,
            open_palm=open_palm,
            number=number,
            l_gesture=l_gesture,
            bravo=bravo,
        )
