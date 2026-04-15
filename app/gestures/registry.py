from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.gestures.bravo import detect_bravo
from app.gestures.fist import detect_fist
from app.gestures.hand_gestures import HandGestures
from app.gestures.pinch import detect_pinch_type, reset_pinch
from app.gestures.pointing import detect_point_left, detect_point_right
from app.gestures.thumbs_down import detect_thumbs_down


@dataclass
class GestureSnapshot:
    """
    Raw gesture candidates detected in the current frame.
    These are NOT actions. They are signals.
    """

    pinch: Optional[str]  # PINCH_* or None
    fist: bool
    closed_palm: bool
    open_palm: bool
    shaka: bool
    peace_sign: bool
    number: Optional[str]  # ONE..FIVE or None
    l_gesture: bool
    bravo: bool
    thumbs_down: bool
    point_right: bool
    point_left: bool


class GestureRegistry:
    """
    Single entry point for all gesture recognition.

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
        Must be called when no hand is detected.
        """
        try:
            reset_pinch()
        except Exception:
            pass

    def detect(self, hand_landmarks: Any) -> GestureSnapshot:
        """
        Run all gesture recognizers on raw landmarks.
        """
        thumbs_down = detect_thumbs_down(hand_landmarks)
        if thumbs_down:
            return GestureSnapshot(
                pinch=None,
                fist=False,
                closed_palm=False,
                open_palm=False,
                shaka=False,
                peace_sign=False,
                number=None,
                l_gesture=False,
                bravo=False,
                thumbs_down=True,
                point_right=False,
                point_left=False,
            )

        fist = detect_fist(hand_landmarks)
        if fist:
            return GestureSnapshot(
                pinch=None,
                fist=True,
                closed_palm=False,
                open_palm=False,
                shaka=False,
                peace_sign=False,
                number=None,
                l_gesture=False,
                bravo=False,
                thumbs_down=False,
                point_right=False,
                point_left=False,
            )

        closed_palm = self.hand.detect_closed_palm(hand_landmarks)
        open_palm = (not closed_palm) and self.hand.detect_open_palm(hand_landmarks)
        shaka = self.hand.detect_shaka(hand_landmarks)
        peace_sign = self.hand.detect_peace_sign(hand_landmarks)
        if peace_sign or shaka:
            pinch = None
        else:
            pinch = detect_pinch_type(hand_landmarks)
        number = self.hand.detect_numbers_1_to_5(hand_landmarks)
        l_gesture = self.hand.detect_L(hand_landmarks)
        point_right = detect_point_right(hand_landmarks)
        point_left = detect_point_left(hand_landmarks)
        bravo = detect_bravo(hand_landmarks)

        return GestureSnapshot(
            pinch=pinch,
            fist=fist,
            closed_palm=closed_palm,
            open_palm=open_palm,
            shaka=shaka,
            peace_sign=peace_sign,
            number=number,
            l_gesture=l_gesture,
            bravo=bravo,
            thumbs_down=False,
            point_right=point_right,
            point_left=point_left,
        )
