from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp


def normalize_mediapipe_handedness(
    handedness: str | None,
    *,
    input_is_mirrored: bool,
) -> str | None:
    """
    MediaPipe Hands assumes selfie-style mirrored input for handedness.
    Normalize to the user's physical hand unless the incoming detection frame is
    already mirrored before inference.
    """
    if handedness not in {"Left", "Right"}:
        return handedness
    if input_is_mirrored:
        return handedness
    return "Right" if handedness == "Left" else "Left"


@dataclass(frozen=True)
class DetectedHand:
    """
    Stable output contract for the perception layer.
    """
    landmarks: Any
    handedness: str | None = None


class HandTracker:
    """
    MediaPipe Hands wrapper.
    Returns a stable `DetectedHand` object containing landmarks + handedness
    (physical, based on input image).
    Provides standard MediaPipe colored drawing utilities.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 0,
        input_is_mirrored: bool = False,
    ):
        self._input_is_mirrored = bool(input_is_mirrored)
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=int(max_num_hands),
            model_complexity=int(model_complexity),
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

        # Proper MediaPipe drawing (colored connections)
        self._drawer = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles

    def detect(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        lm = results.multi_hand_landmarks[0]
        handedness = None
        if results.multi_handedness:
            raw_handedness = results.multi_handedness[0].classification[0].label
            handedness = normalize_mediapipe_handedness(
                raw_handedness,
                input_is_mirrored=self._input_is_mirrored,
            )

        return DetectedHand(landmarks=lm, handedness=handedness)

    def draw(self, frame_bgr, landmarks) -> None:
        """Standard MediaPipe colored connections + landmark styles."""
        self._drawer.draw_landmarks(
            frame_bgr,
            landmarks,
            self._mp_hands.HAND_CONNECTIONS,
            self._styles.get_default_hand_landmarks_style(),
            self._styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        self._hands.close()
