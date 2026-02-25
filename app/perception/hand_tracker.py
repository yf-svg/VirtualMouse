from __future__ import annotations

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
    ):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=int(max_num_hands),
            model_complexity=0,  # FASTEST
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._drawer = mp.solutions.drawing_utils

    def detect(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = None
        if results.multi_handedness:
            handedness = results.multi_handedness[0].classification[0].label  # "Left"/"Right"

        return {"landmarks": hand_landmarks, "handedness": handedness}

    def draw(self, frame_bgr, hand_result) -> None:
        if not hand_result:
            return
        self._drawer.draw_landmarks(
            frame_bgr,
            hand_result["landmarks"],
            self._mp_hands.HAND_CONNECTIONS,
        )

    def close(self) -> None:
        self._hands.close()