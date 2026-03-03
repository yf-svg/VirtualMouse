from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.gestures.features import (
    WRIST,
    THUMB_MCP,
    THUMB_TIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_DIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_DIP,
    MIDDLE_TIP,
    RING_MCP,
    RING_PIP,
    RING_DIP,
    RING_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_DIP,
    PINKY_TIP,
    finger_is_curled,
    finger_metrics,
    get_landmarks,
    hand_scale,
    min_thumb_tip_distance_to_fingers,
    thumb_is_extended,
    thumb_metrics,
)


@dataclass
class BravoCfg:
    use_3d: bool = True
    min_scale: float = 1e-4

    thumb_tip_to_palm: float = 0.72
    thumb_tip_to_wrist: float = 0.90
    thumb_vertical_ratio: float = 1.35
    thumb_above_wrist: float = 0.08
    thumb_pinch_exclusion: float = 0.30
    min_curled: int = 4


class BravoDetector:
    def __init__(self, cfg: BravoCfg = BravoCfg()):
        self.cfg = cfg

    def detect(self, landmarks: Any) -> bool:
        lm = get_landmarks(landmarks)
        scale = hand_scale(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)

        wrist = lm[WRIST]
        thumb = thumb_metrics(lm, use_3d=self.cfg.use_3d, scale=scale)
        thumb_tip = lm[THUMB_TIP]
        thumb_mcp = lm[THUMB_MCP]

        if not thumb_is_extended(thumb, min_tip_to_palm=self.cfg.thumb_tip_to_palm):
            return False
        if thumb.tip_to_wrist < self.cfg.thumb_tip_to_wrist:
            return False
        if (wrist.y - thumb_tip.y) <= (self.cfg.thumb_above_wrist * scale):
            return False

        dx = thumb_tip.x - thumb_mcp.x
        dy = thumb_tip.y - thumb_mcp.y
        if dy >= 0 or abs(dy) < self.cfg.thumb_vertical_ratio * max(abs(dx), 1e-6):
            return False

        if min_thumb_tip_distance_to_fingers(lm, use_3d=self.cfg.use_3d, scale=scale) < self.cfg.thumb_pinch_exclusion:
            return False

        curled = [
            finger_is_curled(finger_metrics(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, use_3d=self.cfg.use_3d, scale=scale)),
            finger_is_curled(finger_metrics(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, use_3d=self.cfg.use_3d, scale=scale)),
            finger_is_curled(finger_metrics(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, use_3d=self.cfg.use_3d, scale=scale)),
            finger_is_curled(finger_metrics(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, use_3d=self.cfg.use_3d, scale=scale)),
        ]
        return sum(1 for v in curled if v) >= self.cfg.min_curled


_DEFAULT = BravoDetector()

def detect_bravo(landmarks: Any) -> bool:
    return _DEFAULT.detect(landmarks)
