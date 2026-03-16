from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.gestures.features import (
    WRIST,
    THUMB_MCP,
    THUMB_IP,
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
    finger_is_extended,
    finger_metrics,
    get_landmarks,
    hand_scale,
    min_thumb_tip_distance_to_fingers,
    thumb_is_extended,
    thumb_metrics,
)


@dataclass
class ThumbsDownCfg:
    use_3d: bool = True
    min_scale: float = 1e-4

    thumb_tip_to_palm: float = 0.62
    thumb_tip_to_wrist: float = 0.82
    thumb_vertical_ratio: float = 1.05
    thumb_below_wrist: float = 0.02
    thumb_below_knuckles: float = 0.04
    thumb_pinch_exclusion: float = 0.24
    compact_tip_to_palm: float = 0.88
    compact_curl_ratio: float = 1.10
    min_curled: int = 4


class ThumbsDownDetector:
    def __init__(self, cfg: ThumbsDownCfg = ThumbsDownCfg()):
        self.cfg = cfg

    def _finger_compact(self, lm, mcp: int, pip: int, dip: int, tip: int, *, scale: float) -> bool:
        """
        A thumbs-down pose is a fist-like hand with the thumb isolated downward.
        For the back-of-hand view, accept either a strict curl or a compact folded
        finger, but reject any finger that reads as clearly extended.
        """
        metrics = finger_metrics(lm, mcp, pip, dip, tip, use_3d=self.cfg.use_3d, scale=scale)
        if finger_is_extended(metrics):
            return False
        return (
            finger_is_curled(metrics)
            or (
                metrics.tip_to_palm <= self.cfg.compact_tip_to_palm
                and metrics.curl_ratio <= self.cfg.compact_curl_ratio
                and lm[tip].y > lm[pip].y
            )
        )

    def detect(self, landmarks: Any) -> bool:
        lm = get_landmarks(landmarks)
        scale = hand_scale(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)

        wrist = lm[WRIST]
        thumb = thumb_metrics(lm, use_3d=self.cfg.use_3d, scale=scale)
        thumb_tip = lm[THUMB_TIP]
        thumb_ip = lm[THUMB_IP]
        thumb_mcp = lm[THUMB_MCP]
        knuckle_y = (
            lm[INDEX_MCP].y + lm[MIDDLE_MCP].y + lm[RING_MCP].y + lm[PINKY_MCP].y
        ) / 4.0

        if not thumb_is_extended(thumb, min_tip_to_palm=self.cfg.thumb_tip_to_palm):
            return False
        if thumb.tip_to_wrist < self.cfg.thumb_tip_to_wrist:
            return False
        if (thumb_tip.y - wrist.y) <= (self.cfg.thumb_below_wrist * scale):
            return False
        if (thumb_tip.y - knuckle_y) <= (self.cfg.thumb_below_knuckles * scale):
            return False
        if not (thumb_tip.y > thumb_ip.y > thumb_mcp.y):
            return False

        dx = thumb_tip.x - thumb_mcp.x
        dy = thumb_tip.y - thumb_mcp.y
        if dy <= 0 or dy < self.cfg.thumb_vertical_ratio * max(abs(dx), 1e-6):
            return False

        if min_thumb_tip_distance_to_fingers(lm, use_3d=self.cfg.use_3d, scale=scale) < self.cfg.thumb_pinch_exclusion:
            return False

        compact = [
            self._finger_compact(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, scale=scale),
            self._finger_compact(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, scale=scale),
            self._finger_compact(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, scale=scale),
            self._finger_compact(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, scale=scale),
        ]
        return sum(1 for v in compact if v) >= self.cfg.min_curled


_DEFAULT = ThumbsDownDetector()


def detect_thumbs_down(landmarks: Any) -> bool:
    return _DEFAULT.detect(landmarks)
