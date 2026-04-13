from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.gestures.features import (
    WRIST,
    THUMB_TIP,
    distance,
    finger_is_curled,
    finger_is_extended,
    finger_metrics,
    get_landmarks,
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    min_thumb_tip_distance_to_fingers,
    palm_center,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    summarize_hand_pose,
)


@dataclass
class FistCfg:
    use_3d: bool = True
    min_curled: int = 2
    min_near_palm: int = 4
    max_extended: int = 0
    tip_near_palm: float = 1.05
    thumb_near_palm: float = 1.15
    pinch_exclusion: float = 0.02
    min_scale: float = 1e-4
    compact_tip_to_palm: float = 0.96
    compact_curl_ratio: float = 1.10
    min_compact: int = 4


class FistDetector:
    def __init__(self, cfg: FistCfg = FistCfg()):
        self.cfg = cfg

    def _finger_compact(self, lm, mcp: int, pip: int, dip: int, tip: int, *, scale: float) -> bool:
        metrics = finger_metrics(lm, mcp, pip, dip, tip, use_3d=self.cfg.use_3d, scale=scale)
        if finger_is_extended(metrics):
            return False
        return (
            finger_is_curled(metrics)
            or (
                metrics.tip_to_palm <= self.cfg.compact_tip_to_palm
                and metrics.curl_ratio <= self.cfg.compact_curl_ratio
            )
        )

    def is_fist(self, landmarks: Any) -> bool:
        lm = get_landmarks(landmarks)
        summary = summarize_hand_pose(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)
        scale = summary.scale
        palm = palm_center(lm)
        if summary.curled_count < self.cfg.min_curled:
            compact = [
                self._finger_compact(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, scale=scale),
                self._finger_compact(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, scale=scale),
                self._finger_compact(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, scale=scale),
                self._finger_compact(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, scale=scale),
            ]
            if sum(1 for v in compact if v) < self.cfg.min_compact:
                return False
        if summary.extended_count > self.cfg.max_extended:
            return False

        tips = [lm[8], lm[12], lm[16], lm[20]]
        near_palm = sum(
            1 for t in tips
            if (distance(t, palm, use_3d=self.cfg.use_3d) / scale) <= self.cfg.tip_near_palm
        )
        if near_palm < self.cfg.min_near_palm:
            return False

        # A natural fist can still produce a borderline "extended" thumb angle
        # when the thumb rests across the curled fingers, so avoid hard-failing
        # on angle alone and instead rely on global hand closure and thumb pose.
        if summary.thumb.tip_to_palm > self.cfg.thumb_near_palm:
            return False

        # Only reject a very isolated thumb-finger pinch; ordinary thumb contact
        # against curled fingers is part of many natural fist poses.
        if min_thumb_tip_distance_to_fingers(lm, use_3d=self.cfg.use_3d, scale=scale) < self.cfg.pinch_exclusion:
            return False

        return True


_DEFAULT = FistDetector()

def detect_fist(landmarks: Any) -> bool:
    return _DEFAULT.is_fist(landmarks)
