from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.gestures.features import (
    WRIST,
    THUMB_TIP,
    distance,
    get_landmarks,
    min_thumb_tip_distance_to_fingers,
    palm_center,
    summarize_hand_pose,
)


@dataclass
class FistCfg:
    use_3d: bool = True
    min_curled: int = 3
    min_near_palm: int = 4
    max_extended: int = 0
    tip_near_palm: float = 1.05
    thumb_near_palm: float = 1.15
    pinch_exclusion: float = 0.02
    min_scale: float = 1e-4


class FistDetector:
    def __init__(self, cfg: FistCfg = FistCfg()):
        self.cfg = cfg

    def is_fist(self, landmarks: Any) -> bool:
        lm = get_landmarks(landmarks)
        summary = summarize_hand_pose(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)
        scale = summary.scale
        palm = palm_center(lm)
        if summary.curled_count < self.cfg.min_curled:
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
