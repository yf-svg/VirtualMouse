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
    distance,
    finger_is_extended,
    finger_is_curled,
    finger_metrics,
    get_landmarks,
    hand_scale,
    min_thumb_tip_distance_to_fingers,
    palm_width,
    thumb_is_extended,
    thumb_metrics,
)


@dataclass
class BravoCfg:
    use_3d: bool = True
    min_scale: float = 1e-4

    thumb_tip_to_palm: float = 0.72
    thumb_tip_to_wrist: float = 0.82
    thumb_vertical_ratio: float = 1.05
    thumb_above_wrist: float = 0.04
    thumb_above_knuckles: float = 0.02
    thumb_above_fingertips_ratio: float = 0.85
    thumb_pinch_exclusion: float = 0.24
    curled_tip_to_palm: float = 0.95
    curled_curl_ratio: float = 1.45
    min_curled: int = 3
    max_extended: int = 0
    max_tip_cluster_width_ratio: float = 1.10
    max_tip_cluster_height_ratio: float = 0.90
    pinky_leak_tip_gap_ratio: float = 0.45
    pinky_leak_tip_to_palm: float = 0.56
    pinky_leak_curl_ratio: float = 1.08


class BravoDetector:
    def __init__(self, cfg: BravoCfg = BravoCfg()):
        self.cfg = cfg

    def _finger_compact(self, lm, mcp: int, pip: int, dip: int, tip: int, *, scale: float) -> bool:
        """
        Back-of-hand camera views often make a curled finger look less tightly folded
        than palm-side views. Accept a strict curl or a compact folded profile, but
        still reject any finger that reads as clearly extended.
        """
        metrics = finger_metrics(lm, mcp, pip, dip, tip, use_3d=self.cfg.use_3d, scale=scale)
        if finger_is_extended(metrics):
            return False
        return (
            finger_is_curled(metrics)
            or (
                metrics.tip_to_palm <= self.cfg.curled_tip_to_palm
                and metrics.curl_ratio <= self.cfg.curled_curl_ratio
            )
        )

    def _non_thumb_tip_cluster_ok(self, lm, *, palm: float) -> bool:
        tip_x = [lm[INDEX_TIP].x, lm[MIDDLE_TIP].x, lm[RING_TIP].x, lm[PINKY_TIP].x]
        tip_y = [lm[INDEX_TIP].y, lm[MIDDLE_TIP].y, lm[RING_TIP].y, lm[PINKY_TIP].y]
        width_ratio = (max(tip_x) - min(tip_x)) / palm
        height_ratio = (max(tip_y) - min(tip_y)) / palm
        return (
            width_ratio <= self.cfg.max_tip_cluster_width_ratio
            and height_ratio <= self.cfg.max_tip_cluster_height_ratio
        )

    def _pinky_visibly_out(self, lm, *, scale: float, palm: float) -> bool:
        """
        Reject thumbs-up / BRAVO when the pinky is clearly leaking away from the
        curled fingertip cluster. This catches wrist-twisted SHAKA-like poses
        where the pinky is not straight enough to satisfy the generic
        finger_is_extended() rule but is still visibly out.
        """
        pinky = finger_metrics(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, use_3d=self.cfg.use_3d, scale=scale)
        ring_pinky_gap_ratio = distance(lm[RING_TIP], lm[PINKY_TIP], use_3d=self.cfg.use_3d) / palm
        return (
            ring_pinky_gap_ratio >= self.cfg.pinky_leak_tip_gap_ratio
            and (
                pinky.tip_to_palm >= self.cfg.pinky_leak_tip_to_palm
                or pinky.curl_ratio >= self.cfg.pinky_leak_curl_ratio
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
        highest_fingertip_y = min(lm[INDEX_TIP].y, lm[MIDDLE_TIP].y, lm[RING_TIP].y, lm[PINKY_TIP].y)
        palm = max(
            palm_width(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale),
            self.cfg.min_scale,
        )

        if not thumb_is_extended(thumb, min_tip_to_palm=self.cfg.thumb_tip_to_palm):
            return False
        if thumb.tip_to_wrist < self.cfg.thumb_tip_to_wrist:
            return False
        if (wrist.y - thumb_tip.y) <= (self.cfg.thumb_above_wrist * scale):
            return False
        if (knuckle_y - thumb_tip.y) <= (self.cfg.thumb_above_knuckles * scale):
            return False
        if (highest_fingertip_y - thumb_tip.y) <= (self.cfg.thumb_above_fingertips_ratio * palm):
            return False
        if not (thumb_tip.y < thumb_ip.y < thumb_mcp.y):
            return False

        dx = thumb_tip.x - thumb_mcp.x
        dy = thumb_tip.y - thumb_mcp.y
        if dy >= 0 or abs(dy) < self.cfg.thumb_vertical_ratio * max(abs(dx), 1e-6):
            return False

        if min_thumb_tip_distance_to_fingers(lm, use_3d=self.cfg.use_3d, scale=scale) < self.cfg.thumb_pinch_exclusion:
            return False
        if self._pinky_visibly_out(lm, scale=scale, palm=palm):
            return False

        fingers = [
            finger_metrics(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, use_3d=self.cfg.use_3d, scale=scale),
            finger_metrics(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, use_3d=self.cfg.use_3d, scale=scale),
            finger_metrics(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, use_3d=self.cfg.use_3d, scale=scale),
            finger_metrics(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, use_3d=self.cfg.use_3d, scale=scale),
        ]
        if sum(1 for metrics in fingers if finger_is_extended(metrics)) > self.cfg.max_extended:
            return False

        curled = [
            self._finger_compact(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, scale=scale),
            self._finger_compact(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, scale=scale),
            self._finger_compact(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, scale=scale),
            self._finger_compact(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, scale=scale),
        ]
        return (
            sum(1 for v in curled if v) >= self.cfg.min_curled
            and self._non_thumb_tip_cluster_ok(lm, palm=palm)
        )


_DEFAULT = BravoDetector()

def detect_bravo(landmarks: Any) -> bool:
    return _DEFAULT.detect(landmarks)
