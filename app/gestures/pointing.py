from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import CONFIG
from app.gestures.features import (
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    finger_is_extended,
    finger_metrics,
    get_landmarks,
    hand_scale,
    palm_center,
    palm_width,
    thumb_metrics,
)


@dataclass(frozen=True)
class PointingCfg:
    use_3d: bool = True
    min_scale: float = 1e-4
    mirror_view: bool = CONFIG.mirror_view

    min_index_tip_to_palm: float = 0.85
    min_horizontal_offset_ratio: float = 0.85
    horizontal_dominance_ratio: float = 1.25
    max_vertical_offset_ratio: float = 0.75

    compact_thumb_max_tip_to_palm: float = 1.25
    compact_thumb_max_tip_gap_ratio: float = 1.40

    visible_tip_depth_ratio: float = 0.06
    hidden_tip_depth_ratio: float = 0.06
    thumb_side_margin_ratio: float = 0.08
    max_extra_extended_fingers: int = 1


@dataclass(frozen=True)
class PointingAnalysis:
    direction: str
    display_dx: float
    display_dy: float
    direction_ok: bool
    non_index_extended_count: int
    thumb_compact: bool
    handedness_orientation: str | None
    visible_count: int
    hidden_count: int
    unclear_count: int
    orientation: str | None
    matched: bool
    reason: str


def _extract_handedness(hand_input: Any) -> str | None:
    if isinstance(hand_input, dict):
        handedness = hand_input.get("handedness")
        return str(handedness) if handedness else None
    handedness = getattr(hand_input, "handedness", None)
    return str(handedness) if handedness else None


class PointingDetector:
    def __init__(self, cfg: PointingCfg = PointingCfg()):
        self.cfg = cfg

    def _display_delta(self, lm) -> tuple[float, float]:
        dx = lm[INDEX_TIP].x - lm[INDEX_MCP].x
        dy = lm[INDEX_TIP].y - lm[INDEX_MCP].y
        return ((-dx if self.cfg.mirror_view else dx), dy)

    def _index_is_pointing_sideways(self, lm, *, scale: float, direction: str) -> bool:
        index = finger_metrics(
            lm,
            INDEX_MCP,
            INDEX_PIP,
            INDEX_DIP,
            INDEX_TIP,
            use_3d=self.cfg.use_3d,
            scale=scale,
        )
        if not finger_is_extended(index):
            return False
        if index.tip_to_palm < self.cfg.min_index_tip_to_palm:
            return False

        palm = palm_width(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)
        display_dx, display_dy = self._display_delta(lm)
        if abs(display_dx) / palm < self.cfg.min_horizontal_offset_ratio:
            return False
        if abs(display_dy) / palm > self.cfg.max_vertical_offset_ratio:
            return False
        if abs(display_dx) < self.cfg.horizontal_dominance_ratio * max(abs(display_dy), 1e-6):
            return False

        if direction == "right":
            return display_dx > 0.0
        if direction == "left":
            return display_dx < 0.0
        raise ValueError(f"Unsupported direction: {direction}")

    def _thumb_is_compact(self, lm, *, scale: float) -> bool:
        thumb = thumb_metrics(lm, use_3d=self.cfg.use_3d, scale=scale)
        thumb_tip_gap_ratio = abs(lm[THUMB_TIP].x - lm[INDEX_MCP].x) / max(
            palm_width(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale),
            self.cfg.min_scale,
        )
        return (
            thumb.tip_to_palm <= self.cfg.compact_thumb_max_tip_to_palm
            and thumb_tip_gap_ratio <= self.cfg.compact_thumb_max_tip_gap_ratio
        )

    def _non_index_extended_count(self, lm, *, scale: float) -> int:
        metrics = [
            finger_metrics(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, use_3d=self.cfg.use_3d, scale=scale),
            finger_metrics(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, use_3d=self.cfg.use_3d, scale=scale),
            finger_metrics(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, use_3d=self.cfg.use_3d, scale=scale),
        ]
        return sum(1 for metric in metrics if finger_is_extended(metric))

    def _finger_depth_state(self, lm, pip: int, tip: int, *, scale: float) -> str:
        depth_ratio = (lm[tip].z - lm[pip].z) / max(scale, self.cfg.min_scale)
        if depth_ratio <= -self.cfg.visible_tip_depth_ratio:
            return "visible"
        if depth_ratio >= self.cfg.hidden_tip_depth_ratio:
            return "hidden"
        return "unclear"

    def _thumb_side_orientation(self, lm, *, handedness: str | None) -> str | None:
        if handedness not in {"Left", "Right"}:
            return None

        center = palm_center(lm)
        palm = max(palm_width(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale), self.cfg.min_scale)
        thumb_side_ratio = (lm[THUMB_TIP].x - center.x) / palm
        if abs(thumb_side_ratio) < self.cfg.thumb_side_margin_ratio:
            return None

        if handedness == "Right":
            return "palm" if thumb_side_ratio < 0.0 else "back"
        return "palm" if thumb_side_ratio > 0.0 else "back"

    def _orientation_counts(self, hand_input: Any, lm, *, scale: float) -> tuple[str | None, int, int, int]:
        finger_states = []
        for mcp, pip, dip, tip in (
            (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
            (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
            (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
        ):
            metrics = finger_metrics(lm, mcp, pip, dip, tip, use_3d=self.cfg.use_3d, scale=scale)
            if not finger_is_extended(metrics):
                finger_states.append(self._finger_depth_state(lm, pip, tip, scale=scale))

        visible_count = sum(1 for state in finger_states if state == "visible")
        hidden_count = sum(1 for state in finger_states if state == "hidden")
        unclear_count = sum(1 for state in finger_states if state == "unclear")

        if len(finger_states) < 2:
            return (None, visible_count, hidden_count, unclear_count)

        handedness_orientation = self._thumb_side_orientation(
            lm,
            handedness=_extract_handedness(hand_input),
        )
        score = visible_count - hidden_count

        if score >= 2 and unclear_count <= 1:
            return ("palm", visible_count, hidden_count, unclear_count)
        if score <= -2 and unclear_count <= 1:
            return ("back", visible_count, hidden_count, unclear_count)
        return (None, visible_count, hidden_count, unclear_count)

    def analyze(self, hand_input: Any, *, direction: str) -> PointingAnalysis:
        lm = get_landmarks(hand_input)
        scale = hand_scale(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)
        display_dx, display_dy = self._display_delta(lm)

        direction_ok = self._index_is_pointing_sideways(lm, scale=scale, direction=direction)
        non_index_extended_count = self._non_index_extended_count(lm, scale=scale)
        thumb_compact = self._thumb_is_compact(lm, scale=scale)
        handedness_orientation = self._thumb_side_orientation(
            lm,
            handedness=_extract_handedness(hand_input),
        )
        orientation, visible_count, hidden_count, unclear_count = self._orientation_counts(
            hand_input,
            lm,
            scale=scale,
        )

        if not direction_ok:
            return PointingAnalysis(
                direction=direction,
                display_dx=display_dx,
                display_dy=display_dy,
                direction_ok=False,
                non_index_extended_count=non_index_extended_count,
                thumb_compact=thumb_compact,
                handedness_orientation=handedness_orientation,
                visible_count=visible_count,
                hidden_count=hidden_count,
                unclear_count=unclear_count,
                orientation=orientation,
                matched=False,
                reason="direction",
            )
        if non_index_extended_count > self.cfg.max_extra_extended_fingers:
            return PointingAnalysis(
                direction=direction,
                display_dx=display_dx,
                display_dy=display_dy,
                direction_ok=True,
                non_index_extended_count=non_index_extended_count,
                thumb_compact=thumb_compact,
                handedness_orientation=handedness_orientation,
                visible_count=visible_count,
                hidden_count=hidden_count,
                unclear_count=unclear_count,
                orientation=orientation,
                matched=False,
                reason="extra_extended",
            )
        if not thumb_compact:
            return PointingAnalysis(
                direction=direction,
                display_dx=display_dx,
                display_dy=display_dy,
                direction_ok=True,
                non_index_extended_count=non_index_extended_count,
                thumb_compact=False,
                handedness_orientation=handedness_orientation,
                visible_count=visible_count,
                hidden_count=hidden_count,
                unclear_count=unclear_count,
                orientation=orientation,
                matched=False,
                reason="thumb",
            )

        expected_orientation = "palm" if direction == "right" else "back"
        # Palm/back inference is useful debug context, but in live camera use it is
        # not stable enough to be a hard rejection. Depth estimates can flip with
        # slight wrist roll or camera angle even when the pointing direction is clear.

        return PointingAnalysis(
            direction=direction,
            display_dx=display_dx,
            display_dy=display_dy,
            direction_ok=True,
            non_index_extended_count=non_index_extended_count,
            thumb_compact=thumb_compact,
            handedness_orientation=handedness_orientation,
            visible_count=visible_count,
            hidden_count=hidden_count,
            unclear_count=unclear_count,
            orientation=orientation,
            matched=True,
            reason="ok" if orientation == expected_orientation else "ok_shape",
        )

    def detect(self, hand_input: Any, *, direction: str) -> bool:
        return self.analyze(hand_input, direction=direction).matched


_DEFAULT = PointingDetector()


def detect_point_right(hand_input: Any) -> bool:
    return _DEFAULT.detect(hand_input, direction="right")


def detect_point_left(hand_input: Any) -> bool:
    return _DEFAULT.detect(hand_input, direction="left")
