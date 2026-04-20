from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.config import CONFIG, PresentationToolConfig
from app.gestures.features import INDEX_MCP, INDEX_TIP, get_landmarks, palm_center


@dataclass(frozen=True)
class CursorPoint:
    """
    Abstract cursor-space point derived from hand landmarks.
    This is intentionally gesture-agnostic so interaction logic does not depend
    on a final cursor-pose decision.
    """

    x: float
    y: float


def cursor_point_from_landmarks(
    landmarks: Any,
    *,
    anchor_mode: str = "palm_center",
    mirror_x: bool = False,
) -> CursorPoint | None:
    lm = get_landmarks(landmarks)
    if lm is None:
        return None

    if anchor_mode == "palm_center":
        point = palm_center(lm)
    elif anchor_mode == "index_tip":
        point = lm[INDEX_TIP]
    else:
        raise ValueError(f"Unsupported cursor anchor mode: {anchor_mode}")

    x = float(point.x)
    if mirror_x:
        x = 1.0 - x
    return CursorPoint(x=x, y=float(point.y))


def cursor_distance(a: CursorPoint, b: CursorPoint) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def remap_cursor_point(
    point: CursorPoint,
    *,
    x_min: float = 0.0,
    x_max: float = 1.0,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> CursorPoint:
    return CursorPoint(
        x=_remap_axis(point.x, low=x_min, high=x_max),
        y=_remap_axis(point.y, low=y_min, high=y_max),
    )


def presentation_pointer_point_from_landmarks(
    landmarks: Any,
    *,
    cfg: PresentationToolConfig | None = None,
    mirror_x: bool = False,
) -> CursorPoint | None:
    cfg = cfg or CONFIG.presentation_tools
    anchor_mode = str(getattr(cfg, "pointer_anchor_mode", "index_tip") or "index_tip")
    point = _presentation_anchor_point(
        landmarks,
        anchor_mode=anchor_mode,
        cfg=cfg,
        mirror_x=mirror_x,
    )
    if point is None:
        return None
    return remap_cursor_point(
        point,
        x_min=float(getattr(cfg, "pointer_input_x_min", 0.0) or 0.0),
        x_max=float(getattr(cfg, "pointer_input_x_max", 1.0) or 1.0),
        y_min=float(getattr(cfg, "pointer_input_y_min", 0.0) or 0.0),
        y_max=float(getattr(cfg, "pointer_input_y_max", 1.0) or 1.0),
    )


def _presentation_anchor_point(
    landmarks: Any,
    *,
    anchor_mode: str,
    cfg: PresentationToolConfig,
    mirror_x: bool,
) -> CursorPoint | None:
    lm = get_landmarks(landmarks)
    if lm is None:
        return None
    if anchor_mode != "index_tip":
        return cursor_point_from_landmarks(lm, anchor_mode=anchor_mode, mirror_x=mirror_x)
    tip = lm[INDEX_TIP]
    mcp = lm[INDEX_MCP]
    raw_x = float(tip.x)
    raw_y = float(tip.y)
    blend_margin = max(0.0, float(getattr(cfg, "pointer_edge_blend_margin", 0.0) or 0.0))
    blend_pull = max(0.0, min(1.0, float(getattr(cfg, "pointer_edge_blend_max_pull", 0.0) or 0.0)))
    if blend_margin > 0.0 and blend_pull > 0.0:
        edge_distance = min(raw_x, 1.0 - raw_x, raw_y, 1.0 - raw_y)
        if edge_distance < blend_margin:
            ratio = min(1.0, max(0.0, (blend_margin - edge_distance) / blend_margin))
            pull = blend_pull * ratio
            raw_x = raw_x + ((float(mcp.x) - raw_x) * pull)
            raw_y = raw_y + ((float(mcp.y) - raw_y) * pull)
    if mirror_x:
        raw_x = 1.0 - raw_x
    return CursorPoint(x=raw_x, y=raw_y)


def _remap_axis(value: float, *, low: float, high: float) -> float:
    value = float(value)
    low = float(low)
    high = float(high)
    span = high - low
    if abs(span) < 1e-6:
        return min(1.0, max(0.0, value))
    normalized = (value - low) / span
    return min(1.0, max(0.0, normalized))
