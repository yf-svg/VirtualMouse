from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# MediaPipe landmark indices
WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def get_landmarks(landmarks: Any):
    if isinstance(landmarks, dict) and "landmarks" in landmarks:
        return get_landmarks(landmarks["landmarks"])
    return landmarks.landmark if hasattr(landmarks, "landmark") else landmarks


def _xyz(p) -> tuple[float, float, float]:
    return (p.x, p.y, getattr(p, "z", 0.0))


def distance(a, b, use_3d: bool = True) -> float:
    ax, ay, az = _xyz(a)
    bx, by, bz = _xyz(b)
    dx = ax - bx
    dy = ay - by
    if not use_3d:
        return math.hypot(dx, dy)
    dz = az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def hand_scale(lm, use_3d: bool = True, min_scale: float = 1e-4) -> float:
    spans = [
        distance(lm[WRIST], lm[INDEX_MCP], use_3d),
        distance(lm[WRIST], lm[PINKY_MCP], use_3d),
        distance(lm[INDEX_MCP], lm[PINKY_MCP], use_3d),
    ]
    scale = sum(spans) / len(spans)
    return max(scale, min_scale)


def palm_center(lm):
    pts = [lm[WRIST], lm[INDEX_MCP], lm[MIDDLE_MCP], lm[RING_MCP], lm[PINKY_MCP]]
    x = sum(p.x for p in pts) / len(pts)
    y = sum(p.y for p in pts) / len(pts)
    z = sum(getattr(p, "z", 0.0) for p in pts) / len(pts)
    return type("P", (), {"x": x, "y": y, "z": z})()


def angle(a, b, c, use_3d: bool = True) -> float:
    ax, ay, az = _xyz(a)
    bx, by, bz = _xyz(b)
    cx, cy, cz = _xyz(c)

    ab = (ax - bx, ay - by, (az - bz) if use_3d else 0.0)
    cb = (cx - bx, cy - by, (cz - bz) if use_3d else 0.0)

    nab = math.sqrt(sum(v * v for v in ab))
    ncb = math.sqrt(sum(v * v for v in cb))
    if nab < 1e-9 or ncb < 1e-9:
        return 0.0

    dot = sum(u * v for u, v in zip(ab, cb))
    cosv = max(-1.0, min(1.0, dot / (nab * ncb)))
    return math.degrees(math.acos(cosv))


@dataclass(frozen=True)
class FingerMetrics:
    pip_angle: float
    dip_angle: float
    tip_to_palm: float
    tip_to_wrist: float
    pip_to_wrist: float

    @property
    def curl_ratio(self) -> float:
        return self.tip_to_wrist / max(self.pip_to_wrist, 1e-6)


def finger_metrics(
    lm,
    mcp: int,
    pip: int,
    dip: int,
    tip: int,
    *,
    use_3d: bool = True,
    scale: float | None = None,
) -> FingerMetrics:
    s = scale or hand_scale(lm, use_3d=use_3d)
    palm = palm_center(lm)
    return FingerMetrics(
        pip_angle=angle(lm[mcp], lm[pip], lm[dip], use_3d=use_3d),
        dip_angle=angle(lm[pip], lm[dip], lm[tip], use_3d=use_3d),
        tip_to_palm=distance(lm[tip], palm, use_3d=use_3d) / s,
        tip_to_wrist=distance(lm[tip], lm[WRIST], use_3d=use_3d) / s,
        pip_to_wrist=distance(lm[pip], lm[WRIST], use_3d=use_3d) / s,
    )


def finger_is_extended(
    metrics: FingerMetrics,
    *,
    min_pip_angle: float = 160.0,
    min_dip_angle: float = 150.0,
    min_tip_to_palm: float = 0.78,
    min_curl_ratio: float = 1.08,
) -> bool:
    return (
        metrics.pip_angle >= min_pip_angle
        and metrics.dip_angle >= min_dip_angle
        and metrics.tip_to_palm >= min_tip_to_palm
        and metrics.curl_ratio >= min_curl_ratio
    )


def finger_is_curled(
    metrics: FingerMetrics,
    *,
    max_pip_angle: float = 125.0,
    max_dip_angle: float = 145.0,
    max_tip_to_palm: float = 0.72,
    max_curl_ratio: float = 1.03,
) -> bool:
    return (
        metrics.pip_angle <= max_pip_angle
        and metrics.dip_angle <= max_dip_angle
        and metrics.tip_to_palm <= max_tip_to_palm
        and metrics.curl_ratio <= max_curl_ratio
    )


@dataclass(frozen=True)
class ThumbMetrics:
    mcp_angle: float
    ip_angle: float
    tip_to_palm: float
    tip_to_wrist: float


def thumb_metrics(lm, *, use_3d: bool = True, scale: float | None = None) -> ThumbMetrics:
    s = scale or hand_scale(lm, use_3d=use_3d)
    palm = palm_center(lm)
    return ThumbMetrics(
        mcp_angle=angle(lm[THUMB_CMC], lm[THUMB_MCP], lm[THUMB_IP], use_3d=use_3d),
        ip_angle=angle(lm[THUMB_MCP], lm[THUMB_IP], lm[THUMB_TIP], use_3d=use_3d),
        tip_to_palm=distance(lm[THUMB_TIP], palm, use_3d=use_3d) / s,
        tip_to_wrist=distance(lm[THUMB_TIP], lm[WRIST], use_3d=use_3d) / s,
    )


def thumb_is_extended(
    metrics: ThumbMetrics,
    *,
    min_mcp_angle: float = 135.0,
    min_ip_angle: float = 150.0,
    min_tip_to_palm: float = 0.62,
) -> bool:
    return (
        metrics.mcp_angle >= min_mcp_angle
        and metrics.ip_angle >= min_ip_angle
        and metrics.tip_to_palm >= min_tip_to_palm
    )


def min_thumb_tip_distance_to_fingers(lm, *, use_3d: bool = True, scale: float | None = None) -> float:
    s = scale or hand_scale(lm, use_3d=use_3d)
    thumb = lm[THUMB_TIP]
    dists = [
        distance(thumb, lm[INDEX_TIP], use_3d=use_3d),
        distance(thumb, lm[MIDDLE_TIP], use_3d=use_3d),
        distance(thumb, lm[RING_TIP], use_3d=use_3d),
        distance(thumb, lm[PINKY_TIP], use_3d=use_3d),
    ]
    return min(dists) / s


@dataclass(frozen=True)
class HandPoseSummary:
    scale: float
    fingers: dict[str, FingerMetrics]
    thumb: ThumbMetrics
    extended_count: int
    curled_count: int
    near_palm_count: int
    thumb_extended: bool
    min_thumb_tip_distance: float


def summarize_hand_pose(lm, *, use_3d: bool = True, min_scale: float = 1e-4) -> HandPoseSummary:
    scale = hand_scale(lm, use_3d=use_3d, min_scale=min_scale)
    fingers = {
        "I": finger_metrics(lm, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, use_3d=use_3d, scale=scale),
        "M": finger_metrics(lm, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, use_3d=use_3d, scale=scale),
        "R": finger_metrics(lm, RING_MCP, RING_PIP, RING_DIP, RING_TIP, use_3d=use_3d, scale=scale),
        "P": finger_metrics(lm, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, use_3d=use_3d, scale=scale),
    }
    thumb = thumb_metrics(lm, use_3d=use_3d, scale=scale)
    extended_count = sum(1 for m in fingers.values() if finger_is_extended(m, min_tip_to_palm=0.74, min_curl_ratio=1.05))
    curled_count = sum(1 for m in fingers.values() if finger_is_curled(m, max_tip_to_palm=0.78, max_curl_ratio=1.05))
    near_palm_count = sum(1 for m in fingers.values() if m.tip_to_palm <= 0.82)
    return HandPoseSummary(
        scale=scale,
        fingers=fingers,
        thumb=thumb,
        extended_count=extended_count,
        curled_count=curled_count,
        near_palm_count=near_palm_count,
        thumb_extended=thumb_is_extended(thumb, min_tip_to_palm=0.58),
        min_thumb_tip_distance=min_thumb_tip_distance_to_fingers(lm, use_3d=use_3d, scale=scale),
    )
