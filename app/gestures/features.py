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


@dataclass(frozen=True)
class HandInputQuality:
    """
    Lightweight gate for rejecting frames that are too small or unstable to feed
    into downstream gesture recognition or ML feature extraction.
    """
    passed: bool
    reason: str
    scale: float
    palm_width: float
    bbox_width: float
    bbox_height: float


@dataclass(frozen=True)
class FeatureVector:
    """
    Fixed-length feature vector for ML and offline dataset capture.
    """
    values: tuple[float, ...]
    schema_version: str = "phase3.v2"

    @property
    def dimension(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class FeatureSchema:
    """
    Explicit fixed-length schema contract for downstream ML and dataset capture.
    """
    version: str
    names: tuple[str, ...]

    @property
    def dimension(self) -> int:
        return len(self.names)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "dimension": self.dimension,
            "names": list(self.names),
        }


@dataclass(frozen=True)
class GeometricFeatures:
    """
    Explicit distance- and angle-based feature block for the hand.
    Distances are normalized so the representation is user- and scale-stable.
    """
    angle_names: tuple[str, ...]
    angle_values: tuple[float, ...]
    distance_names: tuple[str, ...]
    distance_values: tuple[float, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return self.angle_names + self.distance_names

    @property
    def values(self) -> tuple[float, ...]:
        return self.angle_values + self.distance_values


@dataclass(frozen=True)
class NormalizedHandLandmarks:
    """
    Stable Phase 3 representation of the detected hand.
    Coordinates are anchored at the palm center and normalized by hand scale.
    """
    values: tuple[float, ...]
    anchor_x: float
    anchor_y: float
    anchor_z: float
    scale: float

    @property
    def dimension(self) -> int:
        return len(self.values)

    def as_tuple(self) -> tuple[float, ...]:
        return self.values


@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class FeatureExtractionCfg:
    use_3d: bool = True
    min_scale: float = 1e-4
    min_quality_scale: float = 0.05
    min_quality_palm_width: float = 0.04
    min_bbox_width: float = 0.08
    min_bbox_height: float = 0.12


DEFAULT_FEATURE_CFG = FeatureExtractionCfg()
FEATURE_SCHEMA_VERSION = "phase3.v2"


def get_landmarks(landmarks: Any):
    if isinstance(landmarks, dict) and "landmarks" in landmarks:
        return get_landmarks(landmarks["landmarks"])
    if hasattr(landmarks, "landmarks"):
        return get_landmarks(landmarks.landmarks)
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


def palm_width(lm, use_3d: bool = True, min_scale: float = 1e-4) -> float:
    return max(distance(lm[INDEX_MCP], lm[PINKY_MCP], use_3d), min_scale)


def palm_center(lm):
    pts = [lm[WRIST], lm[INDEX_MCP], lm[MIDDLE_MCP], lm[RING_MCP], lm[PINKY_MCP]]
    x = sum(p.x for p in pts) / len(pts)
    y = sum(p.y for p in pts) / len(pts)
    z = sum(getattr(p, "z", 0.0) for p in pts) / len(pts)
    return Point3(x=x, y=y, z=z)


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


def landmark_bbox(lm) -> tuple[float, float]:
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    return max(xs) - min(xs), max(ys) - min(ys)


def normalized_landmark_names() -> tuple[str, ...]:
    labels = (
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    )
    names: list[str] = []
    for label in labels:
        names.extend((f"{label}_x_norm", f"{label}_y_norm", f"{label}_z_norm"))
    return tuple(names)


def extract_normalized_hand_landmarks(
    landmarks: Any,
    *,
    use_3d: bool = True,
    min_scale: float = 1e-4,
) -> NormalizedHandLandmarks:
    """
    Translation- and scale-invariant normalized landmark coordinates.
    The palm center is used as anchor so the representation is more stable than
    using an edge landmark such as the wrist.
    """
    lm = get_landmarks(landmarks)
    scale = hand_scale(lm, use_3d=use_3d, min_scale=min_scale)
    center = palm_center(lm)

    values: list[float] = []
    for p in lm:
        values.append((p.x - center.x) / scale)
        values.append((p.y - center.y) / scale)
        values.append(((getattr(p, "z", 0.0) - center.z) / scale) if use_3d else 0.0)
    return NormalizedHandLandmarks(
        values=tuple(values),
        anchor_x=center.x,
        anchor_y=center.y,
        anchor_z=center.z,
        scale=scale,
    )


def normalize_landmarks(
    landmarks: Any,
    *,
    use_3d: bool = True,
    min_scale: float = 1e-4,
) -> tuple[float, ...]:
    """
    Backwards-compatible flattened normalized landmarks helper.
    """
    return extract_normalized_hand_landmarks(
        landmarks,
        use_3d=use_3d,
        min_scale=min_scale,
    ).as_tuple()


def assess_hand_input_quality(
    landmarks: Any,
    *,
    cfg: FeatureExtractionCfg | None = None,
) -> HandInputQuality:
    """
    Reject obviously weak detections before temporal logic or ML consume them.
    This is intentionally conservative so it suppresses only clearly bad frames.
    """
    cfg = cfg or DEFAULT_FEATURE_CFG
    lm = get_landmarks(landmarks)

    try:
        coords = [_xyz(p) for p in lm]
    except Exception:
        return HandInputQuality(False, "invalid_landmarks", 0.0, 0.0, 0.0, 0.0)

    if len(coords) < 21:
        return HandInputQuality(False, "incomplete_landmarks", 0.0, 0.0, 0.0, 0.0)

    if not all(math.isfinite(v) for xyz in coords for v in xyz):
        return HandInputQuality(False, "non_finite_landmarks", 0.0, 0.0, 0.0, 0.0)

    scale = hand_scale(lm, use_3d=cfg.use_3d, min_scale=cfg.min_scale)
    p_width = palm_width(lm, use_3d=cfg.use_3d, min_scale=cfg.min_scale)
    bbox_w, bbox_h = landmark_bbox(lm)

    if scale < cfg.min_quality_scale:
        return HandInputQuality(False, "hand_too_small", scale, p_width, bbox_w, bbox_h)
    if p_width < cfg.min_quality_palm_width:
        return HandInputQuality(False, "palm_too_narrow", scale, p_width, bbox_w, bbox_h)
    if bbox_w < cfg.min_bbox_width and bbox_h < cfg.min_bbox_height:
        return HandInputQuality(False, "bbox_too_small", scale, p_width, bbox_w, bbox_h)

    return HandInputQuality(True, "ok", scale, p_width, bbox_w, bbox_h)


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


def geometric_angle_names() -> tuple[str, ...]:
    return (
        "thumb_mcp_angle", "thumb_ip_angle",
        "index_pip_angle", "index_dip_angle",
        "middle_pip_angle", "middle_dip_angle",
        "ring_pip_angle", "ring_dip_angle",
        "pinky_pip_angle", "pinky_dip_angle",
    )


def geometric_distance_names() -> tuple[str, ...]:
    return (
        "thumb_tip_to_palm", "index_tip_to_palm", "middle_tip_to_palm", "ring_tip_to_palm", "pinky_tip_to_palm",
        "thumb_tip_to_wrist", "index_tip_to_wrist", "middle_tip_to_wrist", "ring_tip_to_wrist", "pinky_tip_to_wrist",
        "thumb_index_tip_gap", "index_middle_tip_gap", "middle_ring_tip_gap", "ring_pinky_tip_gap", "thumb_pinky_tip_gap",
        "extended_count", "curled_count", "near_palm_count", "min_thumb_tip_distance",
    )


def extract_geometric_features(
    landmarks: Any,
    *,
    cfg: FeatureExtractionCfg | None = None,
) -> GeometricFeatures:
    """
    Dedicated geometric feature extractor for Phase 3.
    """
    cfg = cfg or DEFAULT_FEATURE_CFG
    lm = get_landmarks(landmarks)
    scale = hand_scale(lm, use_3d=cfg.use_3d, min_scale=cfg.min_scale)
    p_width = palm_width(lm, use_3d=cfg.use_3d, min_scale=cfg.min_scale)
    summary = summarize_hand_pose(lm, use_3d=cfg.use_3d, min_scale=cfg.min_scale)

    index = summary.fingers["I"]
    middle = summary.fingers["M"]
    ring = summary.fingers["R"]
    pinky = summary.fingers["P"]

    angle_values = (
        summary.thumb.mcp_angle,
        summary.thumb.ip_angle,
        index.pip_angle,
        index.dip_angle,
        middle.pip_angle,
        middle.dip_angle,
        ring.pip_angle,
        ring.dip_angle,
        pinky.pip_angle,
        pinky.dip_angle,
    )
    distance_values = (
        summary.thumb.tip_to_palm,
        index.tip_to_palm,
        middle.tip_to_palm,
        ring.tip_to_palm,
        pinky.tip_to_palm,
        summary.thumb.tip_to_wrist,
        index.tip_to_wrist,
        middle.tip_to_wrist,
        ring.tip_to_wrist,
        pinky.tip_to_wrist,
        distance(lm[THUMB_TIP], lm[INDEX_TIP], cfg.use_3d) / p_width,
        distance(lm[INDEX_TIP], lm[MIDDLE_TIP], cfg.use_3d) / p_width,
        distance(lm[MIDDLE_TIP], lm[RING_TIP], cfg.use_3d) / p_width,
        distance(lm[RING_TIP], lm[PINKY_TIP], cfg.use_3d) / p_width,
        distance(lm[THUMB_TIP], lm[PINKY_TIP], cfg.use_3d) / p_width,
        float(summary.extended_count),
        float(summary.curled_count),
        float(summary.near_palm_count),
        summary.min_thumb_tip_distance,
    )
    return GeometricFeatures(
        angle_names=geometric_angle_names(),
        angle_values=angle_values,
        distance_names=geometric_distance_names(),
        distance_values=distance_values,
    )


def feature_names() -> tuple[str, ...]:
    names: list[str] = list(normalized_landmark_names())
    names.extend(geometric_angle_names())
    names.extend(geometric_distance_names())
    return tuple(names)


def feature_schema() -> FeatureSchema:
    return FEATURE_SCHEMA


def feature_dimension() -> int:
    return feature_schema().dimension


def extract_feature_vector(
    landmarks: Any,
    *,
    cfg: FeatureExtractionCfg | None = None,
) -> FeatureVector:
    """
    Fixed-length representation for downstream ML:
    - normalized landmark coordinates
    - joint angles
    - normalized distances to palm/wrist
    - fingertip gap ratios and coarse hand-state summary
    """
    cfg = cfg or DEFAULT_FEATURE_CFG
    lm = get_landmarks(landmarks)
    normalized = extract_normalized_hand_landmarks(
        lm,
        use_3d=cfg.use_3d,
        min_scale=cfg.min_scale,
    )
    geometry = extract_geometric_features(lm, cfg=cfg)
    values = normalized.values + geometry.values
    schema = FEATURE_SCHEMA
    if len(values) != schema.dimension:
        raise ValueError(
            f"Feature vector dimension mismatch: got {len(values)}, expected {schema.dimension}"
        )
    return FeatureVector(values=values, schema_version=schema.version)


FEATURE_SCHEMA = FeatureSchema(
    version=FEATURE_SCHEMA_VERSION,
    names=feature_names(),
)
FEATURE_DIMENSION = FEATURE_SCHEMA.dimension
