from __future__ import annotations

# Canonical gesture labels used across validation, runtime, and dataset tooling.

AUTH_LABEL_ORDER = (
    "FIST",
    "CLOSED_PALM",
    "BRAVO",
    "THUMBS_DOWN",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
)

OPS_LABEL_ORDER = (
    "PINCH_IMRP",
    "PINCH_IM",
    "PINCH_INDEX",
    "PINCH_MIDDLE",
    "PINCH_RING",
    "PINCH_PINKY",
    "FIST",
    "CLOSED_PALM",
    "OPEN_PALM",
    "SHAKA",
    "PEACE_SIGN",
    "L",
    "BRAVO",
    "THUMBS_DOWN",
    "POINT_RIGHT",
    "POINT_LEFT",
)

PRESENTATION_LABEL_ORDER = (
    "POINT_RIGHT",
    "POINT_LEFT",
    "OPEN_PALM",
    "PEACE_SIGN",
)

UNIFIED_LABEL_ORDER = (
    "FIST",
    "CLOSED_PALM",
    "BRAVO",
    "THUMBS_DOWN",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "OPEN_PALM",
    "SHAKA",
    "PEACE_SIGN",
    "L",
    "POINT_RIGHT",
    "POINT_LEFT",
    "PINCH_INDEX",
    "PINCH_MIDDLE",
    "PINCH_RING",
    "PINCH_PINKY",
    "PINCH_IM",
    "PINCH_IMRP",
)

AUTH_LABELS = frozenset(
    AUTH_LABEL_ORDER
)

OPS_LABELS = frozenset(
    OPS_LABEL_ORDER
)

PRESENTATION_LABELS = frozenset(
    PRESENTATION_LABEL_ORDER
)

ALL_ALLOWED_LABELS = frozenset(AUTH_LABELS | OPS_LABELS)
