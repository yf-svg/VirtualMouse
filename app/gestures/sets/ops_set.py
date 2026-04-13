# app/gestures/sets/ops_set.py

from app.gestures.sets.labels import OPS_LABELS

OPS_ALLOWED = set(OPS_LABELS)

OPS_PRIORITY = [
    "PINCH_IMRP", "PINCH_IM",
    "PINCH_INDEX", "PINCH_MIDDLE", "PINCH_RING", "PINCH_PINKY",
    "FIST",
    "CLOSED_PALM",
    "OPEN_PALM",
    "PEACE_SIGN",
    "L",
    "BRAVO",
    "THUMBS_DOWN",
    "POINT_RIGHT",
    "POINT_LEFT",
]
