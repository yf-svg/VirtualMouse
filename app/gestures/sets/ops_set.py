# app/gestures/sets/ops_set.py

from app.gestures.sets.labels import OPS_LABELS

OPS_ALLOWED = set(OPS_LABELS)

OPS_PRIORITY = [
    "PINCH_IMRP", "PINCH_IM",
    "PINCH_INDEX", "PINCH_MIDDLE", "PINCH_RING", "PINCH_PINKY",
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
]

# Live ops runtime is allowed to resolve overlaps by priority so higher-value
# interaction gestures survive to the controllers instead of being nulled out
# as ambiguous.
OPS_RUNTIME_ALLOW_PRIORITY = True


def ops_runtime_suite_kwargs() -> dict[str, object]:
    return {
        "allowed": OPS_ALLOWED,
        "priority": OPS_PRIORITY,
        "allow_priority": OPS_RUNTIME_ALLOW_PRIORITY,
    }
