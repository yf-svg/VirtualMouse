# app/gestures/sets/auth_set.py

from app.gestures.sets.labels import AUTH_LABELS

AUTH_ALLOWED = set(AUTH_LABELS)

# Highest priority first
AUTH_PRIORITY = [
    "FIST",    # cancel/reset
    "CLOSED_PALM",
    "BRAVO",   # confirm
    "THUMBS_DOWN",
    "FIVE",
    "FOUR",
    "THREE",
    "TWO",
    "ONE",
]
