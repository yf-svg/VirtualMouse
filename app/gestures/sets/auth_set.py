# app/gestures/sets/auth_set.py

AUTH_ALLOWED = {"ONE", "TWO", "THREE", "FOUR", "FIVE", "BRAVO", "THUMBS_DOWN", "FIST", "CLOSED_PALM"}

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
