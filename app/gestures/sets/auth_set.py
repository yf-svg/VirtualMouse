# app/gestures/sets/auth_set.py

AUTH_ALLOWED = {"ONE", "TWO", "THREE", "FOUR", "FIVE", "BRAVO", "FIST"}

# Highest priority first
AUTH_PRIORITY = [
    "FIST",    # cancel/reset
    "BRAVO",   # confirm
    "FIVE",
    "FOUR",
    "THREE",
    "TWO",
    "ONE",
]