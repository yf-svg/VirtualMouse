from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG, CursorPolicyConfig


@dataclass(frozen=True)
class CursorPolicyDecision:
    eligible: bool
    gesture_label: str | None
    reason: str
    provisional: bool


class CursorPolicy:
    """
    Isolated cursor-ownership policy.
    This keeps the current cursor-pose assumption localized so it can be
    changed later without rewriting the movement/output controller stack.
    """

    def __init__(self, cfg: CursorPolicyConfig | None = None):
        self.cfg = cfg or CONFIG.cursor_policy

    def evaluate(self, gesture_label: str | None) -> CursorPolicyDecision:
        if gesture_label is None:
            return CursorPolicyDecision(
                eligible=False,
                gesture_label=None,
                reason="no_eligible_gesture",
                provisional=self.cfg.provisional,
            )

        if gesture_label in self.cfg.allowed_gestures:
            return CursorPolicyDecision(
                eligible=True,
                gesture_label=gesture_label,
                reason="cursor_policy_match",
                provisional=self.cfg.provisional,
            )

        return CursorPolicyDecision(
            eligible=False,
            gesture_label=gesture_label,
            reason=f"cursor_policy_reject:{gesture_label}",
            provisional=self.cfg.provisional,
        )
