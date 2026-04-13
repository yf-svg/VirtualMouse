from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionIntent:
    action_name: str
    gesture_label: str | None
    executable: bool
    reason: str
    mode: str = "GENERAL"


def no_action(*, reason: str, gesture_label: str | None = None, mode: str = "GENERAL") -> ActionIntent:
    return ActionIntent(
        action_name="NO_ACTION",
        gesture_label=gesture_label,
        executable=False,
        reason=reason,
        mode=mode,
    )


def dry_run_action(
    action_name: str,
    *,
    gesture_label: str | None,
    reason: str,
    mode: str = "GENERAL",
) -> ActionIntent:
    return ActionIntent(
        action_name=action_name,
        gesture_label=gesture_label,
        executable=False,
        reason=reason,
        mode=mode,
    )


def format_action_intent(intent: ActionIntent) -> str:
    if intent.action_name == "NO_ACTION":
        if intent.gesture_label is None:
            return f"ACT:NONE({intent.reason})"
        return f"ACT:PENDING({intent.gesture_label})"
    suffix = "DRYRUN" if not intent.executable else "LIVE"
    return f"ACT:{intent.action_name}[{suffix}]"


class SystemActions:
    def __init__(self):
        raise NotImplementedError("OS action execution is not implemented yet.")
