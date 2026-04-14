from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from app.control.actions import ActionIntent, no_action
from app.control.window_watch import PresentationAppKind, PresentationContext


@dataclass(frozen=True)
class PresentationModeOut:
    intent: ActionIntent
    context: PresentationContext


@dataclass(frozen=True)
class PresentationActionBinding:
    action_name: str
    capability: str
    playback_only: bool = True


PRESENTATION_PLAYBACK_BINDINGS: Mapping[str, PresentationActionBinding] = {
    "POINT_RIGHT": PresentationActionBinding("PRESENT_NEXT", "navigation"),
    "POINT_LEFT": PresentationActionBinding("PRESENT_PREV", "navigation"),
    "OPEN_PALM": PresentationActionBinding("PRESENT_START", "start"),
    "PEACE_SIGN": PresentationActionBinding("PRESENT_EXIT", "exit"),
}

PRESENTATION_PLAYBACK_GESTURES = frozenset(PRESENTATION_PLAYBACK_BINDINGS)
PRESENTATION_PLAYBACK_ACTIONS = frozenset(
    binding.action_name for binding in PRESENTATION_PLAYBACK_BINDINGS.values()
)


def resolve_presentation_action(
    *,
    gesture_label: str | None,
    context: PresentationContext,
) -> PresentationModeOut:
    if gesture_label is None:
        return PresentationModeOut(
            intent=no_action(reason="no_eligible_gesture", mode="PRESENTATION"),
            context=context,
        )

    binding = PRESENTATION_PLAYBACK_BINDINGS.get(gesture_label)
    if binding is None:
        return PresentationModeOut(
            intent=no_action(
                reason=f"unmapped_gesture:{gesture_label}",
                gesture_label=gesture_label,
                mode="PRESENTATION",
            ),
            context=context,
        )

    if not context.allowed or not context.confident:
        return PresentationModeOut(
            intent=no_action(
                reason=f"context_not_allowed:{context.reason}",
                gesture_label=gesture_label,
                mode="PRESENTATION",
            ),
            context=context,
        )

    if not binding.playback_only:
        return PresentationModeOut(
            intent=no_action(
                reason=f"presentation_action_scope_blocked:{binding.action_name}",
                gesture_label=gesture_label,
                mode="PRESENTATION",
            ),
            context=context,
        )

    if binding.capability == "navigation" and not context.navigation_allowed:
        return PresentationModeOut(
            intent=no_action(
                reason=f"presentation_navigation_blocked:{context.reason}",
                gesture_label=gesture_label,
                mode="PRESENTATION",
            ),
            context=context,
        )

    if binding.capability == "start" and not context.supports_start:
        return PresentationModeOut(
            intent=no_action(
                reason=f"presentation_start_blocked:{context.reason}",
                gesture_label=gesture_label,
                mode="PRESENTATION",
            ),
            context=context,
        )

    if binding.capability == "exit" and not context.supports_exit:
        return PresentationModeOut(
            intent=no_action(
                reason=f"presentation_exit_blocked:{context.reason}",
                gesture_label=gesture_label,
                mode="PRESENTATION",
            ),
            context=context,
        )

    return PresentationModeOut(
        intent=ActionIntent(
            action_name=binding.action_name,
            gesture_label=gesture_label,
            executable=False,
            reason="dry_run_only",
            mode="PRESENTATION",
        ),
        context=context,
    )


def map_gesture_to_action(gesture_label: str | None) -> ActionIntent:
    return resolve_presentation_action(
        gesture_label=gesture_label,
        context=PresentationContext(
            allowed=False,
            confident=False,
            kind=PresentationAppKind.NONE,
            process_name=None,
            window_title=None,
            fullscreen_like=False,
            navigation_allowed=False,
            supports_start=False,
            supports_exit=False,
            reason="no_context",
        ),
    ).intent
