from __future__ import annotations

from dataclasses import dataclass

from app.control.actions import ActionIntent, no_action
from app.control.window_watch import PresentationAppKind, PresentationContext


@dataclass(frozen=True)
class PresentationModeOut:
    intent: ActionIntent
    context: PresentationContext


PRESENTATION_DRY_RUN_BINDINGS: dict[str, str] = {
    "POINT_RIGHT": "PRESENT_NEXT",
    "POINT_LEFT": "PRESENT_PREV",
    "OPEN_PALM": "PRESENT_START",
    "FIST": "PRESENT_EXIT",
}


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

    action_name = PRESENTATION_DRY_RUN_BINDINGS.get(gesture_label)
    if action_name is None:
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

    if action_name in {"PRESENT_NEXT", "PRESENT_PREV"} and not context.navigation_allowed:
        return PresentationModeOut(
            intent=no_action(
                reason=f"presentation_navigation_blocked:{context.reason}",
                gesture_label=gesture_label,
                mode="PRESENTATION",
            ),
            context=context,
        )

    if action_name == "PRESENT_START" and not context.supports_start:
        return PresentationModeOut(
            intent=no_action(
                reason=f"presentation_start_blocked:{context.reason}",
                gesture_label=gesture_label,
                mode="PRESENTATION",
            ),
            context=context,
        )

    if action_name == "PRESENT_EXIT" and not context.supports_exit:
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
            action_name=action_name,
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
