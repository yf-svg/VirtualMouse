from __future__ import annotations

from dataclasses import dataclass

from app.control.actions import ActionIntent, no_action
from app.control.clutch import ClutchController, ClutchOut
from app.control.cursor_preview import CursorPreviewController, CursorPreviewOut
from app.control.cursor_space import CursorPoint
from app.control.primary_interaction import PrimaryInteractionController, PrimaryInteractionOut
from app.control.secondary_interaction import SecondaryInteractionController, SecondaryInteractionOut
from app.control.scroll_mode import ScrollModeController, ScrollOut


# Intentionally empty until gesture->action semantics are explicitly approved.
GENERAL_DRY_RUN_BINDINGS: dict[str, str] = {}


@dataclass(frozen=True)
class GeneralModeOut:
    intent: ActionIntent
    clutch: ClutchOut
    scroll: ScrollOut
    primary: PrimaryInteractionOut
    secondary: SecondaryInteractionOut
    cursor: CursorPreviewOut


def map_gesture_to_action(gesture_label: str | None) -> ActionIntent:
    if gesture_label is None:
        return no_action(reason="no_eligible_gesture")

    action_name = GENERAL_DRY_RUN_BINDINGS.get(gesture_label)
    if action_name is None:
        return no_action(
            reason=f"unmapped_gesture:{gesture_label}",
            gesture_label=gesture_label,
        )

    return ActionIntent(
        action_name=action_name,
        gesture_label=gesture_label,
        executable=False,
        reason="dry_run_only",
    )


def resolve_general_action(
    *,
    gesture_label: str | None,
    cursor_point: CursorPoint | None,
    clutch_controller: ClutchController,
    scroll_controller: ScrollModeController,
    primary_controller: PrimaryInteractionController,
    secondary_controller: SecondaryInteractionController,
    cursor_controller: CursorPreviewController,
    now: float,
) -> GeneralModeOut:
    clutch = clutch_controller.update(
        gesture_label=gesture_label,
        cursor_point=cursor_point,
        now=now,
    )
    if clutch.owns_state:
        scroll_controller.reset()
        primary_controller.reset()
        secondary_controller.reset()
        cursor = cursor_controller.update(
            gesture_label=gesture_label,
            cursor_point=cursor_point,
            higher_priority_owned=True,
            now=now,
        )
        return GeneralModeOut(
            intent=clutch.intent,
            clutch=clutch,
            scroll=scroll_controller.update(
                gesture_label=None,
                cursor_point=None,
                now=now,
            ),
            primary=primary_controller.update(
                gesture_label=None,
                cursor_point=None,
                now=now,
            ),
            secondary=secondary_controller.update(
                gesture_label=None,
                cursor_point=None,
                now=now,
            ),
            cursor=cursor,
        )

    scroll = scroll_controller.update(
        gesture_label=gesture_label,
        cursor_point=cursor_point,
        now=now,
    )
    if scroll.owns_state:
        primary_controller.reset()
        secondary_controller.reset()
        cursor = cursor_controller.update(
            gesture_label=gesture_label,
            cursor_point=cursor_point,
            higher_priority_owned=True,
            now=now,
        )
        return GeneralModeOut(
            intent=scroll.intent,
            clutch=clutch,
            scroll=scroll,
            primary=primary_controller.update(
                gesture_label=None,
                cursor_point=None,
                now=now,
            ),
            secondary=secondary_controller.update(
                gesture_label=None,
                cursor_point=None,
                now=now,
            ),
            cursor=cursor,
        )

    primary = primary_controller.update(
        gesture_label=gesture_label,
        cursor_point=cursor_point,
        now=now,
    )
    if primary.owns_state:
        secondary_controller.reset()
        cursor = cursor_controller.update(
            gesture_label=gesture_label,
            cursor_point=cursor_point,
            higher_priority_owned=True,
            now=now,
        )
        return GeneralModeOut(
            intent=primary.intent,
            clutch=clutch,
            scroll=scroll,
            primary=primary,
            secondary=secondary_controller.update(
                gesture_label=None,
                cursor_point=None,
                now=now,
            ),
            cursor=cursor,
        )

    secondary = secondary_controller.update(
        gesture_label=gesture_label,
        cursor_point=cursor_point,
        now=now,
    )
    if secondary.owns_state or secondary.intent.action_name != "NO_ACTION":
        cursor = cursor_controller.update(
            gesture_label=gesture_label,
            cursor_point=cursor_point,
            higher_priority_owned=True,
            now=now,
        )
        return GeneralModeOut(
            intent=secondary.intent,
            clutch=clutch,
            scroll=scroll,
            primary=primary,
            secondary=secondary,
            cursor=cursor,
        )

    cursor = cursor_controller.update(
        gesture_label=gesture_label,
        cursor_point=cursor_point,
        higher_priority_owned=False,
        now=now,
    )
    if cursor.owns_state:
        return GeneralModeOut(
            intent=cursor.intent,
            clutch=clutch,
            scroll=scroll,
            primary=primary,
            secondary=secondary,
            cursor=cursor,
        )

    return GeneralModeOut(
        intent=map_gesture_to_action(gesture_label),
        clutch=clutch,
        scroll=scroll,
        primary=primary,
        secondary=secondary,
        cursor=cursor,
    )
