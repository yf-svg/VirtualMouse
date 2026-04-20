from __future__ import annotations

from app.constants import AppState
from app.control.window_watch import PresentationContext


def _format_execution_policy_status(ctx, safety_summary: str, *, route_summary: str | None = None) -> str:
    parts = [
        f"OVR:{ctx.override_policy.status_text()}",
        f"XPOL:{ctx.executor.policy_status_text()}",
        f"SAFE:{safety_summary}",
    ]
    if route_summary is not None:
        parts.append(f"ROUTE:{route_summary}")
    return "|".join(parts)


def _format_presentation_context(context: PresentationContext) -> str:
    return f"PCTX:{context.summary()}"


def _format_presentation_tools(tool_out) -> str:
    return tool_out.status_text()


def _format_presentation_tool_execution(report) -> str:
    if report.visible and report.laser_point is not None:
        prefix = f"TOOLEX:{report.reason}"
        if getattr(report, "stroke_count", 0):
            prefix += f":S{report.stroke_count}"
        return f"{prefix}@{report.laser_point.x},{report.laser_point.y}"
    if report.visible and getattr(report, "draw_point", None) is not None:
        prefix = f"TOOLEX:{report.reason}"
        if getattr(report, "stroke_count", 0):
            prefix += f":S{report.stroke_count}"
        point = report.draw_point
        return f"{prefix}@{point.x},{point.y}"
    if getattr(report, "stroke_count", 0):
        return f"TOOLEX:{report.reason}:S{report.stroke_count}"
    return f"TOOLEX:{report.reason}"


def _format_mode_status(state: AppState, *, route_summary: str | None = None) -> str:
    if state == AppState.ACTIVE_PRESENTATION:
        if route_summary and route_summary.startswith("force_presentation:"):
            return "MODE:PRESENTATION_FORCED"
        return "MODE:PRESENTATION_AUTO"
    if state == AppState.ACTIVE_GENERAL:
        if route_summary and route_summary.startswith("force_general:"):
            return "MODE:GENERAL_FORCED"
        return "MODE:GENERAL_READY"
    if state == AppState.AUTHENTICATING:
        return "MODE:AUTH_ENTRY"
    if state == AppState.IDLE_LOCKED:
        return "MODE:LOCKED"
    if state == AppState.SLEEP:
        return "MODE:SLEEP"
    if state == AppState.EXITING:
        return "MODE:EXITING"
    return f"MODE:{state.value}"


def _format_cursor_runtime(cursor_out, cursor_report) -> str:
    policy = getattr(cursor_out, "policy", None)
    gesture_label = getattr(policy, "gesture_label", None) or "-"
    policy_reason = getattr(policy, "reason", "no_cursor_policy")
    policy_ready = "ready" if bool(getattr(policy, "eligible", False)) else "blocked"
    state = getattr(getattr(cursor_out, "state", None), "value", None) or str(getattr(cursor_out, "state", "UNKNOWN"))
    return f"CUR:{state}({gesture_label}/{policy_ready}:{policy_reason})|CUREX:{cursor_report.reason}"


def _format_general_controls(ctx) -> str:
    move_labels = getattr(getattr(ctx.cursor_preview, "policy", None), "cfg", None)
    move_display = ",".join(getattr(move_labels, "allowed_gestures", ()) or ("-",))
    scroll_toggle = getattr(ctx.scroll_mode, "toggle_label", "-")
    return (
        f"CTRL:MOVE={move_display}"
        f"|CLK=PINCH_INDEX"
        f"|RMB=PINCH_MIDDLE"
        f"|SCR={scroll_toggle}"
        f"|HOLD=FIST"
    )
