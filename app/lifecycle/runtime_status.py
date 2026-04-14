from __future__ import annotations

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
