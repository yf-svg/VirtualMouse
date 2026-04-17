from __future__ import annotations

from dataclasses import dataclass, replace

from app.config import CONFIG, ExecutionConfig, OperatorOverrideConfig
from app.control.window_watch import PresentationContext

_VALID_EXECUTION_OVERRIDES = frozenset({
    "inherit",
    "dry_run",
    "live",
    "disable",
    "cursor_test",
    "fallback_live",
})
_VALID_ROUTING_OVERRIDES = frozenset({"auto", "force_general", "force_presentation"})


@dataclass(frozen=True)
class RouteOverrideDecision:
    presentation_allowed: bool
    reason: str
    override_mode: str

    def summary(self) -> str:
        return f"{self.override_mode}:{self.reason}"


@dataclass(frozen=True)
class ResolvedOperatorOverridePolicy:
    valid: bool
    reason: str
    execution_override: str
    routing_override: str
    effective_execution: ExecutionConfig

    def status_text(self) -> str:
        core = f"EXEC:{self.execution_override.upper()}|ROUTE:{self.routing_override.upper()}"
        if self.valid:
            return core
        return f"FAILSAFE:{self.reason}|{core}"

    def route_presentation(self, context: PresentationContext) -> RouteOverrideDecision:
        if self.routing_override == "force_general":
            return RouteOverrideDecision(
                presentation_allowed=False,
                reason="forced_general",
                override_mode=self.routing_override,
            )
        if self.routing_override == "force_presentation":
            if context.allowed:
                return RouteOverrideDecision(
                    presentation_allowed=True,
                    reason="forced_presentation",
                    override_mode=self.routing_override,
                )
            return RouteOverrideDecision(
                presentation_allowed=False,
                reason=f"forced_presentation_blocked:{context.reason}",
                override_mode=self.routing_override,
            )

        if context.allowed:
            return RouteOverrideDecision(
                presentation_allowed=True,
                reason="context_allowed",
                override_mode="auto",
            )
        return RouteOverrideDecision(
            presentation_allowed=False,
            reason=f"context_denied:{context.reason}",
            override_mode="auto",
        )


def resolve_operator_override_policy(
    *,
    override_cfg: OperatorOverrideConfig | None = None,
    execution_cfg: ExecutionConfig | None = None,
) -> ResolvedOperatorOverridePolicy:
    override_cfg = override_cfg or CONFIG.operator_override
    execution_cfg = execution_cfg or CONFIG.execution

    reasons: list[str] = []
    execution_override = str(getattr(override_cfg, "execution_override", "inherit")).strip().lower()
    routing_override = str(getattr(override_cfg, "routing_override", "auto")).strip().lower()

    if execution_override not in _VALID_EXECUTION_OVERRIDES:
        reasons.append("invalid_execution_override")
        execution_override = "dry_run"

    if routing_override not in _VALID_ROUTING_OVERRIDES:
        reasons.append("invalid_routing_override")
        routing_override = "auto"

    effective_execution = _apply_execution_override(
        execution_cfg=execution_cfg,
        execution_override=execution_override,
    )
    return ResolvedOperatorOverridePolicy(
        valid=not reasons,
        reason="ok" if not reasons else ",".join(reasons),
        execution_override=execution_override,
        routing_override=routing_override,
        effective_execution=effective_execution,
    )


def _apply_execution_override(
    *,
    execution_cfg: ExecutionConfig,
    execution_override: str,
) -> ExecutionConfig:
    if execution_override == "inherit":
        return execution_cfg
    if execution_override == "live":
        return replace(execution_cfg, profile="live")
    if execution_override == "cursor_test":
        return replace(
            execution_cfg,
            profile="live",
            enable_live_os=True,
            enable_live_cursor=True,
            enable_live_primary=False,
            enable_live_secondary=False,
            enable_live_scroll=False,
            enable_live_presentation=False,
        )
    if execution_override == "fallback_live":
        return replace(
            execution_cfg,
            profile="live",
            enable_live_os=True,
            enable_live_cursor=True,
            enable_live_primary=True,
            enable_live_secondary=True,
            enable_live_scroll=True,
            enable_live_presentation=True,
        )

    return replace(
        execution_cfg,
        profile="dry_run",
        enable_live_os=False,
        enable_live_cursor=False,
        enable_live_primary=False,
        enable_live_secondary=False,
        enable_live_scroll=False,
        enable_live_presentation=False,
    )
