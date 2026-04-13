from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG, ExecutionSafetyConfig
from app.control.primary_interaction import PrimaryInteractionState


@dataclass(frozen=True)
class ExecutionSafetyDecision:
    allow_cursor: bool
    cursor_reason: str
    allow_primary: bool
    primary_reason: str
    cancel_primary_drag: bool
    allow_secondary: bool
    secondary_reason: str
    allow_scroll: bool
    scroll_reason: str

    def summary(self) -> str:
        parts: list[str] = []
        for label, allowed, reason in (
            ("CUR", self.allow_cursor, self.cursor_reason),
            ("PRI", self.allow_primary, self.primary_reason),
            ("SEC", self.allow_secondary, self.secondary_reason),
            ("SCR", self.allow_scroll, self.scroll_reason),
        ):
            if not allowed:
                parts.append(f"{label}:{reason}")
        return "ok" if not parts else "|".join(parts)


@dataclass(frozen=True)
class PresentationSafetyDecision:
    allow: bool
    reason: str


class ExecutionSafetyGate:
    """
    Runtime safety seam between dry-run decisions and OS execution.
    It may suppress unsafe live actions, but it must not invent gestures or
    reinterpret interaction semantics.
    """

    def __init__(self, cfg: ExecutionSafetyConfig | None = None):
        self.cfg = cfg or CONFIG.execution_safety
        self.reset()

    def reset(self) -> None:
        self._primary_click_tainted = False

    def evaluate(
        self,
        *,
        suite_out,
        general_out,
        hand_present: bool,
    ) -> ExecutionSafetyDecision:
        feature_ok = suite_out is not None and getattr(suite_out, "feature_reason", None) == "ok"

        if general_out.primary.state in {
            PrimaryInteractionState.PRIMARY_PINCH_CANDIDATE,
            PrimaryInteractionState.CLICK_PENDING,
            PrimaryInteractionState.HAND_LOST_SAFE,
        }:
            if self.cfg.suppress_on_feature_instability and not feature_ok:
                self._primary_click_tainted = True
            if self.cfg.suppress_on_hand_loss and not hand_present:
                self._primary_click_tainted = True

        allow_cursor = True
        cursor_reason = "ok"
        if self.cfg.suppress_on_hand_loss and not hand_present:
            allow_cursor = False
            cursor_reason = "suppressed_hand_loss"
        elif self.cfg.suppress_on_feature_instability and not feature_ok:
            allow_cursor = False
            cursor_reason = "suppressed_unstable_prediction"

        allow_primary = True
        primary_reason = "ok"
        cancel_primary_drag = False
        primary_action = general_out.primary.intent.action_name
        primary_state = general_out.primary.state

        if primary_action in {"PRIMARY_CLICK", "PRIMARY_DOUBLE_CLICK"} and self._primary_click_tainted:
            allow_primary = False
            primary_reason = "suppressed_tainted_click"
            self._primary_click_tainted = False
        elif primary_action in {"PRIMARY_DRAG_START", "PRIMARY_DRAG_HOLD"}:
            if self.cfg.suppress_on_hand_loss and not hand_present:
                allow_primary = False
                primary_reason = "suppressed_hand_loss"
                cancel_primary_drag = True
            elif self.cfg.suppress_on_feature_instability and not feature_ok:
                allow_primary = False
                primary_reason = "suppressed_unstable_prediction"
                cancel_primary_drag = True
        elif primary_state == PrimaryInteractionState.HAND_LOST_SAFE:
            allow_primary = False
            primary_reason = "suppressed_hand_loss"
            cancel_primary_drag = True
        elif primary_action in {"PRIMARY_CLICK", "PRIMARY_DOUBLE_CLICK"}:
            if self.cfg.suppress_on_hand_loss and not hand_present:
                allow_primary = False
                primary_reason = "suppressed_hand_loss"
            elif self.cfg.suppress_on_feature_instability and not feature_ok:
                allow_primary = False
                primary_reason = "suppressed_unstable_prediction"
            self._primary_click_tainted = False
        elif primary_state == PrimaryInteractionState.NEUTRAL and primary_action == "NO_ACTION":
            self._primary_click_tainted = False

        allow_secondary = True
        secondary_reason = "ok"
        if general_out.secondary.intent.action_name == "SECONDARY_RIGHT_CLICK":
            if self.cfg.suppress_on_hand_loss and not hand_present:
                allow_secondary = False
                secondary_reason = "suppressed_hand_loss"
            elif self.cfg.suppress_on_feature_instability and not feature_ok:
                allow_secondary = False
                secondary_reason = "suppressed_unstable_prediction"

        allow_scroll = True
        scroll_reason = "ok"
        if general_out.scroll.intent.action_name in {"SCROLL_VERTICAL", "SCROLL_HORIZONTAL"}:
            if self.cfg.suppress_on_hand_loss and not hand_present:
                allow_scroll = False
                scroll_reason = "suppressed_hand_loss"
            elif self.cfg.suppress_on_feature_instability and not feature_ok:
                allow_scroll = False
                scroll_reason = "suppressed_unstable_prediction"

        return ExecutionSafetyDecision(
            allow_cursor=allow_cursor,
            cursor_reason=cursor_reason,
            allow_primary=allow_primary,
            primary_reason=primary_reason,
            cancel_primary_drag=cancel_primary_drag,
            allow_secondary=allow_secondary,
            secondary_reason=secondary_reason,
            allow_scroll=allow_scroll,
            scroll_reason=scroll_reason,
        )

    def evaluate_presentation(
        self,
        *,
        suite_out,
        presentation_out,
        hand_present: bool,
    ) -> PresentationSafetyDecision:
        if presentation_out.intent.action_name == "NO_ACTION":
            return PresentationSafetyDecision(False, presentation_out.intent.reason)

        feature_ok = suite_out is not None and getattr(suite_out, "feature_reason", None) == "ok"
        if self.cfg.suppress_on_hand_loss and not hand_present:
            return PresentationSafetyDecision(False, "suppressed_hand_loss")
        if self.cfg.suppress_on_feature_instability and not feature_ok:
            return PresentationSafetyDecision(False, "suppressed_unstable_prediction")
        if not presentation_out.context.allowed or not presentation_out.context.confident:
            return PresentationSafetyDecision(False, f"context_not_allowed:{presentation_out.context.reason}")
        return PresentationSafetyDecision(True, "ok")
