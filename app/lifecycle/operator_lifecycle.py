from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG, OperatorLifecycleConfig
from app.constants import AppState


@dataclass(frozen=True)
class ExitRequest:
    source: str
    reason: str
    trigger: str

    def summary(self) -> str:
        return f"{self.source}:{self.trigger}"


@dataclass(frozen=True)
class RuntimeNeutralizationReport:
    reason: str
    released_primary_drag: bool
    keyboard_release_sent: bool
    controller_reset: bool
    suite_reset: bool
    safety_reset: bool

    def summary(self) -> str:
        parts: list[str] = []
        if self.released_primary_drag:
            parts.append("drag_up")
        if self.keyboard_release_sent:
            parts.append("key_release")
        if self.controller_reset:
            parts.append("owners_reset")
        if self.safety_reset:
            parts.append("safety_reset")
        return ",".join(parts) if parts else "noop"


class OperatorLifecycleController:
    def __init__(self, cfg: OperatorLifecycleConfig | None = None):
        self.cfg = cfg or CONFIG.operator_lifecycle
        self._manual_exit_keys = frozenset(
            token for token in (self._normalize_key_token(item) for item in self.cfg.manual_exit_keys) if token is not None
        )
        self._gesture_exit_label = str(getattr(self.cfg, "gesture_exit_label", "")).strip().upper()
        self._gesture_exit_enabled = bool(self.cfg.enable_gesture_exit) and bool(self._gesture_exit_label)

    def request_from_key(self, key: int) -> ExitRequest | None:
        token = self._key_token_from_code(key)
        if token is None or token not in self._manual_exit_keys:
            return None
        return ExitRequest(source="manual", reason="manual_exit_key", trigger=token)

    def request_from_suite_out(self, *, suite_out, router_state: AppState) -> ExitRequest | None:
        if not self._gesture_exit_enabled:
            return None
        if router_state not in {AppState.ACTIVE_GENERAL, AppState.ACTIVE_PRESENTATION}:
            return None
        if suite_out is None or getattr(suite_out, "down", None) != self._gesture_exit_label:
            return None
        hold_frames = int(getattr(suite_out, "hold_frames", 0) or 0)
        if hold_frames < max(1, int(self.cfg.gesture_exit_min_hold_frames)):
            return None
        return ExitRequest(source="gesture", reason="gesture_exit", trigger=self._gesture_exit_label)

    def status_text(
        self,
        *,
        request: ExitRequest | None = None,
        neutralization: RuntimeNeutralizationReport | None = None,
    ) -> str:
        if request is None:
            return "LIFE:ready"
        text = f"LIFE:exit_requested:{request.summary()}"
        if neutralization is not None:
            text = f"{text}|NEUT:{neutralization.summary()}"
        return text

    @staticmethod
    def _normalize_key_token(value: object) -> str | None:
        text = str(value).strip().upper()
        if not text:
            return None
        if text == "27":
            return "ESC"
        if len(text) == 1 and text.isprintable():
            return text
        if text == "ESCAPE":
            return "ESC"
        return text

    @classmethod
    def _key_token_from_code(cls, key: int) -> str | None:
        if key < 0 or key > 255:
            return None
        if key == 27:
            return "ESC"
        if 32 <= key <= 126:
            return chr(key).upper()
        return None


def neutralize_runtime_ownership(ctx, *, reason: str) -> RuntimeNeutralizationReport:
    ctx.auth_suite.reset()
    ctx.ops_suite.reset()
    ctx.clutch.reset()
    ctx.scroll_mode.reset()
    ctx.primary_interaction.reset()
    ctx.secondary_interaction.reset()
    ctx.cursor_preview.reset()
    ctx.execution_safety.reset()
    ctx.smoother.reset()
    execution_report = ctx.executor.neutralize(reason=reason)
    return RuntimeNeutralizationReport(
        reason=reason,
        released_primary_drag=execution_report.released_primary_drag,
        keyboard_release_sent=execution_report.keyboard_release_sent,
        controller_reset=True,
        suite_reset=True,
        safety_reset=True,
    )
