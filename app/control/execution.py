from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.config import CONFIG, ExecutionConfig
from app.control.cursor_preview import CursorPreviewOut, CursorPreviewState
from app.control.cursor_space import CursorPoint
from app.control.execution_safety import ExecutionSafetyDecision
from app.control.keyboard import KeyboardBackend, NoOpKeyboardBackend, WindowsKeyboardBackend
from app.control.mouse import MouseBackend, NoOpMouseBackend, ScreenPoint, WindowsMouseBackend
from app.control.primary_interaction import PrimaryInteractionOut
from app.control.scroll_mode import ScrollOut
from app.control.secondary_interaction import SecondaryInteractionOut

if TYPE_CHECKING:
    from app.modes.general import GeneralModeOut
    from app.modes.presentation import PresentationModeOut


@dataclass(frozen=True)
class ExecutionReport:
    performed: bool
    action_name: str | None
    reason: str
    target: ScreenPoint | None
    live_enabled: bool


@dataclass(frozen=True)
class ExecutionBatchReport:
    cursor: ExecutionReport
    primary: ExecutionReport
    secondary: ExecutionReport
    scroll: ExecutionReport


@dataclass(frozen=True)
class ExecutionNeutralizationReport:
    reason: str
    released_primary_drag: bool
    keyboard_release_sent: bool


@dataclass(frozen=True)
class ResolvedExecutionPolicy:
    mode: str
    valid: bool
    reason: str
    live_master_enabled: bool
    cursor_enabled: bool
    primary_enabled: bool
    secondary_enabled: bool
    scroll_enabled: bool
    presentation_enabled: bool

    def status_text(self) -> str:
        if not self.live_master_enabled:
            return f"DRY:{self.reason}"
        parts: list[str] = []
        if self.cursor_enabled:
            parts.append("cur")
        if self.primary_enabled:
            parts.append("pri")
        if self.secondary_enabled:
            parts.append("sec")
        if self.scroll_enabled:
            parts.append("scr")
        if self.presentation_enabled:
            parts.append("prs")
        enabled = ",".join(parts) if parts else "-"
        return f"LIVE:{enabled}"


class OSActionExecutor:
    """
    OS side-effect adapter behind the dry-run interaction contract.
    It consumes dry-run outputs only and can be globally disabled.
    """

    def __init__(
        self,
        *,
        cfg: ExecutionConfig | None = None,
        mouse_backend: MouseBackend | None = None,
        keyboard_backend: KeyboardBackend | None = None,
    ):
        self.cfg = cfg or CONFIG.execution
        self.policy = self._resolve_policy(self.cfg)
        if mouse_backend is not None:
            self.mouse_backend = mouse_backend
        elif self.policy.live_master_enabled:
            self.mouse_backend = WindowsMouseBackend()
        else:
            self.mouse_backend = NoOpMouseBackend()
        if keyboard_backend is not None:
            self.keyboard_backend = keyboard_backend
        elif self.policy.live_master_enabled:
            self.keyboard_backend = WindowsKeyboardBackend()
        else:
            self.keyboard_backend = NoOpKeyboardBackend()
        self._last_cursor_target: ScreenPoint | None = None
        self._primary_drag_down = False

    def policy_status_text(self) -> str:
        return self.policy.status_text()

    def current_cursor_normalized(self) -> CursorPoint | None:
        width, height = self.mouse_backend.screen_size()
        if width <= 1 or height <= 1:
            return None
        point = self.mouse_backend.cursor_position_abs()
        x = min(1.0, max(0.0, float(point.x) / float(width - 1)))
        y = min(1.0, max(0.0, float(point.y) / float(height - 1)))
        return CursorPoint(x=x, y=y)

    def neutralize(self, *, reason: str) -> ExecutionNeutralizationReport:
        released_primary_drag = False
        if self._primary_drag_down:
            try:
                self.mouse_backend.left_button_up()
            except OSError:
                released_primary_drag = False
            else:
                self._primary_drag_down = False
                released_primary_drag = True
        self._last_cursor_target = None
        self.keyboard_backend.release_all()
        return ExecutionNeutralizationReport(
            reason=reason,
            released_primary_drag=released_primary_drag,
            keyboard_release_sent=True,
        )

    def apply_general_mode(
        self,
        out: GeneralModeOut,
        *,
        safety: ExecutionSafetyDecision | None = None,
    ) -> ExecutionBatchReport:
        safety = safety or ExecutionSafetyDecision(
            allow_cursor=True,
            cursor_reason="ok",
            allow_primary=True,
            primary_reason="ok",
            cancel_primary_drag=False,
            allow_secondary=True,
            secondary_reason="ok",
            allow_scroll=True,
            scroll_reason="ok",
        )
        return ExecutionBatchReport(
            cursor=self.apply_cursor_preview(out.cursor, allow=safety.allow_cursor, suppress_reason=safety.cursor_reason),
            primary=self.apply_primary_interaction(
                out.primary,
                higher_priority_owned=out.clutch.owns_state or out.scroll.owns_state,
                allow=safety.allow_primary,
                suppress_reason=safety.primary_reason,
                cancel_active_drag=safety.cancel_primary_drag,
            ),
            secondary=self.apply_secondary_interaction(
                out.secondary,
                allow=safety.allow_secondary,
                suppress_reason=safety.secondary_reason,
            ),
            scroll=self.apply_scroll_output(
                out.scroll,
                allow=safety.allow_scroll,
                suppress_reason=safety.scroll_reason,
            ),
        )

    def apply_presentation_mode(
        self,
        out: PresentationModeOut,
        *,
        allow: bool,
        suppress_reason: str,
    ) -> ExecutionReport:
        live_enabled = self.policy.live_master_enabled and self.policy.presentation_enabled
        if not live_enabled:
            reason = self.policy.reason if not self.policy.live_master_enabled else "presentation_disabled_by_policy"
            return ExecutionReport(False, out.intent.action_name, reason, None, False)
        if not allow:
            return ExecutionReport(False, out.intent.action_name, suppress_reason, None, True)

        key_name = self._presentation_key(out.intent.action_name)
        if key_name is None:
            return ExecutionReport(False, out.intent.action_name, "presentation_no_action", None, True)

        self.keyboard_backend.press_key(key_name)
        return ExecutionReport(True, out.intent.action_name, f"presentation_{key_name.lower()}_emitted", None, True)

    def apply_cursor_preview(
        self,
        out: CursorPreviewOut,
        *,
        allow: bool = True,
        suppress_reason: str = "ok",
    ) -> ExecutionReport:
        live_enabled = self.policy.live_master_enabled and self.policy.cursor_enabled
        if not live_enabled:
            reason = self.policy.reason if not self.policy.live_master_enabled else "cursor_disabled_by_policy"
            return ExecutionReport(False, out.intent.action_name, reason, None, False)
        if not allow:
            self._last_cursor_target = None
            return ExecutionReport(False, out.intent.action_name, suppress_reason, None, True)

        if out.state != CursorPreviewState.CURSOR_ACTIVE or not out.owns_state:
            self._last_cursor_target = None
            return ExecutionReport(False, out.intent.action_name, "cursor_not_owned", None, True)

        if not out.policy.eligible or out.preview_point is None:
            return ExecutionReport(False, out.intent.action_name, "cursor_policy_not_eligible", None, True)

        if out.intent.action_name not in {"CURSOR_PREVIEW_MOVE", "CURSOR_PREVIEW_READY", "CURSOR_PREVIEW_HOLD"}:
            return ExecutionReport(False, out.intent.action_name, "cursor_action_not_approved", None, True)

        if out.intent.action_name != "CURSOR_PREVIEW_MOVE":
            return ExecutionReport(False, out.intent.action_name, "cursor_no_move_required", None, True)

        try:
            target, moved = self._move_cursor_from_normalized(out.preview_point.x, out.preview_point.y)
        except OSError as exc:
            return self._backend_error_report("cursor", out.intent.action_name, exc)
        if not moved:
            return ExecutionReport(False, out.intent.action_name, "cursor_target_unchanged", target, True)

        return ExecutionReport(True, out.intent.action_name, "cursor_moved", target, True)

    def apply_primary_interaction(
        self,
        out: PrimaryInteractionOut,
        *,
        higher_priority_owned: bool = False,
        allow: bool = True,
        suppress_reason: str = "ok",
        cancel_active_drag: bool = False,
    ) -> ExecutionReport:
        live_enabled = self.policy.live_master_enabled and self.policy.primary_enabled
        if not live_enabled:
            reason = self.policy.reason if not self.policy.live_master_enabled else "primary_disabled_by_policy"
            return ExecutionReport(False, out.intent.action_name, reason, None, False)

        action_name = out.intent.action_name
        approved = {
            "NO_ACTION",
            "PRIMARY_CLICK",
            "PRIMARY_DOUBLE_CLICK",
            "PRIMARY_DRAG_START",
            "PRIMARY_DRAG_HOLD",
            "PRIMARY_DRAG_END",
        }
        if action_name not in approved:
            return ExecutionReport(False, action_name, "primary_action_not_approved", None, True)

        if not allow:
            if self._primary_drag_down and cancel_active_drag:
                self.mouse_backend.left_button_up()
                self._primary_drag_down = False
                return ExecutionReport(True, action_name, "primary_drag_safety_release", None, True)
            return ExecutionReport(False, action_name, suppress_reason, None, True)

        if self._primary_drag_down and (
            higher_priority_owned
            or (not out.owns_state and action_name not in {"PRIMARY_DRAG_START", "PRIMARY_DRAG_HOLD", "PRIMARY_DRAG_END"})
        ):
            try:
                self.mouse_backend.left_button_up()
            except OSError as exc:
                return self._backend_error_report("primary", action_name, exc)
            self._primary_drag_down = False
            return ExecutionReport(True, action_name, "primary_drag_cancelled", None, True)

        if action_name == "PRIMARY_CLICK":
            try:
                self.mouse_backend.left_click()
            except OSError as exc:
                return self._backend_error_report("primary", action_name, exc)
            return ExecutionReport(True, action_name, "primary_click_emitted", None, True)

        if action_name == "PRIMARY_DOUBLE_CLICK":
            try:
                self.mouse_backend.double_left_click()
            except OSError as exc:
                return self._backend_error_report("primary", action_name, exc)
            return ExecutionReport(True, action_name, "primary_double_click_emitted", None, True)

        if action_name == "PRIMARY_DRAG_START":
            target = None
            moved = False
            try:
                if out.cursor_point is not None:
                    target, moved = self._move_cursor_from_normalized(out.cursor_point.x, out.cursor_point.y)
            except OSError as exc:
                return self._backend_error_report("primary", action_name, exc)
            if not self._primary_drag_down:
                try:
                    self.mouse_backend.left_button_down()
                except OSError as exc:
                    return self._backend_error_report("primary", action_name, exc, target=target)
                self._primary_drag_down = True
                return ExecutionReport(True, action_name, "primary_drag_started", target, True)
            if moved:
                return ExecutionReport(True, action_name, "primary_drag_moved", target, True)
            return ExecutionReport(False, action_name, "primary_drag_already_active", target, True)

        if action_name == "PRIMARY_DRAG_HOLD":
            if not self._primary_drag_down:
                return ExecutionReport(False, action_name, "primary_drag_hold_without_press", None, True)
            if out.cursor_point is None:
                return ExecutionReport(False, action_name, "primary_drag_hold_no_cursor", None, True)
            try:
                target, moved = self._move_cursor_from_normalized(out.cursor_point.x, out.cursor_point.y)
            except OSError as exc:
                return self._backend_error_report("primary", action_name, exc)
            if not moved:
                return ExecutionReport(False, action_name, "primary_drag_holding", target, True)
            return ExecutionReport(True, action_name, "primary_drag_moved", target, True)

        if action_name == "PRIMARY_DRAG_END":
            target = None
            try:
                if out.cursor_point is not None:
                    target, _ = self._move_cursor_from_normalized(out.cursor_point.x, out.cursor_point.y)
            except OSError as exc:
                return self._backend_error_report("primary", action_name, exc)
            if not self._primary_drag_down:
                return ExecutionReport(False, action_name, "primary_drag_end_without_press", target, True)
            try:
                self.mouse_backend.left_button_up()
            except OSError as exc:
                return self._backend_error_report("primary", action_name, exc, target=target)
            self._primary_drag_down = False
            return ExecutionReport(True, action_name, "primary_drag_ended", target, True)

        return ExecutionReport(False, action_name, "primary_no_action", None, True)

    def apply_secondary_interaction(
        self,
        out: SecondaryInteractionOut,
        *,
        allow: bool = True,
        suppress_reason: str = "ok",
    ) -> ExecutionReport:
        live_enabled = self.policy.live_master_enabled and self.policy.secondary_enabled
        if not live_enabled:
            reason = self.policy.reason if not self.policy.live_master_enabled else "secondary_disabled_by_policy"
            return ExecutionReport(False, out.intent.action_name, reason, None, False)

        action_name = out.intent.action_name
        if action_name not in {"NO_ACTION", "SECONDARY_RIGHT_CLICK"}:
            return ExecutionReport(False, action_name, "secondary_action_not_approved", None, True)
        if not allow:
            return ExecutionReport(False, action_name, suppress_reason, None, True)

        if action_name != "SECONDARY_RIGHT_CLICK":
            return ExecutionReport(False, action_name, "secondary_no_action", None, True)

        try:
            self.mouse_backend.right_click()
        except OSError as exc:
            return self._backend_error_report("secondary", action_name, exc)
        return ExecutionReport(True, action_name, "secondary_right_click_emitted", None, True)

    def apply_scroll_output(
        self,
        out: ScrollOut,
        *,
        allow: bool = True,
        suppress_reason: str = "ok",
    ) -> ExecutionReport:
        live_enabled = self.policy.live_master_enabled and self.policy.scroll_enabled
        if not live_enabled:
            reason = self.policy.reason if not self.policy.live_master_enabled else "scroll_disabled_by_policy"
            return ExecutionReport(False, out.intent.action_name, reason, None, False)

        action_name = out.intent.action_name
        approved = {
            "NO_ACTION",
            "SCROLL_MODE_ENTER",
            "SCROLL_MODE_EXIT",
            "SCROLL_VERTICAL",
            "SCROLL_HORIZONTAL",
        }
        if action_name not in approved:
            return ExecutionReport(False, action_name, "scroll_action_not_approved", None, True)
        if not allow:
            return ExecutionReport(False, action_name, suppress_reason, None, True)

        if action_name == "SCROLL_VERTICAL":
            amount = self._to_scroll_units(out.movement)
            if amount == 0:
                return ExecutionReport(False, action_name, "scroll_units_zero", None, True)
            try:
                self.mouse_backend.scroll_vertical(amount)
            except OSError as exc:
                return self._backend_error_report("scroll", action_name, exc)
            return ExecutionReport(True, action_name, "scroll_vertical_emitted", None, True)

        if action_name == "SCROLL_HORIZONTAL":
            amount = self._to_scroll_units(out.movement)
            if amount == 0:
                return ExecutionReport(False, action_name, "scroll_units_zero", None, True)
            try:
                self.mouse_backend.scroll_horizontal(amount)
            except OSError as exc:
                return self._backend_error_report("scroll", action_name, exc)
            return ExecutionReport(True, action_name, "scroll_horizontal_emitted", None, True)

        if action_name == "SCROLL_MODE_ENTER":
            return ExecutionReport(False, action_name, "scroll_mode_enter_no_os_action", None, True)
        if action_name == "SCROLL_MODE_EXIT":
            return ExecutionReport(False, action_name, "scroll_mode_exit_no_os_action", None, True)
        return ExecutionReport(False, action_name, "scroll_no_action", None, True)

    def _move_cursor_from_normalized(self, x_norm: float, y_norm: float) -> tuple[ScreenPoint, bool]:
        target = self._to_screen_point(x_norm, y_norm)
        if self._last_cursor_target == target:
            return target, False
        self.mouse_backend.move_cursor_abs(target)
        self._last_cursor_target = target
        return target, True

    @staticmethod
    def _backend_error_report(
        subsystem: str,
        action_name: str | None,
        exc: OSError,
        *,
        target: ScreenPoint | None = None,
    ) -> ExecutionReport:
        code = getattr(exc, "errno", None)
        reason = f"{subsystem}_backend_error" if code in (None, 0) else f"{subsystem}_backend_error:{code}"
        return ExecutionReport(False, action_name, reason, target, True)

    def _to_screen_point(self, x_norm: float, y_norm: float) -> ScreenPoint:
        width, height = self.mouse_backend.screen_size()
        x = min(1.0, max(0.0, float(x_norm)))
        y = min(1.0, max(0.0, float(y_norm)))
        px = int(round(x * (width - 1)))
        py = int(round(y * (height - 1)))
        return ScreenPoint(px, py)

    def _to_scroll_units(self, movement: float) -> int:
        return int(round(float(movement) * self.cfg.scroll_units_per_motion))

    @staticmethod
    def _presentation_key(action_name: str | None) -> str | None:
        return {
            "PRESENT_NEXT": "RIGHT",
            "PRESENT_PREV": "LEFT",
            "PRESENT_START": "F5",
            "PRESENT_EXIT": "ESC",
        }.get(action_name)

    @staticmethod
    def _resolve_policy(cfg: ExecutionConfig) -> ResolvedExecutionPolicy:
        mode = str(getattr(cfg, "profile", "dry_run")).strip().lower()
        if mode not in {"dry_run", "live"}:
            return ResolvedExecutionPolicy(
                mode=mode,
                valid=False,
                reason="invalid_execution_profile",
                live_master_enabled=False,
                cursor_enabled=False,
                primary_enabled=False,
                secondary_enabled=False,
                scroll_enabled=False,
                presentation_enabled=False,
            )
        if mode == "dry_run":
            return ResolvedExecutionPolicy(
                mode=mode,
                valid=True,
                reason="dry_run_profile",
                live_master_enabled=False,
                cursor_enabled=False,
                primary_enabled=False,
                secondary_enabled=False,
                scroll_enabled=False,
                presentation_enabled=False,
            )
        if not cfg.enable_live_os:
            return ResolvedExecutionPolicy(
                mode=mode,
                valid=False,
                reason="live_profile_missing_master_enable",
                live_master_enabled=False,
                cursor_enabled=False,
                primary_enabled=False,
                secondary_enabled=False,
                scroll_enabled=False,
                presentation_enabled=False,
            )
        if not any((
            cfg.enable_live_cursor,
            cfg.enable_live_primary,
            cfg.enable_live_secondary,
            cfg.enable_live_scroll,
            cfg.enable_live_presentation,
        )):
            return ResolvedExecutionPolicy(
                mode=mode,
                valid=False,
                reason="live_profile_without_subsystems",
                live_master_enabled=False,
                cursor_enabled=False,
                primary_enabled=False,
                secondary_enabled=False,
                scroll_enabled=False,
                presentation_enabled=False,
            )
        return ResolvedExecutionPolicy(
            mode=mode,
            valid=True,
            reason="live_profile_enabled",
            live_master_enabled=True,
            cursor_enabled=bool(cfg.enable_live_cursor),
            primary_enabled=bool(cfg.enable_live_primary),
            secondary_enabled=bool(cfg.enable_live_secondary),
            scroll_enabled=bool(cfg.enable_live_scroll),
            presentation_enabled=bool(cfg.enable_live_presentation),
        )
