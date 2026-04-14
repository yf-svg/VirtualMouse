from __future__ import annotations

from dataclasses import dataclass

from app.security.auth import GestureAuthCfg, GestureAuthOut

_DIGIT_LABELS = {
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
}


@dataclass(frozen=True)
class AuthOverlayState:
    current_sequence: tuple[str, ...]
    total_required: int
    detected_gesture: str | None
    auth_status: str
    expected_next: str | None
    status_text: str
    retry_after_s: float | None
    freeze_sequence: bool

    @property
    def display_digits(self) -> tuple[str, ...]:
        return tuple(_DIGIT_LABELS.get(label, "?") for label in self.current_sequence)


@dataclass(slots=True)
class AuthOverlayStateStore:
    _committed_sequence: tuple[str, ...] = ()
    _locked_sequence: tuple[str, ...] = ()

    def reset(self) -> None:
        self._committed_sequence = ()
        self._locked_sequence = ()

    def build_state(
        self,
        *,
        cfg: GestureAuthCfg,
        auth_out: GestureAuthOut,
        auth_status: str,
        detected_gesture: str | None,
    ) -> AuthOverlayState:
        current_sequence, freeze_sequence = self._resolve_sequence(
            cfg=cfg,
            auth_out=auth_out,
            auth_status=auth_status,
        )
        return AuthOverlayState(
            current_sequence=current_sequence,
            total_required=len(cfg.sequence),
            detected_gesture=detected_gesture,
            auth_status=auth_status,
            expected_next=auth_out.expected_next,
            status_text=_status_text(auth_status=auth_status, auth_out=auth_out),
            retry_after_s=auth_out.retry_after_s,
            freeze_sequence=freeze_sequence,
        )

    def _resolve_sequence(
        self,
        *,
        cfg: GestureAuthCfg,
        auth_out: GestureAuthOut,
        auth_status: str,
    ) -> tuple[tuple[str, ...], bool]:
        total_required = len(cfg.sequence)
        accepted_count = min(auth_out.matched_steps, total_required)
        accepted_sequence = tuple(cfg.sequence[:accepted_count])

        if auth_status == "success":
            self._locked_sequence = ()
            self._committed_sequence = tuple(cfg.sequence)
            return self._committed_sequence, False

        if auth_status in {"started", "progress", "authenticated"}:
            self._locked_sequence = ()
            if len(accepted_sequence) >= len(self._committed_sequence):
                self._committed_sequence = accepted_sequence
            return self._committed_sequence, False

        if auth_status == "step_back":
            self._locked_sequence = ()
            self._committed_sequence = accepted_sequence
            return self._committed_sequence, False

        if auth_status == "locked_out":
            if not self._locked_sequence:
                self._locked_sequence = self._committed_sequence
            self._committed_sequence = ()
            return self._locked_sequence, True

        if auth_status in {"reset_wrong", "reset_timeout", "reset_cancel"}:
            previous_sequence = self._committed_sequence
            self._committed_sequence = ()
            self._locked_sequence = ()
            return previous_sequence, False

        self._locked_sequence = ()
        return self._committed_sequence, False


def build_auth_overlay_state(
    *,
    cfg: GestureAuthCfg,
    auth_out: GestureAuthOut,
    auth_status: str,
    detected_gesture: str | None,
) -> AuthOverlayState:
    return AuthOverlayStateStore().build_state(
        cfg=cfg,
        auth_out=auth_out,
        auth_status=auth_status,
        detected_gesture=detected_gesture,
    )


def _status_text(*, auth_status: str, auth_out: GestureAuthOut) -> str:
    if auth_status == "success":
        return "Approved"
    if auth_status == "reset_wrong":
        return "Wrong input - restarting"
    if auth_status == "reset_timeout":
        return "Timeout - try again"
    if auth_status == "reset_cancel":
        return "Reset"
    if auth_status == "locked_out":
        seconds = max(1, int((auth_out.retry_after_s or 0.0) + 0.999))
        return f"Locked ({seconds}s)"
    if auth_status == "step_back":
        return f"Waiting for: {auth_out.expected_next or 'BRAVO'}"
    if auth_status in {"started", "progress"}:
        if auth_out.expected_next is not None:
            return f"Waiting for: {auth_out.expected_next}"
        return "Approved"
    return f"Waiting for: {auth_out.expected_next or 'ONE'}"
