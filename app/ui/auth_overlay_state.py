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
    _last_visible_sequence: tuple[str, ...] = ()

    def reset(self) -> None:
        self._last_visible_sequence = ()

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
        committed_sequence = tuple(auth_out.committed_sequence)

        if auth_status == "locked_out":
            frozen = self._last_visible_sequence or committed_sequence
            return frozen, True

        self._last_visible_sequence = committed_sequence
        return committed_sequence, False


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
    if auth_status == "reset_cancel":
        return "Reset"
    if auth_status == "locked_out":
        seconds = max(1, int((auth_out.retry_after_s or 0.0) + 0.999))
        return f"Locked ({seconds}s)"
    if auth_status == "step_back":
        return f"Waiting for: {auth_out.expected_next or 'BRAVO'}"
    if auth_status == "ready_to_submit":
        return "Waiting for: BRAVO"
    if auth_status in {"started", "progress"}:
        if auth_out.expected_next is not None:
            return f"Waiting for: {auth_out.expected_next}"
        return "Approved"
    return f"Waiting for: {auth_out.expected_next or 'ONE'}"
