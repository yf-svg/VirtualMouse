from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
from dataclasses import dataclass
from enum import Enum

from app.config import CONFIG, PresentationContextConfig


class PresentationAppKind(str, Enum):
    NONE = "NONE"
    POWERPOINT = "POWERPOINT"
    BROWSER_PRESENTATION = "BROWSER_PRESENTATION"
    PDF_VIEWER = "PDF_VIEWER"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class ForegroundWindowSnapshot:
    process_name: str | None
    window_title: str | None
    window_rect: tuple[int, int, int, int] | None
    screen_size: tuple[int, int] | None
    valid: bool
    reason: str


@dataclass(frozen=True)
class PresentationContext:
    allowed: bool
    confident: bool
    kind: PresentationAppKind
    process_name: str | None
    window_title: str | None
    fullscreen_like: bool
    navigation_allowed: bool
    supports_start: bool
    supports_exit: bool
    reason: str

    def summary(self) -> str:
        process = self.process_name or "-"
        return f"{self.kind.value}:{process}:{self.reason}"


class ForegroundWindowBackend:
    def snapshot(self) -> ForegroundWindowSnapshot:
        raise NotImplementedError


class StubForegroundWindowBackend(ForegroundWindowBackend):
    def __init__(self, snapshot: ForegroundWindowSnapshot | None = None):
        self._snapshot = snapshot or ForegroundWindowSnapshot(
            process_name=None,
            window_title=None,
            window_rect=None,
            screen_size=None,
            valid=False,
            reason="stub_none",
        )

    def set_snapshot(self, snapshot: ForegroundWindowSnapshot) -> None:
        self._snapshot = snapshot

    def snapshot(self) -> ForegroundWindowSnapshot:
        return self._snapshot


class WindowsForegroundWindowBackend(ForegroundWindowBackend):
    def __init__(self):
        self._user32 = ctypes.windll.user32
        self._kernel32 = ctypes.windll.kernel32

    def snapshot(self) -> ForegroundWindowSnapshot:
        hwnd = self._user32.GetForegroundWindow()
        if not hwnd:
            return ForegroundWindowSnapshot(
                process_name=None,
                window_title=None,
                window_rect=None,
                screen_size=None,
                valid=False,
                reason="no_foreground_window",
            )

        title = self._window_title(hwnd)
        pid = ctypes.c_ulong()
        self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process_name = self._process_name(int(pid.value))
        rect = self._window_rect(hwnd)
        screen_size = (
            int(self._user32.GetSystemMetrics(0)),
            int(self._user32.GetSystemMetrics(1)),
        )
        valid = process_name is not None
        return ForegroundWindowSnapshot(
            process_name=process_name,
            window_title=title,
            window_rect=rect,
            screen_size=screen_size,
            valid=valid,
            reason="ok" if valid else "process_lookup_failed",
        )

    def _window_title(self, hwnd) -> str | None:
        length = int(self._user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return None
        buffer = ctypes.create_unicode_buffer(length + 1)
        self._user32.GetWindowTextW(hwnd, buffer, length + 1)
        text = buffer.value.strip()
        return text or None

    def _window_rect(self, hwnd) -> tuple[int, int, int, int] | None:
        rect = ctypes.wintypes.RECT()
        ok = self._user32.GetWindowRect(hwnd, ctypes.byref(rect))
        if not ok:
            return None
        return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))

    def _process_name(self, pid: int) -> str | None:
        if pid <= 0:
            return None
        process = self._kernel32.OpenProcess(0x1000, False, pid)
        if not process:
            return None
        try:
            size = ctypes.c_ulong(260)
            buffer = ctypes.create_unicode_buffer(260)
            ok = self._kernel32.QueryFullProcessImageNameW(process, 0, buffer, ctypes.byref(size))
            if not ok:
                return None
            path = buffer.value
            return os.path.basename(path).lower() or None
        finally:
            self._kernel32.CloseHandle(process)


class WindowWatch:
    """
    Foreground-app context seam for Presentation Mode.
    It exposes a conservative, auditable presentation context and keeps
    app-detection logic out of gesture/runtime controllers.
    """

    def __init__(
        self,
        *,
        allowed: bool | None = None,
        backend: ForegroundWindowBackend | None = None,
        cfg: PresentationContextConfig | None = None,
    ):
        self.cfg = cfg or CONFIG.presentation_context
        self._forced_allowed = allowed if allowed is None else bool(allowed)
        self._backend = backend or WindowsForegroundWindowBackend()

    def presentation_context(self) -> PresentationContext:
        if self._forced_allowed is not None:
            return PresentationContext(
                allowed=bool(self._forced_allowed),
                confident=bool(self._forced_allowed),
                kind=PresentationAppKind.POWERPOINT if self._forced_allowed else PresentationAppKind.NONE,
                process_name=None,
                window_title=None,
                fullscreen_like=False,
                navigation_allowed=bool(self._forced_allowed),
                supports_start=bool(self._forced_allowed),
                supports_exit=bool(self._forced_allowed),
                reason="forced_allow" if self._forced_allowed else "forced_deny",
            )

        snapshot = self._backend.snapshot()
        return self._classify_snapshot(snapshot)

    def presentation_allowed(self) -> bool:
        return self.presentation_context().allowed

    def set_presentation_allowed(self, allowed: bool) -> None:
        self._forced_allowed = bool(allowed)

    def clear_forced_permission(self) -> None:
        self._forced_allowed = None

    def _classify_snapshot(self, snapshot: ForegroundWindowSnapshot) -> PresentationContext:
        if not snapshot.valid:
            return PresentationContext(
                allowed=False,
                confident=False,
                kind=PresentationAppKind.NONE,
                process_name=snapshot.process_name,
                window_title=snapshot.window_title,
                fullscreen_like=False,
                navigation_allowed=False,
                supports_start=False,
                supports_exit=False,
                reason=snapshot.reason,
            )

        process_name = (snapshot.process_name or "").lower()
        title = (snapshot.window_title or "").lower()
        fullscreen_like = self._is_fullscreen_like(snapshot)

        if process_name in self.cfg.powerpoint_processes:
            return PresentationContext(
                allowed=True,
                confident=True,
                kind=PresentationAppKind.POWERPOINT,
                process_name=process_name,
                window_title=snapshot.window_title,
                fullscreen_like=fullscreen_like,
                navigation_allowed=fullscreen_like,
                supports_start=True,
                supports_exit=fullscreen_like,
                reason="powerpoint_foreground",
            )

        if process_name in self.cfg.browser_processes:
            has_presentation_title = any(keyword in title for keyword in self.cfg.browser_title_keywords)
            if not has_presentation_title:
                return PresentationContext(
                    allowed=False,
                    confident=False,
                    kind=PresentationAppKind.UNSUPPORTED,
                    process_name=process_name,
                    window_title=snapshot.window_title,
                    fullscreen_like=fullscreen_like,
                    navigation_allowed=False,
                    supports_start=False,
                    supports_exit=False,
                    reason="browser_without_presentation_title",
                )
            return PresentationContext(
                allowed=True,
                confident=True,
                kind=PresentationAppKind.BROWSER_PRESENTATION,
                process_name=process_name,
                window_title=snapshot.window_title,
                fullscreen_like=fullscreen_like,
                navigation_allowed=fullscreen_like,
                supports_start=True,
                supports_exit=fullscreen_like,
                reason="browser_presentation_detected",
            )

        if process_name in self.cfg.pdf_processes:
            return PresentationContext(
                allowed=True,
                confident=True,
                kind=PresentationAppKind.PDF_VIEWER,
                process_name=process_name,
                window_title=snapshot.window_title,
                fullscreen_like=fullscreen_like,
                navigation_allowed=True,
                supports_start=False,
                supports_exit=fullscreen_like,
                reason="pdf_viewer_detected",
            )

        return PresentationContext(
            allowed=False,
            confident=False,
            kind=PresentationAppKind.UNSUPPORTED,
            process_name=process_name or None,
            window_title=snapshot.window_title,
            fullscreen_like=fullscreen_like,
            navigation_allowed=False,
            supports_start=False,
            supports_exit=False,
            reason="unsupported_foreground_app",
        )

    def _is_fullscreen_like(self, snapshot: ForegroundWindowSnapshot) -> bool:
        if snapshot.window_rect is None or snapshot.screen_size is None:
            return False
        left, top, right, bottom = snapshot.window_rect
        width = max(0, right - left)
        height = max(0, bottom - top)
        screen_width, screen_height = snapshot.screen_size
        margin = max(0, int(self.cfg.fullscreen_margin_px))
        return (
            abs(width - screen_width) <= margin
            and abs(height - screen_height) <= margin
            and left <= margin
            and top <= margin
        )
