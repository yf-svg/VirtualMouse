from __future__ import annotations

import ctypes
from dataclasses import dataclass
from ctypes import wintypes


@dataclass(frozen=True)
class ScreenPoint:
    x: int
    y: int


class _WindowsMouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    ]


class _WindowsInputUnion(ctypes.Union):
    _fields_ = [("mi", _WindowsMouseInput)]


class _WindowsInput(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _WindowsInputUnion),
    ]


class MouseBackend:
    def screen_size(self) -> tuple[int, int]:
        raise NotImplementedError

    def cursor_position_abs(self) -> ScreenPoint:
        raise NotImplementedError

    def move_cursor_abs(self, point: ScreenPoint) -> None:
        raise NotImplementedError

    def left_click(self) -> None:
        raise NotImplementedError

    def double_left_click(self) -> None:
        raise NotImplementedError

    def left_button_down(self) -> None:
        raise NotImplementedError

    def left_button_up(self) -> None:
        raise NotImplementedError

    def right_click(self) -> None:
        raise NotImplementedError

    def scroll_vertical(self, amount: int) -> None:
        raise NotImplementedError

    def scroll_horizontal(self, amount: int) -> None:
        raise NotImplementedError


class NoOpMouseBackend(MouseBackend):
    def __init__(self, *, width: int = 1920, height: int = 1080):
        self._width = int(width)
        self._height = int(height)
        self._cursor = ScreenPoint(self._width // 2, self._height // 2)
        self.moves: list[ScreenPoint] = []
        self.left_clicks = 0
        self.double_left_clicks = 0
        self.left_downs = 0
        self.left_ups = 0
        self.right_clicks = 0
        self.vertical_scrolls: list[int] = []
        self.horizontal_scrolls: list[int] = []

    def screen_size(self) -> tuple[int, int]:
        return (self._width, self._height)

    def cursor_position_abs(self) -> ScreenPoint:
        return self._cursor

    def move_cursor_abs(self, point: ScreenPoint) -> None:
        self._cursor = point
        self.moves.append(point)

    def left_click(self) -> None:
        self.left_clicks += 1

    def double_left_click(self) -> None:
        self.double_left_clicks += 1

    def left_button_down(self) -> None:
        self.left_downs += 1

    def left_button_up(self) -> None:
        self.left_ups += 1

    def right_click(self) -> None:
        self.right_clicks += 1

    def scroll_vertical(self, amount: int) -> None:
        self.vertical_scrolls.append(int(amount))

    def scroll_horizontal(self, amount: int) -> None:
        self.horizontal_scrolls.append(int(amount))


class WindowsMouseBackend(MouseBackend):
    _INPUT_MOUSE = 0
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP = 0x0004
    _MOUSEEVENTF_RIGHTDOWN = 0x0008
    _MOUSEEVENTF_RIGHTUP = 0x0010
    _MOUSEEVENTF_WHEEL = 0x0800
    _MOUSEEVENTF_HWHEEL = 0x1000

    def __init__(self, *, user32=None):
        if user32 is None:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
            user32.GetSystemMetrics.restype = ctypes.c_int
            user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
            user32.GetCursorPos.restype = wintypes.BOOL
            user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
            user32.SetCursorPos.restype = wintypes.BOOL
            user32.SendInput.argtypes = (
                wintypes.UINT,
                ctypes.POINTER(_WindowsInput),
                ctypes.c_int,
            )
            user32.SendInput.restype = wintypes.UINT
        self._user32 = user32

    def screen_size(self) -> tuple[int, int]:
        width = int(self._user32.GetSystemMetrics(0))
        height = int(self._user32.GetSystemMetrics(1))
        return (max(1, width), max(1, height))

    def cursor_position_abs(self) -> ScreenPoint:
        point = wintypes.POINT()
        ok = self._user32.GetCursorPos(ctypes.byref(point))
        if not ok:
            self._raise_last_error("GetCursorPos")
        return ScreenPoint(int(point.x), int(point.y))

    def move_cursor_abs(self, point: ScreenPoint) -> None:
        print(f"[OS] move x={int(point.x)} y={int(point.y)}", flush=True)
        ok = self._user32.SetCursorPos(int(point.x), int(point.y))
        if not ok:
            self._raise_last_error("SetCursorPos")

    def left_click(self) -> None:
        print("[OS] left_click", flush=True)
        self._send_mouse_inputs(
            self._MOUSEEVENTF_LEFTDOWN,
            self._MOUSEEVENTF_LEFTUP,
        )

    def double_left_click(self) -> None:
        print("[OS] double_left_click", flush=True)
        self.left_click()
        self.left_click()

    def left_button_down(self) -> None:
        print("[OS] left_button_down", flush=True)
        self._send_mouse_inputs(self._MOUSEEVENTF_LEFTDOWN)

    def left_button_up(self) -> None:
        print("[OS] left_button_up", flush=True)
        self._send_mouse_inputs(self._MOUSEEVENTF_LEFTUP)

    def right_click(self) -> None:
        print("[OS] right_click", flush=True)
        self._send_mouse_inputs(
            self._MOUSEEVENTF_RIGHTDOWN,
            self._MOUSEEVENTF_RIGHTUP,
        )

    def scroll_vertical(self, amount: int) -> None:
        print(f"[OS] scroll_vertical amount={int(amount)}", flush=True)
        self._send_mouse_inputs(self._MOUSEEVENTF_WHEEL, mouse_data=int(amount))

    def scroll_horizontal(self, amount: int) -> None:
        print(f"[OS] scroll_horizontal amount={int(amount)}", flush=True)
        self._send_mouse_inputs(self._MOUSEEVENTF_HWHEEL, mouse_data=int(amount))

    def _send_mouse_inputs(self, *flags: int, mouse_data: int = 0) -> None:
        inputs = (_WindowsInput * len(flags))(
            *[
                _WindowsInput(
                    type=self._INPUT_MOUSE,
                    mi=_WindowsMouseInput(
                        dx=0,
                        dy=0,
                        mouseData=mouse_data,
                        dwFlags=flag,
                        time=0,
                        dwExtraInfo=0,
                    ),
                )
                for flag in flags
            ]
        )
        sent = int(self._user32.SendInput(len(inputs), inputs, ctypes.sizeof(_WindowsInput)))
        if sent != len(inputs):
            self._raise_last_error("SendInput", sent=sent, expected=len(inputs))

    @staticmethod
    def _raise_last_error(context: str, *, sent: int | None = None, expected: int | None = None) -> None:
        code = int(ctypes.get_last_error())
        if code:
            print(f"[OS_ERROR] context={context} code={code}", flush=True)
            raise OSError(code, f"{context} failed", None, code)
        if sent is not None and expected is not None:
            print(f"[OS_ERROR] context={context} partial_send={sent}/{expected}", flush=True)
            raise OSError(f"{context} partial_send:{sent}/{expected}")
        print(f"[OS_ERROR] context={context} code=unknown", flush=True)
        raise OSError(f"{context} failed")
