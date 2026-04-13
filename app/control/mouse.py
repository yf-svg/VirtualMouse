from __future__ import annotations

import ctypes
from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenPoint:
    x: int
    y: int


class MouseBackend:
    def screen_size(self) -> tuple[int, int]:
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

    def move_cursor_abs(self, point: ScreenPoint) -> None:
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
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP = 0x0004
    _MOUSEEVENTF_RIGHTDOWN = 0x0008
    _MOUSEEVENTF_RIGHTUP = 0x0010
    _MOUSEEVENTF_WHEEL = 0x0800
    _MOUSEEVENTF_HWHEEL = 0x1000

    def __init__(self):
        user32 = ctypes.windll.user32
        self._user32 = user32

    def screen_size(self) -> tuple[int, int]:
        width = int(self._user32.GetSystemMetrics(0))
        height = int(self._user32.GetSystemMetrics(1))
        return (max(1, width), max(1, height))

    def move_cursor_abs(self, point: ScreenPoint) -> None:
        self._user32.SetCursorPos(int(point.x), int(point.y))

    def left_click(self) -> None:
        self.left_button_down()
        self.left_button_up()

    def double_left_click(self) -> None:
        self.left_click()
        self.left_click()

    def left_button_down(self) -> None:
        self._user32.mouse_event(self._MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def left_button_up(self) -> None:
        self._user32.mouse_event(self._MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def right_click(self) -> None:
        self._user32.mouse_event(self._MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        self._user32.mouse_event(self._MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)

    def scroll_vertical(self, amount: int) -> None:
        self._user32.mouse_event(self._MOUSEEVENTF_WHEEL, 0, 0, int(amount), 0)

    def scroll_horizontal(self, amount: int) -> None:
        self._user32.mouse_event(self._MOUSEEVENTF_HWHEEL, 0, 0, int(amount), 0)
