from __future__ import annotations

import unittest

from app.control.mouse import ScreenPoint, WindowsMouseBackend


class _FakeUser32:
    def __init__(self):
        self.send_input_calls: list[list[tuple[int, int]]] = []
        self.cursor_calls: list[tuple[int, int]] = []
        self.system_metrics = {0: 1920, 1: 1080}
        self.send_input_result: int | None = None
        self.set_cursor_result = 1

    def GetSystemMetrics(self, index: int) -> int:
        return self.system_metrics[int(index)]

    def SetCursorPos(self, x: int, y: int) -> int:
        self.cursor_calls.append((int(x), int(y)))
        return int(self.set_cursor_result)

    def SendInput(self, count, inputs, cb_size):
        events: list[tuple[int, int]] = []
        for idx in range(int(count)):
            item = inputs[idx]
            events.append((int(item.mi.dwFlags), int(item.mi.mouseData)))
        self.send_input_calls.append(events)
        if self.send_input_result is None:
            return int(count)
        return int(self.send_input_result)


class Phase6WindowsMouseBackendTests(unittest.TestCase):
    def test_left_click_uses_sendinput_down_and_up(self):
        user32 = _FakeUser32()
        backend = WindowsMouseBackend(user32=user32)

        backend.left_click()

        self.assertEqual(
            user32.send_input_calls,
            [[
                (WindowsMouseBackend._MOUSEEVENTF_LEFTDOWN, 0),
                (WindowsMouseBackend._MOUSEEVENTF_LEFTUP, 0),
            ]],
        )

    def test_right_click_uses_sendinput_down_and_up(self):
        user32 = _FakeUser32()
        backend = WindowsMouseBackend(user32=user32)

        backend.right_click()

        self.assertEqual(
            user32.send_input_calls,
            [[
                (WindowsMouseBackend._MOUSEEVENTF_RIGHTDOWN, 0),
                (WindowsMouseBackend._MOUSEEVENTF_RIGHTUP, 0),
            ]],
        )

    def test_vertical_scroll_uses_sendinput_wheel_data(self):
        user32 = _FakeUser32()
        backend = WindowsMouseBackend(user32=user32)

        backend.scroll_vertical(120)

        self.assertEqual(
            user32.send_input_calls,
            [[(WindowsMouseBackend._MOUSEEVENTF_WHEEL, 120)]],
        )

    def test_move_cursor_checks_setcursorpos_result(self):
        user32 = _FakeUser32()
        backend = WindowsMouseBackend(user32=user32)

        backend.move_cursor_abs(ScreenPoint(25, 40))

        self.assertEqual(user32.cursor_calls, [(25, 40)])

    def test_partial_send_raises_oserror(self):
        user32 = _FakeUser32()
        user32.send_input_result = 1
        backend = WindowsMouseBackend(user32=user32)

        with self.assertRaises(OSError):
            backend.left_click()


if __name__ == "__main__":
    unittest.main()
