from __future__ import annotations

import ctypes


class KeyboardBackend:
    def press_key(self, key_name: str) -> None:
        raise NotImplementedError

    def release_all(self) -> None:
        raise NotImplementedError


class NoOpKeyboardBackend(KeyboardBackend):
    def __init__(self):
        self.keys: list[str] = []
        self.release_all_calls = 0

    def press_key(self, key_name: str) -> None:
        self.keys.append(str(key_name))

    def release_all(self) -> None:
        self.release_all_calls += 1


class WindowsKeyboardBackend(KeyboardBackend):
    _KEYEVENTF_KEYUP = 0x0002
    _VK_MAP = {
        "LEFT": 0x25,
        "RIGHT": 0x27,
        "ESC": 0x1B,
        "F5": 0x74,
    }

    def __init__(self):
        self._user32 = ctypes.windll.user32

    def press_key(self, key_name: str) -> None:
        vk = self._VK_MAP.get(str(key_name).upper())
        if vk is None:
            raise ValueError(f"Unsupported presentation key: {key_name}")
        self._user32.keybd_event(vk, 0, 0, 0)
        self._user32.keybd_event(vk, 0, self._KEYEVENTF_KEYUP, 0)

    def release_all(self) -> None:
        for vk in self._VK_MAP.values():
            self._user32.keybd_event(vk, 0, self._KEYEVENTF_KEYUP, 0)
