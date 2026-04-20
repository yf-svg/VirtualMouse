from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import math
from dataclasses import dataclass

from app.control.mouse import ScreenPoint


_LRESULT = getattr(wintypes, "LRESULT", getattr(ctypes, "c_ssize_t", wintypes.LPARAM))


@dataclass(frozen=True)
class PresentationStroke:
    points: tuple[ScreenPoint, ...]
    color_argb: int
    glow_color_argb: int
    radius: int
    glow_radius: int
    pen_kind: str = "pen"


@dataclass(frozen=True)
class PresentationDrawStyle:
    color_argb: int
    glow_color_argb: int
    radius: int
    glow_radius: int
    pen_kind: str = "pen"


@dataclass(frozen=True)
class PresentationPanelItemState:
    item_id: str
    kind: str
    bounds: tuple[int, int, int, int]
    center: ScreenPoint
    fill_argb: int
    preview_argb: int | None = None
    preview_glow_argb: int | None = None
    preview_radius: int | None = None
    preview_kind: str | None = None
    label_text: str | None = None
    active: bool = False
    hovered: bool = False


@dataclass(frozen=True)
class PresentationPanelRenderState:
    visible: bool
    frame: tuple[int, int, int, int]
    color_section_bounds: tuple[int, int, int, int]
    pen_section_bounds: tuple[int, int, int, int]
    size_section_bounds: tuple[int, int, int, int]
    expansion: float
    active_color_argb: int
    active_pen_radius: int
    active_pen_glow_radius: int
    active_pen_kind: str = "pen"
    items: tuple[PresentationPanelItemState, ...] = ()


@dataclass(frozen=True)
class PresentationOverlayState:
    visible: bool
    window_rect: tuple[int, int, int, int] | None
    laser_point: ScreenPoint | None = None
    draw_point: ScreenPoint | None = None
    draw_cursor_style: str | None = None
    draw_style: PresentationDrawStyle | None = None
    strokes: tuple[PresentationStroke, ...] = ()
    panel: PresentationPanelRenderState | None = None


class PresentationOverlayBackend:
    def render(self, state: PresentationOverlayState) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.reset()


class NoOpPresentationOverlayBackend(PresentationOverlayBackend):
    def __init__(self):
        self.states: list[PresentationOverlayState] = []
        self.reset_calls = 0

    def render(self, state: PresentationOverlayState) -> None:
        self.states.append(state)

    def reset(self) -> None:
        self.reset_calls += 1
        self.states.append(PresentationOverlayState(False, None))


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _SIZE(ctypes.Structure):
    _fields_ = [("cx", wintypes.LONG), ("cy", wintypes.LONG)]


class _BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", ctypes.c_ubyte),
        ("BlendFlags", ctypes.c_ubyte),
        ("SourceConstantAlpha", ctypes.c_ubyte),
        ("AlphaFormat", ctypes.c_ubyte),
    ]


class _WNDCLASSEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("style", wintypes.UINT),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HANDLE),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HANDLE),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HANDLE),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _RGBQUAD(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", ctypes.c_ubyte),
        ("rgbGreen", ctypes.c_ubyte),
        ("rgbRed", ctypes.c_ubyte),
        ("rgbReserved", ctypes.c_ubyte),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
        ("bmiColors", _RGBQUAD * 1),
    ]


class _MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", _POINT),
    ]


def map_pointer_to_window(
    pointer_point,
    window_rect: tuple[int, int, int, int] | None,
) -> ScreenPoint | None:
    if pointer_point is None or window_rect is None:
        return None
    left, top, right, bottom = window_rect
    width = max(1, int(right - left))
    height = max(1, int(bottom - top))
    x = min(1.0, max(0.0, float(pointer_point.x)))
    y = min(1.0, max(0.0, float(pointer_point.y)))
    return ScreenPoint(
        x=int(left + round(x * max(0, width - 1))),
        y=int(top + round(y * max(0, height - 1))),
    )


def _with_alpha(color: int, alpha: int) -> int:
    alpha = max(0, min(255, int(alpha)))
    return (alpha << 24) | (int(color) & 0x00FFFFFF)


def _scale_alpha(color: int, factor: float) -> int:
    base_alpha = (int(color) >> 24) & 0xFF
    return _with_alpha(color, int(round(base_alpha * max(0.0, factor))))


class WindowsPresentationOverlayBackend(PresentationOverlayBackend):
    _WS_EX_LAYERED = 0x00080000
    _WS_EX_TRANSPARENT = 0x00000020
    _WS_EX_TOPMOST = 0x00000008
    _WS_EX_TOOLWINDOW = 0x00000080
    _WS_EX_NOACTIVATE = 0x08000000
    _WS_POPUP = 0x80000000
    _ULW_ALPHA = 0x00000002
    _AC_SRC_OVER = 0x00
    _AC_SRC_ALPHA = 0x01
    _SW_HIDE = 0
    _SW_SHOWNOACTIVATE = 4
    _PM_REMOVE = 0x0001
    _DIB_RGB_COLORS = 0
    _BI_RGB = 0
    _LASER_HALO_RADIUS = 22
    _LASER_HALO_COLOR = 0x60FF5050
    _LASER_DOT_RADIUS = 10
    _LASER_DOT_COLOR = 0xC8FF3030
    _LASER_CORE_RADIUS = 4
    _LASER_CORE_COLOR = 0xF8FFF6E8
    _DRAW_CURSOR_OUTER_RADIUS = 10
    _DRAW_CURSOR_OUTER_COLOR = 0xB048D8FF
    _DRAW_CURSOR_INNER_RADIUS = 4
    _DRAW_CURSOR_INNER_COLOR = 0xF0FFFFFF
    _PEN_BODY_GLOW_COLOR = 0x6064E8FF
    _PEN_BODY_COLOR = 0xD03CC8FF
    _PEN_ACCENT_COLOR = 0xE0FFF3C8
    _PEN_TIP_COLOR = 0xF8FFFDF8
    _ERASER_GLOW_COLOR = 0x50FFD8F2
    _ERASER_BODY_COLOR = 0xD0FF92D0
    _ERASER_BAND_COLOR = 0xF0FFF6F2
    _PANEL_SHADOW_COLOR = 0x40000000
    _PANEL_BASE_COLOR = 0xE0121A25
    _PANEL_RING_COLOR = 0x30FFFFFF
    _PANEL_ITEM_BG = 0xDC111925
    _PANEL_ITEM_HOVER = 0xF01D2937
    _PANEL_ITEM_ACTIVE = 0xF0243445
    _PANEL_ITEM_ACTIVE_RING = 0xEAF9FCFF
    _PANEL_SECTION_BG = 0xA4101823
    _PANEL_SECTION_EDGE = 0x18FFFFFF
    _PANEL_DIVIDER = 0x20FFFFFF
    _PANEL_SHEEN = 0x14FFFFFF
    _PANEL_PEN_NEUTRAL = 0xECF7FAFF
    _PANEL_PEN_GLOW = 0x4CEAF3FF
    _STROKE_CURVE_SUBDIVISIONS = 5
    _CLASS_NAME = "VirtualMousePresentationOverlay"
    _CLASS_ATOM = 0
    _WNDPROC = None

    def __init__(self):
        self._user32 = ctypes.windll.user32
        self._gdi32 = ctypes.windll.gdi32
        self._kernel32 = ctypes.windll.kernel32
        self._hwnd = None
        self._memdc = None
        self._bitmap = None
        self._prev_bitmap = None
        self._bits_ptr = None
        self._buffer = None
        self._width = 0
        self._height = 0
        self._visible = False
        self._register_class()

    def render(self, state: PresentationOverlayState) -> None:
        self._pump_messages()
        if not state.visible or state.window_rect is None:
            self.reset()
            return
        left, top, right, bottom = state.window_rect
        width = max(1, int(right - left))
        height = max(1, int(bottom - top))
        self._ensure_window(left, top, width, height)
        self._clear_buffer()
        drew_any = False
        panel = state.panel if state.panel is not None and (state.panel.visible or state.panel.expansion > 0.02) else None
        for stroke in state.strokes:
            self._draw_stroke(stroke, left, top)
            drew_any = True
        if panel is not None:
            self._draw_panel(panel, left, top)
            drew_any = True
        if state.draw_point is not None:
            draw_x = state.draw_point.x - left
            draw_y = state.draw_point.y - top
            if state.draw_cursor_style == "eraser":
                self._draw_eraser_cursor(draw_x, draw_y)
            elif state.draw_cursor_style == "pen":
                self._draw_pen_cursor(draw_x, draw_y, state.draw_style)
            else:
                self._draw_cursor(draw_x, draw_y)
            drew_any = True
        if state.laser_point is not None:
            self._draw_laser(state.laser_point.x - left, state.laser_point.y - top)
            drew_any = True
        if not drew_any:
            self.reset()
            return
        blend = _BLENDFUNCTION(self._AC_SRC_OVER, 0, 255, self._AC_SRC_ALPHA)
        src = _POINT(0, 0)
        pos = _POINT(int(left), int(top))
        size = _SIZE(int(width), int(height))
        ok = self._user32.UpdateLayeredWindow(
            self._hwnd,
            None,
            ctypes.byref(pos),
            ctypes.byref(size),
            self._memdc,
            ctypes.byref(src),
            0,
            ctypes.byref(blend),
            self._ULW_ALPHA,
        )
        if not ok:
            self._raise_last_error("UpdateLayeredWindow")
        if not self._visible:
            self._user32.ShowWindow(self._hwnd, self._SW_SHOWNOACTIVATE)
            self._visible = True

    def reset(self) -> None:
        self._pump_messages()
        if self._hwnd and self._visible:
            self._user32.ShowWindow(self._hwnd, self._SW_HIDE)
            self._visible = False

    def close(self) -> None:
        self.reset()
        self._release_surface()
        if self._hwnd:
            self._user32.DestroyWindow(self._hwnd)
            self._hwnd = None

    def _register_class(self) -> None:
        if WindowsPresentationOverlayBackend._CLASS_ATOM:
            return
        if WindowsPresentationOverlayBackend._WNDPROC is None:
            WindowsPresentationOverlayBackend._WNDPROC = ctypes.WINFUNCTYPE(
                _LRESULT,
                wintypes.HWND,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )(self._wnd_proc)
        instance = self._kernel32.GetModuleHandleW(None)
        wc = _WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(_WNDCLASSEXW)
        wc.lpfnWndProc = ctypes.cast(WindowsPresentationOverlayBackend._WNDPROC, ctypes.c_void_p).value
        wc.hInstance = instance
        wc.lpszClassName = self._CLASS_NAME
        atom = self._user32.RegisterClassExW(ctypes.byref(wc))
        if atom == 0:
            code = int(ctypes.get_last_error())
            if code != 1410:
                self._raise_last_error("RegisterClassExW")
        WindowsPresentationOverlayBackend._CLASS_ATOM = atom or 1

    def _ensure_window(self, left: int, top: int, width: int, height: int) -> None:
        if self._hwnd is None:
            instance = self._kernel32.GetModuleHandleW(None)
            self._hwnd = self._user32.CreateWindowExW(
                self._WS_EX_LAYERED
                | self._WS_EX_TRANSPARENT
                | self._WS_EX_TOPMOST
                | self._WS_EX_TOOLWINDOW
                | self._WS_EX_NOACTIVATE,
                self._CLASS_NAME,
                None,
                self._WS_POPUP,
                int(left),
                int(top),
                int(width),
                int(height),
                None,
                None,
                instance,
                None,
            )
            if not self._hwnd:
                self._raise_last_error("CreateWindowExW")
        if width != self._width or height != self._height:
            self._resize_surface(width, height)

    def _resize_surface(self, width: int, height: int) -> None:
        self._release_surface()
        screen_dc = self._user32.GetDC(None)
        self._memdc = self._gdi32.CreateCompatibleDC(screen_dc)
        if not self._memdc:
            self._user32.ReleaseDC(None, screen_dc)
            self._raise_last_error("CreateCompatibleDC")
        bmi = _BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = int(width)
        bmi.bmiHeader.biHeight = -int(height)
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = self._BI_RGB
        bits_ptr = ctypes.c_void_p()
        self._bitmap = self._gdi32.CreateDIBSection(
            self._memdc,
            ctypes.byref(bmi),
            self._DIB_RGB_COLORS,
            ctypes.byref(bits_ptr),
            None,
            0,
        )
        self._user32.ReleaseDC(None, screen_dc)
        if not self._bitmap or not bits_ptr.value:
            self._raise_last_error("CreateDIBSection")
        self._prev_bitmap = self._gdi32.SelectObject(self._memdc, self._bitmap)
        self._bits_ptr = bits_ptr
        self._buffer = (ctypes.c_uint32 * (width * height)).from_address(bits_ptr.value)
        self._width = int(width)
        self._height = int(height)

    def _release_surface(self) -> None:
        if self._memdc and self._prev_bitmap:
            self._gdi32.SelectObject(self._memdc, self._prev_bitmap)
            self._prev_bitmap = None
        if self._bitmap:
            self._gdi32.DeleteObject(self._bitmap)
            self._bitmap = None
        if self._memdc:
            self._gdi32.DeleteDC(self._memdc)
            self._memdc = None
        self._bits_ptr = None
        self._buffer = None
        self._width = 0
        self._height = 0

    def _clear_buffer(self) -> None:
        if self._buffer is None:
            return
        self._buffer[:] = [0] * len(self._buffer)

    def _draw_laser(self, x: int, y: int) -> None:
        self._draw_disc(x, y, self._LASER_HALO_RADIUS, self._LASER_HALO_COLOR)
        self._draw_disc(x, y, self._LASER_DOT_RADIUS, self._LASER_DOT_COLOR)
        self._draw_disc(x, y, self._LASER_CORE_RADIUS, self._LASER_CORE_COLOR)

    def _draw_cursor(self, x: int, y: int) -> None:
        self._draw_disc(x, y, self._DRAW_CURSOR_OUTER_RADIUS, self._DRAW_CURSOR_OUTER_COLOR)
        self._draw_disc(x, y, self._DRAW_CURSOR_INNER_RADIUS, self._DRAW_CURSOR_INNER_COLOR)

    def _draw_pen_cursor(self, x: int, y: int, style: PresentationDrawStyle | None) -> None:
        if style is None:
            glow_color = self._PEN_BODY_GLOW_COLOR
            body_color = self._PEN_BODY_COLOR
            body_radius = 4
            glow_radius = 6
        else:
            glow_color = style.glow_color_argb
            body_color = style.color_argb
            body_radius = max(3, int(style.radius) + 1)
            glow_radius = max(body_radius + 2, int(style.glow_radius))
        self._draw_line(x - 9, y + 8, x + 5, y - 6, glow_color, glow_radius)
        self._draw_line(x - 8, y + 7, x + 4, y - 5, body_color, body_radius)
        self._draw_line(x + 1, y - 10, x + 9, y - 2, self._PEN_ACCENT_COLOR, 3)
        self._draw_line(x - 12, y + 11, x - 8, y + 7, self._PEN_TIP_COLOR, 2)
        self._draw_disc(x - 12, y + 11, 1, self._PEN_TIP_COLOR)

    def _draw_eraser_cursor(self, x: int, y: int) -> None:
        self._draw_line(x - 10, y + 8, x + 6, y - 8, self._ERASER_GLOW_COLOR, 8)
        self._draw_line(x - 10, y + 8, x + 6, y - 8, self._ERASER_BODY_COLOR, 6)
        self._draw_line(x + 1, y - 3, x + 10, y - 12, self._ERASER_BAND_COLOR, 4)

    def _draw_stroke(self, stroke: PresentationStroke, left: int, top: int) -> None:
        if not stroke.points:
            return
        if len(stroke.points) == 1:
            point = stroke.points[0]
            self._draw_disc(point.x - left, point.y - top, stroke.glow_radius, stroke.glow_color_argb)
            self._draw_disc(point.x - left, point.y - top, stroke.radius, stroke.color_argb)
            return
        for start, end in zip(stroke.points, stroke.points[1:]):
            self._draw_brush_segment(
                start.x - left,
                start.y - top,
                end.x - left,
                end.y - top,
                stroke,
            )

    def _draw_brush_segment(self, x0: int, y0: int, x1: int, y1: int, stroke: PresentationStroke) -> None:
        kind = str(getattr(stroke, "pen_kind", "pen") or "pen").lower()
        if kind == "marker":
            self._draw_line(x0, y0, x1, y1, stroke.glow_color_argb, max(stroke.glow_radius, stroke.radius + 4))
            self._draw_line(x0, y0, x1, y1, _scale_alpha(stroke.color_argb, 0.72), max(stroke.radius + 1, 1))
            self._draw_line(x0, y0, x1, y1, stroke.color_argb, max(stroke.radius, 1))
            return
        if kind == "highlighter":
            self._draw_line(x0, y0, x1, y1, _scale_alpha(stroke.glow_color_argb, 0.82), max(stroke.glow_radius, stroke.radius + 6))
            self._draw_line(x0, y0, x1, y1, _scale_alpha(stroke.color_argb, 0.88), max(stroke.radius + 2, 1))
            self._draw_line(x0, y0, x1, y1, _scale_alpha(stroke.color_argb, 0.58), max(stroke.radius, 1))
            return
        if kind == "brush":
            self._draw_line(x0, y0, x1, y1, stroke.glow_color_argb, max(stroke.glow_radius, stroke.radius + 3))
            self._draw_line(x0, y0, x1, y1, _scale_alpha(stroke.color_argb, 0.76), max(stroke.radius + 1, 1))
            self._draw_line(x0 - 1, y0 + 1, x1 - 1, y1 + 1, _scale_alpha(stroke.color_argb, 0.62), max(1, stroke.radius - 1))
            self._draw_line(x0 + 1, y0 - 1, x1 + 1, y1 - 1, _scale_alpha(stroke.color_argb, 0.54), max(1, stroke.radius - 2))
            return
        if kind == "quill":
            self._draw_quill_segment(x0, y0, x1, y1, stroke)
            return
        self._draw_line(x0, y0, x1, y1, stroke.glow_color_argb, max(stroke.glow_radius, stroke.radius + 1))
        self._draw_line(x0, y0, x1, y1, stroke.color_argb, max(stroke.radius, 1))
        self._draw_line(x0, y0, x1, y1, _scale_alpha(0xFFFFFFFF, 0.22), max(1, stroke.radius - 1))

    def _draw_quill_segment(self, x0: int, y0: int, x1: int, y1: int, stroke: PresentationStroke) -> None:
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        length = max(1.0, math.hypot(dx, dy))
        perp_x = -dy / length
        perp_y = dx / length
        spread = max(1.0, float(stroke.radius))
        self._draw_line(
            int(round(x0 - (perp_x * spread * 0.85))),
            int(round(y0 - (perp_y * spread * 0.85))),
            int(round(x1 - (perp_x * spread * 0.85))),
            int(round(y1 - (perp_y * spread * 0.85))),
            stroke.glow_color_argb,
            max(1, stroke.glow_radius - 1),
        )
        self._draw_line(x0, y0, x1, y1, stroke.color_argb, max(1, stroke.radius))
        self._draw_line(
            int(round(x0 + (perp_x * spread * 1.15))),
            int(round(y0 + (perp_y * spread * 1.15))),
            int(round(x1 + (perp_x * spread * 1.15))),
            int(round(y1 + (perp_y * spread * 1.15))),
            _scale_alpha(stroke.color_argb, 0.55),
            max(1, stroke.radius - 1),
        )

    def _smooth_path_points(self, points: tuple[ScreenPoint, ...]) -> tuple[ScreenPoint, ...]:
        if len(points) <= 2:
            return points
        subdivisions = max(1, int(self._STROKE_CURVE_SUBDIVISIONS))
        if subdivisions <= 1:
            return points
        expanded = [points[0], *points, points[-1]]
        smoothed: list[ScreenPoint] = [points[0]]
        for idx in range(1, len(expanded) - 2):
            p0, p1, p2, p3 = expanded[idx - 1], expanded[idx], expanded[idx + 1], expanded[idx + 2]
            min_x = min(p1.x, p2.x)
            max_x = max(p1.x, p2.x)
            min_y = min(p1.y, p2.y)
            max_y = max(p1.y, p2.y)
            for step in range(1, subdivisions + 1):
                t = float(step) / float(subdivisions)
                x = 0.5 * (
                    (2.0 * p1.x)
                    + ((-p0.x + p2.x) * t)
                    + ((2.0 * p0.x - (5.0 * p1.x) + (4.0 * p2.x) - p3.x) * (t * t))
                    + ((-p0.x + (3.0 * p1.x) - (3.0 * p2.x) + p3.x) * (t * t * t))
                )
                y = 0.5 * (
                    (2.0 * p1.y)
                    + ((-p0.y + p2.y) * t)
                    + ((2.0 * p0.y - (5.0 * p1.y) + (4.0 * p2.y) - p3.y) * (t * t))
                    + ((-p0.y + (3.0 * p1.y) - (3.0 * p2.y) + p3.y) * (t * t * t))
                )
                candidate = ScreenPoint(
                    x=int(round(min(max(x, min_x), max_x))),
                    y=int(round(min(max(y, min_y), max_y))),
                )
                if candidate != smoothed[-1]:
                    smoothed.append(candidate)
        if smoothed[-1] != points[-1]:
            smoothed.append(points[-1])
        return tuple(smoothed)

    def _draw_pen_glyph(
        self,
        x: int,
        y: int,
        *,
        kind: str,
        color: int,
        glow_color: int,
        radius: int,
        glow_radius: int,
        size: int,
    ) -> None:
        kind = str(kind or "pen").lower()
        if kind == "marker":
            self._draw_line(x - size, y + max(2, size // 2), x + size, y - max(2, size // 2), glow_color, max(glow_radius, radius + 3))
            self._draw_line(x - size, y + max(2, size // 2), x + size, y - max(2, size // 2), _scale_alpha(color, 0.72), max(radius + 1, 1))
            self._draw_line(x - size + 1, y + max(2, size // 2) - 1, x + size - 1, y - max(2, size // 2) + 1, color, max(radius, 1))
            return
        if kind == "highlighter":
            self._draw_line(x - size, y, x + size, y, _scale_alpha(glow_color, 0.84), max(glow_radius, radius + 4))
            self._draw_line(x - size, y, x + size, y, _scale_alpha(color, 0.78), max(radius + 1, 1))
            self._draw_line(x - size + 2, y, x + size - 2, y, _scale_alpha(0xFFFFFFFF, 0.18), max(1, radius - 2))
            return
        if kind == "brush":
            self._draw_line(x - size, y + max(1, size // 3), x + size, y - max(1, size // 3), glow_color, max(glow_radius, radius + 2))
            self._draw_line(x - size + 1, y + max(1, size // 3), x + size - 1, y - max(1, size // 3), _scale_alpha(color, 0.78), max(radius + 1, 1))
            self._draw_line(x - size + 1, y + max(2, size // 2), x + size - 2, y, _scale_alpha(color, 0.52), max(1, radius - 1))
            return
        if kind == "quill":
            self._draw_line(x - size, y + max(2, size // 2), x + size - 2, y - max(2, size // 2), glow_color, max(glow_radius, radius + 1))
            self._draw_line(x - size + 1, y + max(2, size // 2) - 1, x + size - 3, y - max(2, size // 2) + 1, color, max(radius, 1))
            self._draw_line(x - size + 4, y + max(2, size // 2), x + size - 1, y - max(1, size // 3), _scale_alpha(color, 0.55), max(1, radius - 1))
            self._draw_disc(x + size - 1, y - max(2, size // 2), max(2, radius + 1), color)
            return
        self._draw_line(x - size, y + max(2, size // 2), x + size, y - max(2, size // 2), glow_color, max(glow_radius, radius + 1))
        self._draw_line(x - size + 1, y + max(2, size // 2) - 1, x + size - 1, y - max(2, size // 2) + 1, color, max(radius, 1))
        self._draw_line(x - size + 1, y + max(1, size // 3), x + size - 2, y - max(1, size // 3), _scale_alpha(0xFFFFFFFF, 0.18), max(1, radius - 1))

    def _draw_panel(self, panel: PresentationPanelRenderState, left: int, top: int) -> None:
        expansion = max(0.0, min(1.0, float(panel.expansion)))
        if not panel.visible and expansion <= 0.02:
            return
        frame = self._relative_bounds(panel.frame, left, top)
        color_bounds = self._relative_bounds(panel.color_section_bounds, left, top)
        pen_bounds = self._relative_bounds(panel.pen_section_bounds, left, top)
        size_bounds = self._relative_bounds(panel.size_section_bounds, left, top)
        frame_radius = max(16, int(round(min(frame[2] - frame[0], frame[3] - frame[1]) * 0.10)))
        section_radius = max(10, int(round(frame_radius * 0.72)))
        shadow = self._offset_bounds(frame, 8, 10)
        self._draw_round_rect(shadow, frame_radius + 4, _scale_alpha(self._PANEL_SHADOW_COLOR, 0.82))
        self._draw_round_rect(self._offset_bounds(frame, 2, 4), frame_radius + 2, _scale_alpha(0x1E33485D, expansion))
        self._draw_round_rect(frame, frame_radius, _scale_alpha(self._PANEL_BASE_COLOR, 0.98))
        self._draw_round_rect(self._inset_bounds(frame, 2), max(6, frame_radius - 2), self._PANEL_RING_COLOR)
        sheen_height = max(18, int(round((frame[3] - frame[1]) * 0.16)))
        sheen_bounds = (frame[0] + 10, frame[1] + 8, frame[2] - 10, frame[1] + sheen_height)
        self._draw_round_rect(sheen_bounds, max(8, frame_radius - 6), _scale_alpha(self._PANEL_SHEEN, expansion))
        self._draw_round_rect(color_bounds, section_radius, _scale_alpha(self._PANEL_SECTION_BG, 0.96))
        self._draw_round_rect(self._inset_bounds(color_bounds, 1), max(4, section_radius - 1), self._PANEL_SECTION_EDGE)
        self._draw_round_rect(pen_bounds, section_radius, _scale_alpha(self._PANEL_SECTION_BG, 0.96))
        self._draw_round_rect(self._inset_bounds(pen_bounds, 1), max(4, section_radius - 1), self._PANEL_SECTION_EDGE)
        self._draw_round_rect(size_bounds, section_radius, _scale_alpha(self._PANEL_SECTION_BG, 0.96))
        self._draw_round_rect(self._inset_bounds(size_bounds, 1), max(4, section_radius - 1), self._PANEL_SECTION_EDGE)
        first_divider_y = pen_bounds[1] - max(6, int(round((pen_bounds[1] - color_bounds[3]) * 0.5)))
        second_divider_y = size_bounds[1] - max(6, int(round((size_bounds[1] - pen_bounds[3]) * 0.5)))
        self._fill_rect(frame[0] + 18, first_divider_y, frame[2] - 18, first_divider_y + 1, self._PANEL_DIVIDER)
        self._fill_rect(frame[0] + 18, second_divider_y, frame[2] - 18, second_divider_y + 1, self._PANEL_DIVIDER)
        for item in panel.items:
            item_bounds = self._relative_bounds(item.bounds, left, top)
            item_center = ScreenPoint(
                x=int(round((item_bounds[0] + item_bounds[2]) * 0.5)),
                y=int(round((item_bounds[1] + item_bounds[3]) * 0.5)),
            )
            item_radius = max(8, int(round(min(item_bounds[2] - item_bounds[0], item_bounds[3] - item_bounds[1]) * 0.24)))
            card_radius = max(10, int(round(min(item_bounds[2] - item_bounds[0], item_bounds[3] - item_bounds[1]) * 0.22)))
            if item.hovered:
                self._draw_round_rect(self._offset_bounds(item_bounds, 0, 2), card_radius + 1, _scale_alpha(item.fill_argb, 0.28))
                self._draw_round_rect(item_bounds, card_radius, self._PANEL_ITEM_ACTIVE_RING)
            elif item.active:
                self._draw_round_rect(self._offset_bounds(item_bounds, 0, 2), card_radius + 1, _scale_alpha(item.fill_argb, 0.24))
                self._draw_round_rect(item_bounds, card_radius, _scale_alpha(panel.active_color_argb, 0.96))
            fill_color = self._PANEL_ITEM_ACTIVE if item.active else self._PANEL_ITEM_HOVER if item.hovered else self._PANEL_ITEM_BG
            self._draw_round_rect(self._inset_bounds(item_bounds, 2), max(6, card_radius - 2), fill_color)
            self._draw_round_rect(self._inset_bounds(item_bounds, 3), max(5, card_radius - 3), self._PANEL_RING_COLOR)
            if item.kind == "color":
                self._draw_disc(item_center.x, item_center.y, item_radius + 4, _scale_alpha(item.fill_argb, 0.28))
                self._draw_disc(item_center.x, item_center.y, item_radius + 1, 0xEAF9FCFF if item.active or item.hovered else 0x54FFFFFF)
                self._draw_disc(item_center.x, item_center.y, item_radius - 1, item.fill_argb)
                self._draw_disc(
                    item_center.x - max(1, item_radius // 3),
                    item_center.y - max(1, item_radius // 3),
                    max(2, item_radius // 3),
                    0x34FFFFFF,
                )
                continue
            if item.kind == "size":
                self._draw_round_rect(self._inset_bounds(item_bounds, 8), max(6, card_radius - 7), 0xD0101721)
                if item.preview_argb is not None and item.preview_radius is not None:
                    preview_y = item_bounds[1] + max(12, int(round((item_bounds[3] - item_bounds[1]) * 0.36)))
                    preview_width = max(10, min((item_bounds[2] - item_bounds[0]) // 4, 18))
                    self._draw_pen_glyph(
                        item_center.x,
                        preview_y,
                        kind=item.preview_kind or "pen",
                        color=item.preview_argb,
                        glow_color=item.preview_glow_argb or _scale_alpha(item.preview_argb, 0.35),
                        radius=int(item.preview_radius),
                        glow_radius=max(item.preview_radius + 2, int(item.preview_radius) + 2),
                        size=preview_width,
                    )
                if item.label_text:
                    self._draw_numeric_label(
                        text=item.label_text,
                        center_x=item_center.x,
                        top=item_bounds[3] - max(15, (item_bounds[3] - item_bounds[1]) // 3),
                        color=0xECF7FAFF if item.active or item.hovered else 0xB8D8E7F2,
                        scale=max(1, min(2, (item_bounds[3] - item_bounds[1]) // 24)),
                    )
                continue
            self._draw_round_rect(self._inset_bounds(item_bounds, 8), max(6, card_radius - 7), 0xD0101721)
            if item.preview_argb is not None and item.preview_radius is not None:
                preview_glow_radius = max(item.preview_radius + 2, int(item.preview_radius) + 2)
                self._draw_pen_glyph(
                    item_center.x,
                    item_center.y,
                    kind=item.preview_kind or "pen",
                    color=item.preview_argb,
                    glow_color=item.preview_glow_argb or _scale_alpha(item.preview_argb, 0.35),
                    radius=int(item.preview_radius),
                    glow_radius=preview_glow_radius,
                    size=max(8, min((item_bounds[2] - item_bounds[0]) // 3, (item_bounds[3] - item_bounds[1]) // 3)),
                )

    def _draw_numeric_label(
        self,
        *,
        text: str,
        center_x: int,
        top: int,
        color: int,
        scale: int,
    ) -> None:
        glyph_width = 5 * scale
        gap = max(1, scale)
        total_width = (len(text) * glyph_width) + (max(0, len(text) - 1) * gap)
        x = int(center_x - (total_width // 2))
        for char in text:
            self._draw_digit(x, top, char, color, scale)
            x += glyph_width + gap

    def _draw_digit(self, x: int, y: int, char: str, color: int, scale: int) -> None:
        segments_by_digit = {
            "0": "abcedf",
            "1": "bc",
            "2": "abged",
            "3": "abgcd",
            "4": "fgbc",
            "5": "afgcd",
            "6": "afgecd",
            "7": "abc",
            "8": "abcdefg",
            "9": "abfgcd",
        }
        segments = segments_by_digit.get(char, "")
        thickness = max(1, scale)
        width = 5 * scale
        height = 9 * scale
        mid_y = y + (height // 2)
        if "a" in segments:
            self._fill_rect(x + thickness, y, x + width - thickness, y + thickness, color)
        if "b" in segments:
            self._fill_rect(x + width - thickness, y + thickness, x + width, mid_y, color)
        if "c" in segments:
            self._fill_rect(x + width - thickness, mid_y, x + width, y + height - thickness, color)
        if "d" in segments:
            self._fill_rect(x + thickness, y + height - thickness, x + width - thickness, y + height, color)
        if "e" in segments:
            self._fill_rect(x, mid_y, x + thickness, y + height - thickness, color)
        if "f" in segments:
            self._fill_rect(x, y + thickness, x + thickness, mid_y, color)
        if "g" in segments:
            self._fill_rect(x + thickness, mid_y - (thickness // 2), x + width - thickness, mid_y + ((thickness + 1) // 2), color)

    def _relative_bounds(
        self,
        bounds: tuple[int, int, int, int],
        left: int,
        top: int,
    ) -> tuple[int, int, int, int]:
        return (bounds[0] - left, bounds[1] - top, bounds[2] - left, bounds[3] - top)

    @staticmethod
    def _offset_bounds(
        bounds: tuple[int, int, int, int],
        dx: int,
        dy: int,
    ) -> tuple[int, int, int, int]:
        return (bounds[0] + dx, bounds[1] + dy, bounds[2] + dx, bounds[3] + dy)

    @staticmethod
    def _inset_bounds(
        bounds: tuple[int, int, int, int],
        inset: int,
    ) -> tuple[int, int, int, int]:
        return (bounds[0] + inset, bounds[1] + inset, bounds[2] - inset, bounds[3] - inset)

    def _draw_round_rect(self, bounds: tuple[int, int, int, int], radius: int, color: int) -> None:
        left, top, right, bottom = bounds
        if right < left or bottom < top:
            return
        radius = max(0, min(radius, (right - left) // 2, (bottom - top) // 2))
        if radius <= 0:
            self._fill_rect(left, top, right, bottom, color)
            return
        self._fill_rect(left + radius, top, right - radius, bottom, color)
        self._fill_rect(left, top + radius, right, bottom - radius, color)
        self._draw_disc(left + radius, top + radius, radius, color)
        self._draw_disc(right - radius, top + radius, radius, color)
        self._draw_disc(left + radius, bottom - radius, radius, color)
        self._draw_disc(right - radius, bottom - radius, radius, color)

    def _fill_rect(self, left: int, top: int, right: int, bottom: int, color: int) -> None:
        if self._buffer is None:
            return
        left = max(0, int(left))
        top = max(0, int(top))
        right = min(self._width - 1, int(right))
        bottom = min(self._height - 1, int(bottom))
        if right < left or bottom < top:
            return
        for yy in range(top, bottom + 1):
            row = yy * self._width
            for xx in range(left, right + 1):
                self._buffer[row + xx] = color

    def _draw_line(self, x0: int, y0: int, x1: int, y1: int, color: int, radius: int) -> None:
        dx = int(x1 - x0)
        dy = int(y1 - y0)
        steps = max(1, abs(dx), abs(dy))
        for i in range(steps + 1):
            x = int(round(x0 + (dx * i) / steps))
            y = int(round(y0 + (dy * i) / steps))
            self._draw_disc(x, y, radius, color)

    def _draw_disc(self, x: int, y: int, radius: int, color: int) -> None:
        if self._buffer is None:
            return
        for yy in range(max(0, y - radius), min(self._height, y + radius + 1)):
            dy = yy - y
            for xx in range(max(0, x - radius), min(self._width, x + radius + 1)):
                dx = xx - x
                if dx * dx + dy * dy > radius * radius:
                    continue
                self._buffer[(yy * self._width) + xx] = color

    def _pump_messages(self) -> None:
        msg = _MSG()
        while self._user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, self._PM_REMOVE):
            self._user32.TranslateMessage(ctypes.byref(msg))
            self._user32.DispatchMessageW(ctypes.byref(msg))

    @staticmethod
    def _wnd_proc(hwnd, msg, wparam, lparam):
        return ctypes.windll.user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    @staticmethod
    def _raise_last_error(context: str) -> None:
        code = int(ctypes.get_last_error())
        if code:
            raise OSError(code, f"{context} failed", None, code)
        raise OSError(f"{context} failed")
