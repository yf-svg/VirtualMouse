from __future__ import annotations

from dataclasses import dataclass
import time

from app.config import CONFIG, PresentationToolConfig
from app.control.presentation_panel import (
    COLOR_OPTIONS_BY_KEY,
    PEN_OPTIONS_BY_KEY,
    SIZE_OPTIONS_BY_KEY,
    default_draw_color_key,
    default_draw_pen_key,
    default_draw_size_key,
    panel_frame,
    panel_item_layouts,
    panel_sections,
)
from app.control.cursor_space import CursorPoint
from app.control.mouse import ScreenPoint
from app.control.pointer_filters import AdaptiveEmaFilter, DropoutHold
from app.control.presentation_overlay import (
    NoOpPresentationOverlayBackend,
    PresentationDrawStyle,
    PresentationOverlayBackend,
    PresentationOverlayState,
    PresentationPanelItemState,
    PresentationPanelRenderState,
    PresentationStroke,
    WindowsPresentationOverlayBackend,
    map_pointer_to_window,
)
from app.modes.presentation_tools import PresentationToolOut, PresentationToolState


@dataclass(frozen=True)
class PresentationToolExecutionReport:
    performed: bool
    reason: str
    visible: bool
    laser_point: ScreenPoint | None
    draw_point: ScreenPoint | None = None
    stroke_count: int = 0


@dataclass
class _BufferedStroke:
    points: list[CursorPoint]
    style: PresentationDrawStyle
    cached_window_rect: tuple[int, int, int, int] | None = None
    cached_render_points: tuple[ScreenPoint, ...] = ()


class PresentationToolExecutor:
    def __init__(
        self,
        *,
        live_enabled: bool,
        cfg: PresentationToolConfig | None = None,
        overlay_backend: PresentationOverlayBackend | None = None,
    ):
        self.cfg = cfg or CONFIG.presentation_tools
        self.live_enabled = bool(live_enabled and self.cfg.enable_live_presentation_tools)
        if overlay_backend is not None:
            self.overlay_backend = overlay_backend
        elif self.live_enabled:
            self.overlay_backend = WindowsPresentationOverlayBackend()
        else:
            self.overlay_backend = NoOpPresentationOverlayBackend()
        self._strokes: list[_BufferedStroke] = []
        self._active_stroke: list[CursorPoint] | None = None
        self._active_stroke_style: PresentationDrawStyle | None = None
        self._laser_filter = AdaptiveEmaFilter(
            history_size=max(2, int(getattr(self.cfg, "laser_history_size", 6) or 6)),
            alpha_min=float(getattr(self.cfg, "laser_smoothing_alpha_min", self.cfg.laser_smoothing_alpha)),
            alpha_max=float(getattr(self.cfg, "laser_smoothing_alpha_max", self.cfg.laser_smoothing_alpha)),
            speed_low=float(getattr(self.cfg, "laser_speed_low", 0.006)),
            speed_high=float(getattr(self.cfg, "laser_speed_high", 0.040)),
        )
        self._laser_hold = DropoutHold(
            hold_frames=max(0, int(getattr(self.cfg, "laser_hold_last_frames", 0) or 0))
        )
        self._draw_idle_filter = AdaptiveEmaFilter(
            history_size=max(2, int(getattr(self.cfg, "draw_idle_history_size", 4) or 4)),
            alpha_min=float(self.cfg.draw_smoothing_alpha),
            alpha_max=float(self.cfg.draw_smoothing_alpha),
        )
        self._draw_hold = DropoutHold(
            hold_frames=max(0, int(getattr(self.cfg, "draw_hold_last_frames", 0) or 0))
        )
        self._draw_stroke_filter = AdaptiveEmaFilter(
            history_size=max(2, int(getattr(self.cfg, "draw_stroke_history_size", 4) or 4)),
            alpha_min=float(getattr(self.cfg, "draw_stroke_smoothing_alpha_min", 0.68)),
            alpha_max=float(getattr(self.cfg, "draw_stroke_smoothing_alpha_max", 0.94)),
            speed_low=float(getattr(self.cfg, "draw_stroke_speed_low", 0.006)),
            speed_high=float(getattr(self.cfg, "draw_stroke_speed_high", 0.040)),
        )
        self._eraser_feedback_frames = 0
        self._panel_expansion = 0.0

    def apply(self, tool_out: PresentationToolOut, context) -> PresentationToolExecutionReport:
        if not self.live_enabled:
            self._strokes.clear()
            self._active_stroke = None
            self._active_stroke_style = None
            self._reset_smoothed_points()
            self._eraser_feedback_frames = 0
            self._panel_expansion = 0.0
            self.overlay_backend.reset()
            return PresentationToolExecutionReport(False, "presentation_tools_disabled_by_policy", False, None)

        self._update_smoothing_state(tool_out.state)
        context_ready = bool(context.allowed and context.confident and context.window_rect is not None)
        if not context_ready:
            self._finish_active_stroke()
            self._reset_smoothed_points()
            self._eraser_feedback_frames = 0
            self._panel_expansion = self._approach_panel_expansion(0.0)
            self.overlay_backend.reset()
            if context.window_rect is None:
                return PresentationToolExecutionReport(
                    False,
                    "presentation_tool_missing_window_rect",
                    False,
                    None,
                    stroke_count=len(self._strokes),
                )
            return PresentationToolExecutionReport(
                False,
                f"context_not_allowed:{context.reason}",
                False,
                None,
                stroke_count=len(self._strokes),
            )

        draw_style = self._resolve_draw_style(tool_out)
        smoothed_pointer = self._resolve_pointer(tool_out)
        point = map_pointer_to_window(smoothed_pointer, context.window_rect)
        erased = False
        if tool_out.intent.action_name in {"PRESENTATION_DRAW_CLEAR", "PRESENTATION_DRAW_UNDO"}:
            self._finish_active_stroke(window_rect=context.window_rect)
        if tool_out.intent.action_name == "PRESENTATION_DRAW_CLEAR":
            self._strokes.clear()
            erased = True
            self._eraser_feedback_frames = max(1, int(self.cfg.draw_clear_feedback_frames))
        elif tool_out.intent.action_name == "PRESENTATION_DRAW_UNDO":
            if self._strokes:
                self._strokes.pop()
            erased = True
            self._eraser_feedback_frames = max(1, int(self.cfg.draw_clear_feedback_frames))
        if tool_out.stroke_capturing and smoothed_pointer is not None and point is not None:
            self._append_stroke_point(smoothed_pointer, point, context.window_rect, draw_style)
        else:
            self._finish_active_stroke(window_rect=context.window_rect)

        strokes = self._map_strokes(context.window_rect)
        laser_point = point if tool_out.state == PresentationToolState.LASER else None
        draw_point = point if tool_out.state in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING} else None
        self._panel_expansion = self._approach_panel_expansion(1.0 if tool_out.panel_state.expanded else 0.0)
        panel_state = self._build_panel_render_state(
            tool_out=tool_out,
            window_rect=context.window_rect,
            draw_style=draw_style,
        )
        draw_cursor_style = self._resolve_draw_cursor_style(
            state=tool_out.state,
            draw_point=draw_point,
            erased=erased,
        )
        visible = bool(laser_point or draw_point or strokes or (panel_state is not None and (panel_state.visible or panel_state.expansion > 0.02)))
        if not visible:
            self._eraser_feedback_frames = 0
            self.overlay_backend.reset()
            return PresentationToolExecutionReport(False, "presentation_tool_not_visible", False, None)

        self.overlay_backend.render(
            PresentationOverlayState(
                visible=True,
                window_rect=context.window_rect,
                laser_point=laser_point,
                draw_point=draw_point,
                draw_cursor_style=draw_cursor_style,
                draw_style=draw_style,
                strokes=strokes,
                panel=panel_state,
            )
        )
        if self._eraser_feedback_frames > 0:
            self._eraser_feedback_frames -= 1
        reason = self._report_reason(
            state=tool_out.state,
            intent_action_name=tool_out.intent.action_name,
            laser_point=laser_point,
            draw_point=draw_point,
            stroke_count=len(strokes),
            erased=erased,
        )
        return PresentationToolExecutionReport(
            True,
            reason,
            True,
            laser_point,
            draw_point=draw_point,
            stroke_count=len(strokes),
        )

    def reset(self, *, reason: str = "presentation_tool_reset") -> PresentationToolExecutionReport:
        self._strokes.clear()
        self._active_stroke = None
        self._active_stroke_style = None
        self._reset_smoothed_points()
        self._eraser_feedback_frames = 0
        self._panel_expansion = 0.0
        self.overlay_backend.reset()
        return PresentationToolExecutionReport(False, reason, False, None)

    def close(self) -> None:
        self._strokes.clear()
        self._active_stroke = None
        self._active_stroke_style = None
        self._reset_smoothed_points()
        self._eraser_feedback_frames = 0
        self._panel_expansion = 0.0
        self.overlay_backend.close()

    def _append_stroke_point(
        self,
        cursor_point: CursorPoint,
        mapped_point: ScreenPoint,
        window_rect: tuple[int, int, int, int],
        draw_style: PresentationDrawStyle,
    ) -> None:
        if self._active_stroke is None:
            self._active_stroke = [cursor_point]
            self._active_stroke_style = draw_style
            return
        last_cursor_point = self._active_stroke[-1]
        if last_cursor_point == cursor_point:
            return
        last_mapped = map_pointer_to_window(last_cursor_point, window_rect)
        if last_mapped is not None and self._points_too_close(last_mapped, mapped_point):
            return
        self._active_stroke.append(cursor_point)

    def _finish_active_stroke(
        self,
        *,
        window_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        if self._active_stroke and self._active_stroke_style is not None:
            stroke = _BufferedStroke(
                points=list(self._active_stroke),
                style=self._active_stroke_style,
            )
            if window_rect is not None:
                stroke.cached_window_rect = window_rect
                stroke.cached_render_points = self._build_render_points(stroke.points, window_rect)
            self._strokes.append(stroke)
        self._active_stroke = None
        self._active_stroke_style = None

    def _map_strokes(self, window_rect: tuple[int, int, int, int]) -> tuple[PresentationStroke, ...]:
        mapped: list[PresentationStroke] = []
        for stroke in self._strokes:
            points = self._render_points_for_committed_stroke(stroke, window_rect)
            if points:
                mapped.append(
                    PresentationStroke(
                        points=points,
                        color_argb=stroke.style.color_argb,
                        glow_color_argb=stroke.style.glow_color_argb,
                        radius=stroke.style.radius,
                        glow_radius=stroke.style.glow_radius,
                        pen_kind=stroke.style.pen_kind,
                    )
                )
        if self._active_stroke and self._active_stroke_style is not None:
            points = self._build_render_points(self._active_stroke, window_rect)
            if points:
                mapped.append(
                    PresentationStroke(
                        points=points,
                        color_argb=self._active_stroke_style.color_argb,
                        glow_color_argb=self._active_stroke_style.glow_color_argb,
                        radius=self._active_stroke_style.radius,
                        glow_radius=self._active_stroke_style.glow_radius,
                        pen_kind=self._active_stroke_style.pen_kind,
                    )
                )
        return tuple(mapped)

    def _render_points_for_committed_stroke(
        self,
        stroke: _BufferedStroke,
        window_rect: tuple[int, int, int, int],
    ) -> tuple[ScreenPoint, ...]:
        if stroke.cached_window_rect != window_rect or not stroke.cached_render_points:
            stroke.cached_window_rect = window_rect
            stroke.cached_render_points = self._build_render_points(stroke.points, window_rect)
        return stroke.cached_render_points

    def _build_render_points(
        self,
        cursor_points: list[CursorPoint],
        window_rect: tuple[int, int, int, int],
    ) -> tuple[ScreenPoint, ...]:
        mapped = tuple(
            point
            for cursor_point in cursor_points
            if (point := map_pointer_to_window(cursor_point, window_rect)) is not None
        )
        if not mapped:
            return ()
        simplified = _dedupe_screen_points(
            mapped,
            threshold=max(1, int(self.cfg.stroke_min_point_delta_px)),
        )
        return _curve_screen_points(
            simplified,
            subdivisions=max(1, int(getattr(self.cfg, "stroke_curve_subdivisions", 1) or 1)),
        )

    def _points_too_close(self, first: ScreenPoint, second: ScreenPoint) -> bool:
        threshold = max(1, int(self.cfg.stroke_min_point_delta_px))
        return (abs(int(first.x) - int(second.x)) + abs(int(first.y) - int(second.y))) < threshold

    def _resolve_pointer(self, tool_out: PresentationToolOut) -> CursorPoint | None:
        pointer = tool_out.pointer_point
        timestamp = time.monotonic()
        if tool_out.state == PresentationToolState.LASER:
            if pointer is None:
                return self._laser_hold.apply(None)
            legacy_alpha = max(0.0, min(1.0, float(self.cfg.laser_smoothing_alpha)))
            if legacy_alpha < 1.0:
                smoothed = self._laser_filter.update(
                    pointer,
                    timestamp=timestamp,
                    alpha_min=legacy_alpha,
                    alpha_max=legacy_alpha,
                )
            else:
                smoothed = self._laser_filter.update(pointer, timestamp=timestamp)
            return self._laser_hold.apply(smoothed)
        if pointer is None:
            return self._draw_hold.apply(None)
        if tool_out.state == PresentationToolState.DRAW_IDLE:
            alpha = max(0.0, min(1.0, float(self.cfg.draw_smoothing_alpha)))
            smoothed = self._draw_idle_filter.update(
                pointer,
                timestamp=timestamp,
                alpha_min=alpha,
                alpha_max=alpha,
            )
            return self._draw_hold.apply(smoothed)
        if tool_out.state == PresentationToolState.DRAW_STROKING:
            smoothed = self._draw_stroke_filter.update(pointer, timestamp=timestamp)
            return self._draw_hold.apply(smoothed)
        return pointer

    def _update_smoothing_state(self, state: PresentationToolState) -> None:
        if state != PresentationToolState.LASER:
            self._laser_filter.reset()
            self._laser_hold.reset()
        if state != PresentationToolState.DRAW_IDLE:
            self._draw_idle_filter.reset()
        if state != PresentationToolState.DRAW_STROKING:
            self._draw_stroke_filter.reset()
        if state not in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING}:
            self._draw_hold.reset()
            self._eraser_feedback_frames = 0

    def _reset_smoothed_points(self) -> None:
        self._laser_filter.reset()
        self._laser_hold.reset()
        self._draw_idle_filter.reset()
        self._draw_hold.reset()
        self._draw_stroke_filter.reset()

    def _resolve_draw_style(self, tool_out: PresentationToolOut) -> PresentationDrawStyle:
        color = self._resolve_color_option(tool_out.selected_color_key)
        pen = self._resolve_pen_option(tool_out.selected_pen_key)
        size = self._resolve_size_option(tool_out.selected_size_key)
        return self._compose_draw_style(color=color, pen=pen, size=size)

    def _approach_panel_expansion(self, target: float) -> float:
        target = max(0.0, min(1.0, float(target)))
        alpha = max(0.0, min(1.0, float(self.cfg.panel_animation_lerp)))
        if alpha >= 1.0:
            return target
        return self._panel_expansion + ((target - self._panel_expansion) * alpha)

    def _build_panel_render_state(
        self,
        *,
        tool_out: PresentationToolOut,
        window_rect: tuple[int, int, int, int],
        draw_style: PresentationDrawStyle,
    ) -> PresentationPanelRenderState | None:
        if not tool_out.panel_state.visible and self._panel_expansion <= 0.02:
            return None
        frame_bounds = _map_bounds_to_window(panel_frame(self._panel_expansion, cfg=self.cfg), window_rect)
        sections = panel_sections(self._panel_expansion, cfg=self.cfg)
        color_section_bounds = _map_bounds_to_window(sections.colors_bounds, window_rect)
        pen_section_bounds = _map_bounds_to_window(sections.pens_bounds, window_rect)
        size_section_bounds = _map_bounds_to_window(sections.sizes_bounds, window_rect)
        if frame_bounds is None or color_section_bounds is None or pen_section_bounds is None or size_section_bounds is None:
            return None
        items: list[PresentationPanelItemState] = []
        selected_color = self._resolve_color_option(tool_out.selected_color_key)
        selected_pen = self._resolve_pen_option(tool_out.selected_pen_key)
        selected_size = self._resolve_size_option(tool_out.selected_size_key)
        for layout in panel_item_layouts(self._panel_expansion, cfg=self.cfg):
            mapped_center = map_pointer_to_window(layout.center, window_rect)
            mapped_bounds = _map_bounds_to_window(layout.bounds, window_rect)
            if mapped_center is None or mapped_bounds is None:
                continue
            if layout.kind == "color":
                color = COLOR_OPTIONS_BY_KEY[layout.key]
                items.append(
                    PresentationPanelItemState(
                        item_id=layout.option_id,
                        kind="color",
                        bounds=mapped_bounds,
                        center=mapped_center,
                        fill_argb=color.swatch_argb,
                        active=(tool_out.selected_color_key == layout.key),
                        hovered=(tool_out.panel_state.hovered_option_id == layout.option_id),
                    )
                )
                continue
            if layout.kind == "pen":
                pen = self._resolve_pen_option(layout.key)
                preview_style = self._compose_draw_style(color=selected_color, pen=pen, size=selected_size)
                items.append(
                    PresentationPanelItemState(
                        item_id=layout.option_id,
                        kind="pen",
                        bounds=mapped_bounds,
                        center=mapped_center,
                        fill_argb=selected_color.swatch_argb,
                        preview_argb=preview_style.color_argb,
                        preview_glow_argb=preview_style.glow_color_argb,
                        preview_radius=max(1, int(preview_style.radius)),
                        preview_kind=preview_style.pen_kind,
                        active=(tool_out.selected_pen_key == layout.key),
                        hovered=(tool_out.panel_state.hovered_option_id == layout.option_id),
                    )
                )
                continue
            size = self._resolve_size_option(layout.key)
            preview_style = self._compose_draw_style(color=selected_color, pen=selected_pen, size=size)
            items.append(
                PresentationPanelItemState(
                    item_id=layout.option_id,
                    kind="size",
                    bounds=mapped_bounds,
                    center=mapped_center,
                    fill_argb=selected_color.swatch_argb,
                    preview_argb=preview_style.color_argb,
                    preview_glow_argb=preview_style.glow_color_argb,
                    preview_radius=max(1, int(preview_style.radius)),
                    preview_kind=preview_style.pen_kind,
                    label_text=size.label,
                    active=(tool_out.selected_size_key == layout.key),
                    hovered=(tool_out.panel_state.hovered_option_id == layout.option_id),
                )
            )
        return PresentationPanelRenderState(
            visible=tool_out.panel_state.visible,
            frame=frame_bounds,
            color_section_bounds=color_section_bounds,
            pen_section_bounds=pen_section_bounds,
            size_section_bounds=size_section_bounds,
            expansion=self._panel_expansion,
            active_color_argb=selected_color.swatch_argb,
            active_pen_radius=max(1, int(draw_style.radius)),
            active_pen_glow_radius=max(2, int(draw_style.glow_radius)),
            active_pen_kind=draw_style.pen_kind,
            items=tuple(items),
        )

    def _resolve_color_option(self, key: str):
        return COLOR_OPTIONS_BY_KEY.get(key) or COLOR_OPTIONS_BY_KEY[default_draw_color_key(self.cfg)]

    def _resolve_pen_option(self, key: str):
        return PEN_OPTIONS_BY_KEY.get(key) or PEN_OPTIONS_BY_KEY[default_draw_pen_key(self.cfg)]

    def _resolve_size_option(self, key: str):
        return SIZE_OPTIONS_BY_KEY.get(key) or SIZE_OPTIONS_BY_KEY[default_draw_size_key(self.cfg)]

    def _compose_draw_style(self, *, color, pen, size) -> PresentationDrawStyle:
        base_width = max(1.0, float(getattr(size, "width", 10)))
        width_scale = max(0.2, float(getattr(pen, "width_scale", 1.0)))
        glow_scale = max(width_scale, float(getattr(pen, "glow_scale", 1.0)))
        alpha_scale = max(0.12, min(1.0, float(getattr(pen, "alpha_scale", 1.0))))
        radius = max(1, int(round(base_width * width_scale * 0.5)))
        glow_radius = max(radius + 2, int(round(base_width * glow_scale * 0.5)))
        stroke_alpha_scale = alpha_scale if alpha_scale < 0.7 else 1.0
        return PresentationDrawStyle(
            color_argb=_scale_alpha(color.stroke_argb, stroke_alpha_scale),
            glow_color_argb=_scale_alpha(color.glow_argb, alpha_scale),
            radius=radius,
            glow_radius=glow_radius,
            pen_kind=str(getattr(pen, "pen_kind", getattr(pen, "key", "pen"))),
        )

    def _resolve_draw_cursor_style(
        self,
        *,
        state: PresentationToolState,
        draw_point: ScreenPoint | None,
        erased: bool,
    ) -> str | None:
        if draw_point is None:
            return None
        if state not in {PresentationToolState.DRAW_IDLE, PresentationToolState.DRAW_STROKING}:
            return None
        if erased or self._eraser_feedback_frames > 0:
            return "eraser"
        return "pen"

    @staticmethod
    def _report_reason(
        *,
        state: PresentationToolState,
        intent_action_name: str,
        laser_point: ScreenPoint | None,
        draw_point: ScreenPoint | None,
        stroke_count: int,
        erased: bool,
    ) -> str:
        if intent_action_name == "PRESENTATION_DRAW_CLEAR":
            if draw_point is not None:
                return "draw_cleared_visible"
            return "draw_cleared"
        if intent_action_name == "PRESENTATION_DRAW_UNDO":
            if draw_point is not None:
                return "draw_undo_visible"
            return "draw_undo"
        if laser_point is not None:
            return "laser_visible"
        if state == PresentationToolState.DRAW_STROKING:
            return "draw_stroking_visible"
        if draw_point is not None:
            return "draw_ready_visible"
        if stroke_count > 0:
            return "annotations_visible"
        if erased:
            return "draw_cleared"
        return "presentation_tool_not_visible"


def _map_bounds_to_window(
    bounds: tuple[float, float, float, float],
    window_rect: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    left_top = map_pointer_to_window(CursorPoint(bounds[0], bounds[1]), window_rect)
    right_bottom = map_pointer_to_window(CursorPoint(bounds[2], bounds[3]), window_rect)
    if left_top is None or right_bottom is None:
        return None
    return (left_top.x, left_top.y, right_bottom.x, right_bottom.y)


def _scale_alpha(color: int, factor: float) -> int:
    base_alpha = (int(color) >> 24) & 0xFF
    scaled_alpha = max(0, min(255, int(round(base_alpha * max(0.0, factor)))))
    return (scaled_alpha << 24) | (int(color) & 0x00FFFFFF)


def _dedupe_screen_points(
    points: tuple[ScreenPoint, ...],
    *,
    threshold: int,
) -> tuple[ScreenPoint, ...]:
    if len(points) <= 1:
        return points
    kept = [points[0]]
    threshold = max(1, int(threshold))
    for point in points[1:]:
        previous = kept[-1]
        if (abs(int(point.x) - int(previous.x)) + abs(int(point.y) - int(previous.y))) >= threshold:
            kept.append(point)
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return tuple(kept)


def _curve_screen_points(
    points: tuple[ScreenPoint, ...],
    *,
    subdivisions: int,
) -> tuple[ScreenPoint, ...]:
    if len(points) <= 2:
        return points
    subdivisions = max(1, int(subdivisions))
    if subdivisions <= 1:
        return points
    expanded = [points[0], *points, points[-1]]
    curved: list[ScreenPoint] = [points[0]]
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
            if candidate != curved[-1]:
                curved.append(candidate)
    if curved[-1] != points[-1]:
        curved.append(points[-1])
    return tuple(curved)
