from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG, PresentationToolConfig
from app.control.cursor_space import CursorPoint


@dataclass(frozen=True)
class PresentationColorOption:
    key: str
    label: str
    swatch_argb: int
    stroke_argb: int
    glow_argb: int


@dataclass(frozen=True)
class PresentationPenOption:
    key: str
    label: str
    pen_kind: str
    width_scale: float
    glow_scale: float
    alpha_scale: float


@dataclass(frozen=True)
class PresentationSizeOption:
    key: str
    label: str
    width: int


@dataclass(frozen=True)
class PresentationPanelOptionSpec:
    option_id: str
    kind: str
    key: str
    row: int
    column: int


@dataclass(frozen=True)
class PresentationPanelItemLayout:
    option_id: str
    kind: str
    key: str
    center: CursorPoint
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class PresentationPanelSections:
    colors_bounds: tuple[float, float, float, float]
    pens_bounds: tuple[float, float, float, float]
    sizes_bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class PresentationPanelState:
    visible: bool
    expanded: bool
    hovered_option_id: str | None
    hovered_kind: str | None
    hovered_key: str | None
    blocked_by_panel: bool
    selected_color_key: str
    selected_pen_key: str
    selected_size_key: str


PRESENTATION_COLOR_OPTIONS: tuple[PresentationColorOption, ...] = (
    PresentationColorOption(
        key="gold",
        label="Gold",
        swatch_argb=0xF4FFD24A,
        stroke_argb=0xD0FFD84A,
        glow_argb=0x60FFF2A0,
    ),
    PresentationColorOption(
        key="coral",
        label="Coral",
        swatch_argb=0xF4FF8376,
        stroke_argb=0xD8FF907F,
        glow_argb=0x58FFD0BA,
    ),
    PresentationColorOption(
        key="azure",
        label="Azure",
        swatch_argb=0xF458CCFF,
        stroke_argb=0xD85ED4FF,
        glow_argb=0x5856F1FF,
    ),
    PresentationColorOption(
        key="mint",
        label="Mint",
        swatch_argb=0xF45BE9BB,
        stroke_argb=0xD864EDC1,
        glow_argb=0x5868FFD6,
    ),
    PresentationColorOption(
        key="magenta",
        label="Magenta",
        swatch_argb=0xF4FF71D8,
        stroke_argb=0xDEFF7AE0,
        glow_argb=0x60FFB3F0,
    ),
    PresentationColorOption(
        key="white",
        label="White",
        swatch_argb=0xF6F8FBFF,
        stroke_argb=0xECFFFFFF,
        glow_argb=0x66FFFFFF,
    ),
)

PRESENTATION_PEN_OPTIONS: tuple[PresentationPenOption, ...] = (
    PresentationPenOption(key="pen", label="Pen", pen_kind="pen", width_scale=0.52, glow_scale=0.72, alpha_scale=1.00),
    PresentationPenOption(key="marker", label="Marker", pen_kind="marker", width_scale=1.00, glow_scale=1.70, alpha_scale=0.88),
    PresentationPenOption(key="highlighter", label="Highlighter", pen_kind="highlighter", width_scale=1.18, glow_scale=2.10, alpha_scale=0.48),
    PresentationPenOption(key="brush", label="Brush", pen_kind="brush", width_scale=0.82, glow_scale=1.45, alpha_scale=0.92),
    PresentationPenOption(key="quill", label="Quill", pen_kind="quill", width_scale=0.64, glow_scale=1.05, alpha_scale=1.00),
)

PRESENTATION_SIZE_OPTIONS: tuple[PresentationSizeOption, ...] = (
    PresentationSizeOption(key="05", label="5", width=5),
    PresentationSizeOption(key="10", label="10", width=10),
    PresentationSizeOption(key="15", label="15", width=15),
    PresentationSizeOption(key="20", label="20", width=20),
    PresentationSizeOption(key="25", label="25", width=25),
)

COLOR_OPTIONS_BY_KEY = {option.key: option for option in PRESENTATION_COLOR_OPTIONS}
PEN_OPTIONS_BY_KEY = {option.key: option for option in PRESENTATION_PEN_OPTIONS}
SIZE_OPTIONS_BY_KEY = {option.key: option for option in PRESENTATION_SIZE_OPTIONS}

PRESENTATION_PANEL_OPTION_SPECS: tuple[PresentationPanelOptionSpec, ...] = (
    PresentationPanelOptionSpec("color:gold", "color", "gold", 0, 0),
    PresentationPanelOptionSpec("color:coral", "color", "coral", 0, 1),
    PresentationPanelOptionSpec("color:azure", "color", "azure", 1, 0),
    PresentationPanelOptionSpec("color:mint", "color", "mint", 1, 1),
    PresentationPanelOptionSpec("color:magenta", "color", "magenta", 2, 0),
    PresentationPanelOptionSpec("color:white", "color", "white", 2, 1),
    PresentationPanelOptionSpec("pen:pen", "pen", "pen", 0, 0),
    PresentationPanelOptionSpec("pen:marker", "pen", "marker", 1, 0),
    PresentationPanelOptionSpec("pen:highlighter", "pen", "highlighter", 2, 0),
    PresentationPanelOptionSpec("pen:brush", "pen", "brush", 3, 0),
    PresentationPanelOptionSpec("pen:quill", "pen", "quill", 4, 0),
    PresentationPanelOptionSpec("size:05", "size", "05", 0, 0),
    PresentationPanelOptionSpec("size:10", "size", "10", 1, 0),
    PresentationPanelOptionSpec("size:15", "size", "15", 2, 0),
    PresentationPanelOptionSpec("size:20", "size", "20", 3, 0),
    PresentationPanelOptionSpec("size:25", "size", "25", 4, 0),
)


def default_draw_color_key(cfg: PresentationToolConfig | None = None) -> str:
    cfg = cfg or CONFIG.presentation_tools
    key = str(getattr(cfg, "default_draw_color_key", PRESENTATION_COLOR_OPTIONS[0].key))
    if key in COLOR_OPTIONS_BY_KEY:
        return key
    return PRESENTATION_COLOR_OPTIONS[0].key


def default_draw_pen_key(cfg: PresentationToolConfig | None = None) -> str:
    cfg = cfg or CONFIG.presentation_tools
    key = str(getattr(cfg, "default_draw_pen_key", PRESENTATION_PEN_OPTIONS[1].key))
    if key in PEN_OPTIONS_BY_KEY:
        return key
    return PRESENTATION_PEN_OPTIONS[1].key


def default_draw_size_key(cfg: PresentationToolConfig | None = None) -> str:
    cfg = cfg or CONFIG.presentation_tools
    key = str(getattr(cfg, "default_draw_size_key", PRESENTATION_SIZE_OPTIONS[1].key))
    if key in SIZE_OPTIONS_BY_KEY:
        return key
    return PRESENTATION_SIZE_OPTIONS[1].key


def panel_frame(
    progress: float = 1.0,
    *,
    cfg: PresentationToolConfig | None = None,
) -> tuple[float, float, float, float]:
    cfg = cfg or CONFIG.presentation_tools
    progress = max(0.0, min(1.0, float(progress)))
    eased = _ease_out_cubic(progress)
    final_width = max(0.12, float(getattr(cfg, "panel_width", 0.155)))
    final_height = max(0.38, float(getattr(cfg, "panel_height", 0.72)))
    right = 1.0 - max(0.0, float(getattr(cfg, "panel_margin_right", 0.016)))
    top = max(0.0, float(getattr(cfg, "panel_top_y", 0.10)))
    width = final_width * max(0.22, eased)
    height = final_height * (0.74 + (0.26 * eased))
    left = max(0.0, right - width)
    bottom = min(1.0, top + height)
    return (left, top, right, bottom)


def panel_anchor(cfg: PresentationToolConfig | None = None) -> CursorPoint:
    left, top, right, bottom = panel_frame(1.0, cfg=cfg)
    return CursorPoint(x=right, y=(top + bottom) * 0.5)


def panel_sections(
    progress: float = 1.0,
    *,
    cfg: PresentationToolConfig | None = None,
) -> PresentationPanelSections:
    left, top, right, bottom = panel_frame(progress, cfg=cfg)
    width = right - left
    height = bottom - top
    inset_x = width * 0.09
    inset_y = height * 0.05
    section_gap = height * 0.022
    content_top = top + inset_y
    content_bottom = bottom - inset_y
    usable_height = max(0.001, content_bottom - content_top)
    colors_height = usable_height * 0.30
    pens_height = usable_height * 0.39
    sizes_height = usable_height - colors_height - pens_height - (section_gap * 2.0)
    colors_bounds = (
        left + inset_x,
        content_top,
        right - inset_x,
        content_top + colors_height,
    )
    pens_bounds = (
        left + inset_x,
        colors_bounds[3] + section_gap,
        right - inset_x,
        colors_bounds[3] + section_gap + pens_height,
    )
    sizes_bounds = (
        left + inset_x,
        pens_bounds[3] + section_gap,
        right - inset_x,
        min(content_bottom, pens_bounds[3] + section_gap + sizes_height),
    )
    return PresentationPanelSections(
        colors_bounds=colors_bounds,
        pens_bounds=pens_bounds,
        sizes_bounds=sizes_bounds,
    )


def panel_positions(
    progress: float,
    *,
    cfg: PresentationToolConfig | None = None,
) -> dict[str, CursorPoint]:
    return {item.option_id: item.center for item in panel_item_layouts(progress, cfg=cfg)}


def panel_item_layouts(
    progress: float = 1.0,
    *,
    cfg: PresentationToolConfig | None = None,
) -> tuple[PresentationPanelItemLayout, ...]:
    cfg = cfg or CONFIG.presentation_tools
    colors_specs = tuple(spec for spec in PRESENTATION_PANEL_OPTION_SPECS if spec.kind == "color")
    pen_specs = tuple(spec for spec in PRESENTATION_PANEL_OPTION_SPECS if spec.kind == "pen")
    size_specs = tuple(spec for spec in PRESENTATION_PANEL_OPTION_SPECS if spec.kind == "size")
    sections = panel_sections(progress, cfg=cfg)
    items: list[PresentationPanelItemLayout] = []
    items.extend(_layout_grid_items(colors_specs, sections.colors_bounds, rows=3, columns=2, gap_x=0.08, gap_y=0.08))
    items.extend(_layout_grid_items(pen_specs, sections.pens_bounds, rows=5, columns=1, gap_x=0.00, gap_y=0.07))
    items.extend(_layout_grid_items(size_specs, sections.sizes_bounds, rows=5, columns=1, gap_x=0.00, gap_y=0.08))
    return tuple(items)


def panel_contains_point(
    pointer_point: CursorPoint | None,
    *,
    padding: float = 0.0,
    cfg: PresentationToolConfig | None = None,
) -> bool:
    if pointer_point is None:
        return False
    left, top, right, bottom = panel_frame(1.0, cfg=cfg)
    left -= padding
    top -= padding
    right += padding
    bottom += padding
    return left <= pointer_point.x <= right and top <= pointer_point.y <= bottom


def resolve_panel_state(
    pointer_point: CursorPoint | None,
    *,
    draw_mode_active: bool,
    stroke_active: bool,
    selected_color_key: str,
    selected_pen_key: str,
    selected_size_key: str,
    panel_open: bool = False,
    cfg: PresentationToolConfig | None = None,
) -> PresentationPanelState:
    cfg = cfg or CONFIG.presentation_tools
    selected_color_key = selected_color_key if selected_color_key in COLOR_OPTIONS_BY_KEY else default_draw_color_key(cfg)
    selected_pen_key = selected_pen_key if selected_pen_key in PEN_OPTIONS_BY_KEY else default_draw_pen_key(cfg)
    selected_size_key = selected_size_key if selected_size_key in SIZE_OPTIONS_BY_KEY else default_draw_size_key(cfg)
    if not draw_mode_active or stroke_active or not panel_open:
        return PresentationPanelState(
            visible=False,
            expanded=False,
            hovered_option_id=None,
            hovered_kind=None,
            hovered_key=None,
            blocked_by_panel=False,
            selected_color_key=selected_color_key,
            selected_pen_key=selected_pen_key,
            selected_size_key=selected_size_key,
        )

    hovered_option_id = None
    hovered_kind = None
    hovered_key = None
    if pointer_point is not None:
        for item in panel_item_layouts(1.0, cfg=cfg):
            if _point_in_bounds(pointer_point, item.bounds):
                hovered_option_id = item.option_id
                hovered_kind = item.kind
                hovered_key = item.key
                break

    blocked_by_panel = panel_contains_point(pointer_point, cfg=cfg)
    if hovered_option_id is not None:
        blocked_by_panel = True

    return PresentationPanelState(
        visible=True,
        expanded=True,
        hovered_option_id=hovered_option_id,
        hovered_kind=hovered_kind,
        hovered_key=hovered_key,
        blocked_by_panel=blocked_by_panel,
        selected_color_key=selected_color_key,
        selected_pen_key=selected_pen_key,
        selected_size_key=selected_size_key,
    )


def _layout_grid_items(
    specs: tuple[PresentationPanelOptionSpec, ...],
    bounds: tuple[float, float, float, float],
    *,
    rows: int,
    columns: int,
    gap_x: float,
    gap_y: float,
) -> list[PresentationPanelItemLayout]:
    left, top, right, bottom = bounds
    width = max(0.001, right - left)
    height = max(0.001, bottom - top)
    gap_x_px = width * gap_x
    gap_y_px = height * gap_y
    cell_width = max(0.001, (width - (gap_x_px * max(0, columns - 1))) / max(1, columns))
    cell_height = max(0.001, (height - (gap_y_px * max(0, rows - 1))) / max(1, rows))
    layouts: list[PresentationPanelItemLayout] = []
    for spec in specs:
        item_left = left + (spec.column * (cell_width + gap_x_px))
        item_top = top + (spec.row * (cell_height + gap_y_px))
        item_right = item_left + cell_width
        item_bottom = item_top + cell_height
        center = CursorPoint(
            x=(item_left + item_right) * 0.5,
            y=(item_top + item_bottom) * 0.5,
        )
        layouts.append(
            PresentationPanelItemLayout(
                option_id=spec.option_id,
                kind=spec.kind,
                key=spec.key,
                center=center,
                bounds=(item_left, item_top, item_right, item_bottom),
            )
        )
    return layouts


def _point_in_bounds(point: CursorPoint, bounds: tuple[float, float, float, float]) -> bool:
    left, top, right, bottom = bounds
    return left <= point.x <= right and top <= point.y <= bottom


def _ease_out_cubic(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    return 1.0 - ((1.0 - value) ** 3)
