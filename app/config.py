from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    models_dir: Path = root / "models"
    data_dir: Path = root / "data"


@dataclass(frozen=True)
class PrimaryInteractionConfig:
    drag_start_distance: float = 0.045
    click_release_tolerance: float = 0.022
    double_click_window_s: float = 0.35
    hand_loss_grace_s: float = 0.18
    enable_double_click: bool = False


@dataclass(frozen=True)
class ClutchInteractionConfig:
    hand_loss_grace_s: float = 0.18


@dataclass(frozen=True)
class SecondaryInteractionConfig:
    release_tolerance: float = 0.028
    hand_loss_grace_s: float = 0.18


@dataclass(frozen=True)
class ScrollInteractionConfig:
    toggle_gesture_label: str = "SHAKA"
    dead_zone: float = 0.035
    axis_dominance_margin: float = 0.012
    pause_reset_s: float = 0.35
    pause_motion_epsilon: float = 0.006
    scroll_gain: float = 1.0
    hand_loss_grace_s: float = 0.18


@dataclass(frozen=True)
class CursorPolicyConfig:
    allowed_gestures: tuple[str, ...] = ("CLOSED_PALM",)
    provisional: bool = True


@dataclass(frozen=True)
class CursorPreviewConfig:
    move_epsilon: float = 0.003
    gain: float = 1.0


@dataclass(frozen=True)
class CursorSpaceConfig:
    anchor_mode: str = "palm_center"
    mirror_x: bool = True


@dataclass(frozen=True)
class ExecutionConfig:
    profile: str = "dry_run"
    enable_live_os: bool = False
    enable_live_cursor: bool = False
    enable_live_primary: bool = False
    enable_live_secondary: bool = False
    enable_live_scroll: bool = False
    enable_live_presentation: bool = False
    scroll_units_per_motion: float = 1200.0


@dataclass(frozen=True)
class ExecutionSafetyConfig:
    suppress_on_feature_instability: bool = True
    suppress_on_hand_loss: bool = True


@dataclass(frozen=True)
class OperatorLifecycleConfig:
    manual_exit_keys: tuple[str, ...] = ("ESC", "Q")
    enable_gesture_exit: bool = True
    gesture_exit_label: str = "THUMBS_DOWN"
    gesture_exit_min_hold_frames: int = 2
    startup_manual_exit_guard_s: float = 0.75


@dataclass(frozen=True)
class OperatorOverrideConfig:
    execution_override: str = "fallback_live"
    routing_override: str = "auto"


@dataclass(frozen=True)
class PresentationContextConfig:
    powerpoint_processes: tuple[str, ...] = ("powerpnt.exe",)
    browser_processes: tuple[str, ...] = ("chrome.exe", "msedge.exe", "firefox.exe", "brave.exe")
    pdf_processes: tuple[str, ...] = ("acrord32.exe", "acrobat.exe", "sumatrapdf.exe", "foxitpdfreader.exe")
    browser_title_keywords: tuple[str, ...] = ("google slides", "powerpoint", "slide show", "slideshow", "prezi", "canva")
    fullscreen_margin_px: int = 24


@dataclass(frozen=True)
class PresentationRuntimeConfig:
    navigation_confirm_frames: int = 1
    session_control_confirm_frames: int = 2
    release_grace_frames: int = 2


@dataclass(frozen=True)
class PresentationToolConfig:
    enable_live_presentation_tools: bool = True
    pointer_anchor_mode: str = "index_tip"
    pointer_edge_blend_margin: float = 0.14
    pointer_edge_blend_max_pull: float = 0.24
    pointer_input_x_min: float = 0.0
    pointer_input_x_max: float = 1.0
    pointer_input_y_min: float = 0.0
    pointer_input_y_max: float = 0.76
    laser_toggle_confirm_frames: int = 2
    laser_hold_last_frames: int = 3
    draw_toggle_confirm_frames: int = 2
    draw_activation_confirm_frames: int = 1
    draw_undo_min_ms: int = 60
    draw_undo_max_ms: int = 650
    draw_undo_min_detected_frames: int = 1
    draw_clear_hold_ms: int = 800
    draw_clear_confirm_frames: int = 3
    draw_clear_feedback_frames: int = 6
    draw_release_grace_frames: int = 2
    panel_pointer_smoothing_alpha: float = 0.22
    panel_pointer_slow_alpha: float = 0.22
    panel_pointer_slow_scale: float = 0.48
    panel_pointer_history_size: int = 6
    panel_select_confirm_frames: int = 2
    panel_toggle_confirm_frames: int = 2
    panel_min_open_ms: int = 5000
    panel_leave_grace_frames: int = 6
    panel_leave_padding: float = 0.022
    panel_slow_padding: float = 0.018
    laser_smoothing_alpha: float = 1.0
    laser_smoothing_alpha_min: float = 0.38
    laser_smoothing_alpha_max: float = 0.88
    laser_speed_low: float = 0.006
    laser_speed_high: float = 0.040
    laser_history_size: int = 6
    draw_smoothing_alpha: float = 0.42
    draw_hold_last_frames: int = 3
    draw_idle_history_size: int = 4
    draw_stroke_smoothing_alpha_min: float = 0.68
    draw_stroke_smoothing_alpha_max: float = 0.94
    draw_stroke_speed_low: float = 0.006
    draw_stroke_speed_high: float = 0.040
    draw_stroke_history_size: int = 4
    stroke_curve_subdivisions: int = 5
    stroke_min_point_delta_px: int = 2
    default_draw_color_key: str = "gold"
    default_draw_pen_key: str = "marker"
    default_draw_size_key: str = "10"
    panel_margin_right: float = 0.016
    panel_top_y: float = 0.10
    panel_width: float = 0.155
    panel_height: float = 0.72
    panel_anchor_x: float = 0.976
    panel_anchor_y: float = 0.948
    panel_reveal_radius: float = 0.045
    panel_idle_radius: float = 0.028
    panel_activation_radius: float = 0.185
    panel_dismiss_radius: float = 0.235
    panel_slow_zone_radius: float = 0.20
    panel_block_radius: float = 0.14
    panel_item_hit_radius: float = 0.045
    panel_animation_lerp: float = 0.28


@dataclass(frozen=True)
class RuntimeDebugConfig:
    pipeline_trace: bool = True


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "Virtual Gesture Mouse"
    target_fps: int = 30
    paths: Paths = field(default_factory=Paths)

    # Camera
    camera_index: int = 0
    cam_width: int = 640
    cam_height: int = 480
    requested_fps: int = 30
    prefer_mjpg: bool = True

    # Exposure (use auto unless you *must* lock it)
    use_auto_exposure: bool = True
    exposure_value: float = -4.0   # only used if auto exposure is off
    gain_value: float = 10.0       # only used if auto exposure is off

    # View
    mirror_view: bool = True
    tracker_input_is_mirrored: bool = False

    # Detection performance (these two strongly affect "No hand detected")
    detect_scale: float = 0.4
    inference_stride: int = 1

    # Preprocessing
    enable_preprocessing: bool = True
    primary_interaction: PrimaryInteractionConfig = field(default_factory=PrimaryInteractionConfig)
    clutch_interaction: ClutchInteractionConfig = field(default_factory=ClutchInteractionConfig)
    secondary_interaction: SecondaryInteractionConfig = field(default_factory=SecondaryInteractionConfig)
    scroll_interaction: ScrollInteractionConfig = field(default_factory=ScrollInteractionConfig)
    cursor_space: CursorSpaceConfig = field(default_factory=CursorSpaceConfig)
    cursor_policy: CursorPolicyConfig = field(default_factory=CursorPolicyConfig)
    cursor_preview: CursorPreviewConfig = field(default_factory=CursorPreviewConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    execution_safety: ExecutionSafetyConfig = field(default_factory=ExecutionSafetyConfig)
    operator_lifecycle: OperatorLifecycleConfig = field(default_factory=OperatorLifecycleConfig)
    operator_override: OperatorOverrideConfig = field(default_factory=OperatorOverrideConfig)
    presentation_context: PresentationContextConfig = field(default_factory=PresentationContextConfig)
    presentation_runtime: PresentationRuntimeConfig = field(default_factory=PresentationRuntimeConfig)
    presentation_tools: PresentationToolConfig = field(default_factory=PresentationToolConfig)
    runtime_debug: RuntimeDebugConfig = field(default_factory=RuntimeDebugConfig)


CONFIG = AppConfig()
