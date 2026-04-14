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


@dataclass(frozen=True)
class ClutchInteractionConfig:
    hand_loss_grace_s: float = 0.18


@dataclass(frozen=True)
class SecondaryInteractionConfig:
    release_tolerance: float = 0.028
    hand_loss_grace_s: float = 0.18


@dataclass(frozen=True)
class ScrollInteractionConfig:
    dead_zone: float = 0.035
    axis_dominance_margin: float = 0.012
    pause_reset_s: float = 0.35
    pause_motion_epsilon: float = 0.006
    scroll_gain: float = 1.0
    hand_loss_grace_s: float = 0.18


@dataclass(frozen=True)
class CursorPolicyConfig:
    allowed_gestures: tuple[str, ...] = ("L",)
    provisional: bool = True


@dataclass(frozen=True)
class CursorPreviewConfig:
    move_epsilon: float = 0.003
    gain: float = 1.0


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


@dataclass(frozen=True)
class OperatorOverrideConfig:
    execution_override: str = "inherit"
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

    # Detection performance (these two strongly affect "No hand detected")
    detect_scale: float = 0.4
    inference_stride: int = 1

    # Preprocessing
    enable_preprocessing: bool = True
    primary_interaction: PrimaryInteractionConfig = field(default_factory=PrimaryInteractionConfig)
    clutch_interaction: ClutchInteractionConfig = field(default_factory=ClutchInteractionConfig)
    secondary_interaction: SecondaryInteractionConfig = field(default_factory=SecondaryInteractionConfig)
    scroll_interaction: ScrollInteractionConfig = field(default_factory=ScrollInteractionConfig)
    cursor_policy: CursorPolicyConfig = field(default_factory=CursorPolicyConfig)
    cursor_preview: CursorPreviewConfig = field(default_factory=CursorPreviewConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    execution_safety: ExecutionSafetyConfig = field(default_factory=ExecutionSafetyConfig)
    operator_lifecycle: OperatorLifecycleConfig = field(default_factory=OperatorLifecycleConfig)
    operator_override: OperatorOverrideConfig = field(default_factory=OperatorOverrideConfig)
    presentation_context: PresentationContextConfig = field(default_factory=PresentationContextConfig)
    presentation_runtime: PresentationRuntimeConfig = field(default_factory=PresentationRuntimeConfig)


CONFIG = AppConfig()
