from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    models_dir: Path = root / "models"
    data_dir: Path = root / "data"


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


CONFIG = AppConfig()
