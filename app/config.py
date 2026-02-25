# from dataclasses import dataclass
# from pathlib import Path


# @dataclass(frozen=True)
# class Paths:
#     root: Path = Path(__file__).resolve().parents[1]
#     models_dir: Path = root / "models"
#     data_dir: Path = root / "data"


# @dataclass(frozen=True)
# class AppConfig:
#     app_name: str = "Virtual Gesture Mouse"
#     camera_index: int = 0
#     target_fps: int = 30
#     paths: Paths = Paths()


# CONFIG = AppConfig()

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path = Path(__file__).resolve().parents[1]
    models_dir: Path = root / "models"
    data_dir: Path = root / "data"


@dataclass(frozen=True)
class AppConfig:
    app_name: str = "Virtual Gesture Mouse"

    # Camera
    camera_index: int = 0
    cam_width: int = 640
    cam_height: int = 480

    # Performance / UX
    mirror_view: bool = False           # selfie-style
    enable_preprocessing: bool = False  # background robustness

    target_fps: int = 30
    paths: Paths = Paths()


CONFIG = AppConfig()