from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gestures.features import assess_hand_input_quality, extract_feature_vector


_PALM_IDS = (0, 5, 9, 13, 17)
_FINGER_GROUPS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_BASE_LANDMARKS = (
    (0.00, 0.00, 0.00),
    (-0.10, -0.04, 0.00), (-0.12, -0.12, 0.00), (-0.13, -0.21, 0.00), (-0.14, -0.30, 0.00),
    (-0.05, -0.03, 0.00), (-0.04, -0.16, 0.00), (-0.03, -0.27, 0.00), (-0.02, -0.38, 0.00),
    (0.00, -0.03, 0.00),  (0.00, -0.18, 0.00),  (0.00, -0.31, 0.00),  (0.00, -0.44, 0.00),
    (0.05, -0.03, 0.00),  (0.04, -0.16, 0.00),  (0.03, -0.28, 0.00),  (0.02, -0.40, 0.00),
    (0.10, -0.02, 0.00),  (0.10, -0.13, 0.00),  (0.09, -0.23, 0.00),  (0.08, -0.33, 0.00),
)


def _make_landmarks(
    *,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
):
    return [
        SimpleNamespace(
            x=(x * scale) + tx,
            y=(y * scale) + ty,
            z=(z * scale) + tz,
        )
        for x, y, z in _BASE_LANDMARKS
    ]


def _mean_abs_delta(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def _palm_center(landmarks) -> tuple[float, float, float]:
    xs = [landmarks[i].x for i in _PALM_IDS]
    ys = [landmarks[i].y for i in _PALM_IDS]
    zs = [landmarks[i].z for i in _PALM_IDS]
    return (
        sum(xs) / len(xs),
        sum(ys) / len(ys),
        sum(zs) / len(zs),
    )


def _make_user_variant(
    *,
    thumb_scale: float = 1.0,
    index_scale: float = 1.0,
    middle_scale: float = 1.0,
    ring_scale: float = 1.0,
    pinky_scale: float = 1.0,
    spread_scale: float = 1.0,
    base_scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 0.0,
):
    landmarks = _make_landmarks(scale=base_scale, tx=tx, ty=ty, tz=tz)
    center_x, center_y, center_z = _palm_center(landmarks)
    scale_map = {
        "thumb": thumb_scale,
        "index": index_scale,
        "middle": middle_scale,
        "ring": ring_scale,
        "pinky": pinky_scale,
    }

    out = []
    for idx, point in enumerate(landmarks):
        finger_scale = 1.0
        for finger_name, finger_ids in _FINGER_GROUPS.items():
            if idx in finger_ids:
                finger_scale = scale_map[finger_name]
                break

        dx = (point.x - center_x) * finger_scale
        dy = (point.y - center_y) * finger_scale
        dz = (point.z - center_z) * finger_scale
        if idx in (5, 9, 13, 17):
            dx *= spread_scale

        out.append(
            SimpleNamespace(
                x=center_x + dx,
                y=center_y + dy,
                z=center_z + dz,
            )
        )
    return out


class _WrappedDetection:
    def __init__(self, landmarks, *, background_tag: str, frame_bgr):
        self.landmarks = landmarks
        self.background_tag = background_tag
        self.frame_bgr = frame_bgr


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def main() -> None:
    base = extract_feature_vector(_make_landmarks()).values
    shifted = extract_feature_vector(
        _make_landmarks(scale=2.4, tx=0.6, ty=-0.35, tz=0.15)
    ).values
    translation_scale_delta = _mean_abs_delta(base, shifted)
    _assert(translation_scale_delta <= 1e-6, "Feature vector is not translation/scale invariant enough")

    plain = extract_feature_vector(
        _WrappedDetection(
            _make_landmarks(),
            background_tag="plain_wall",
            frame_bgr=[[0, 0], [0, 0]],
        )
    ).values
    cluttered = extract_feature_vector(
        {
            "landmarks": _WrappedDetection(
                _make_landmarks(),
                background_tag="busy_room",
                frame_bgr=[[255, 255], [64, 128]],
            )
        }
    ).values
    background_delta = _mean_abs_delta(plain, cluttered)
    _assert(background_delta <= 1e-6, "Feature extraction is unexpectedly sensitive to background metadata")

    user_deltas = []
    for variant in (
        _make_user_variant(
            thumb_scale=0.98,
            index_scale=1.01,
            middle_scale=1.0,
            ring_scale=0.99,
            pinky_scale=0.97,
            spread_scale=1.01,
            base_scale=1.2,
            tx=0.2,
            ty=-0.1,
        ),
        _make_user_variant(
            thumb_scale=1.03,
            index_scale=0.99,
            middle_scale=1.02,
            ring_scale=1.01,
            pinky_scale=1.04,
            spread_scale=1.03,
            base_scale=0.85,
            tx=-0.5,
            ty=0.3,
        ),
        _make_user_variant(
            thumb_scale=0.96,
            index_scale=1.04,
            middle_scale=1.03,
            ring_scale=0.98,
            pinky_scale=1.01,
            spread_scale=0.98,
            base_scale=1.6,
            tx=0.8,
            ty=0.2,
        ),
    ):
        user_deltas.append(_mean_abs_delta(base, extract_feature_vector(variant).values))
    max_user_delta = max(user_deltas)
    _assert(max_user_delta < 0.06, "Cross-user feature drift exceeds the accepted Phase 3 threshold")

    incomplete = assess_hand_input_quality(_make_landmarks()[:10])
    tiny = assess_hand_input_quality(_make_landmarks(scale=0.03))
    _assert(not incomplete.passed and incomplete.reason == "incomplete_landmarks", "Incomplete landmark gating failed")
    _assert(not tiny.passed and tiny.reason == "hand_too_small", "Tiny-hand gating failed")

    print("Phase 3 validation: PASS")
    print(f"- translation/scale delta: {translation_scale_delta:.10f}")
    print(f"- background metadata delta: {background_delta:.10f}")
    print(f"- max synthetic cross-user delta: {max_user_delta:.6f}")
    print(f"- incomplete landmark gate: {incomplete.reason}")
    print(f"- tiny-hand gate: {tiny.reason}")


if __name__ == "__main__":
    main()
