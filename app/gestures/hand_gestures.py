from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from app.gestures.features import (
    finger_is_curled,
    finger_is_extended,
    finger_metrics,
    get_landmarks,
    hand_scale,
    summarize_hand_pose,
    thumb_is_extended,
    thumb_metrics,
)

# --- MediaPipe landmark indices ---
WRIST = 0

# Thumb
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

# Index
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8

# Middle
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12

# Ring
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16

# Pinky
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20


def _dist2d(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _dist3d(a, b) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = (a.z - b.z) if hasattr(a, "z") else 0.0
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _norm2(u: Tuple[float, float]) -> float:
    return math.hypot(u[0], u[1])


def _vec2(a, b) -> Tuple[float, float]:
    return (b.x - a.x, b.y - a.y)


def _dot2(u: Tuple[float, float], v: Tuple[float, float]) -> float:
    return u[0] * v[0] + u[1] * v[1]


@dataclass
class GestureCfg:
    use_3d: bool = True
    min_scale: float = 1e-4

    # Extended vs curled tests (ratio-based; scale invariant)
    extend_ratio: float = 1.12  # stricter -> higher
    curl_ratio: float = 1.02    # stricter curl -> lower

    # Open palm
    open_palm_require_thumb: bool = False
    thumb_ext_thresh: float = 0.48
    open_palm_min_extended: int = 4
    open_palm_max_curled: int = 1
    open_palm_max_near_palm: int = 1

    # L gesture perpendicular check
    l_perp_cos_max: float = 0.45

    # BRAVO / thumbs-up
    thumb_up_y_margin: float = 0.08     # normalized by scale
    bravo_thumb_far: float = 0.70       # dist(thumb_tip, wrist)/scale
    bravo_require_strict_curl: bool = True


class HandGestures:
    def __init__(self, cfg: GestureCfg = GestureCfg()):
        self.cfg = cfg

    def _lm(self, landmarks: Any):
        return get_landmarks(landmarks)

    def _d(self, a, b) -> float:
        return _dist3d(a, b) if self.cfg.use_3d else _dist2d(a, b)

    def _scale(self, lm) -> float:
        return max(self._d(lm[WRIST], lm[INDEX_MCP]), self.cfg.min_scale)

    def _palm_center(self, lm):
        pts = [lm[WRIST], lm[INDEX_MCP], lm[MIDDLE_MCP], lm[RING_MCP], lm[PINKY_MCP]]
        x = sum(p.x for p in pts) / len(pts)
        y = sum(p.y for p in pts) / len(pts)
        z = sum(getattr(p, "z", 0.0) for p in pts) / len(pts)
        return type("P", (), {"x": x, "y": y, "z": z})()

    # -------- finger state helpers --------
    def finger_extended(self, lm, tip: int, pip: int, mcp: int) -> bool:
        wrist = lm[WRIST]
        tip_p = lm[tip]
        pip_p = lm[pip]
        mcp_p = lm[mcp]

        finger_len = max(self._d(mcp_p, tip_p), 1e-6)
        tip_w = self._d(tip_p, wrist) / finger_len
        pip_w = self._d(pip_p, wrist) / finger_len
        return (tip_w / max(pip_w, 1e-6)) >= self.cfg.extend_ratio

    def finger_curled(self, lm, tip: int, pip: int, mcp: int) -> bool:
        wrist = lm[WRIST]
        tip_p = lm[tip]
        pip_p = lm[pip]
        mcp_p = lm[mcp]

        finger_len = max(self._d(mcp_p, tip_p), 1e-6)
        tip_w = self._d(tip_p, wrist) / finger_len
        pip_w = self._d(pip_p, wrist) / finger_len
        return (tip_w / max(pip_w, 1e-6)) <= self.cfg.curl_ratio

    def thumb_extended(self, lm) -> bool:
        scale = self._scale(lm)
        palm = self._palm_center(lm)
        d = self._d(lm[THUMB_TIP], palm) / scale
        return d >= self.cfg.thumb_ext_thresh

    # -------- gestures --------
    def detect_open_palm(self, landmarks: Any) -> bool:
        lm = self._lm(landmarks)
        summary = summarize_hand_pose(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)

        if summary.extended_count < self.cfg.open_palm_min_extended:
            return False
        if summary.curled_count > self.cfg.open_palm_max_curled:
            return False
        if summary.near_palm_count > self.cfg.open_palm_max_near_palm:
            return False

        if self.cfg.open_palm_require_thumb:
            return thumb_is_extended(summary.thumb, min_tip_to_palm=self.cfg.thumb_ext_thresh)

        # Relaxed thumb rule for natural open hands: thumb may be splayed sideways
        # instead of fully extended, but it should not be folded tightly into the palm.
        return summary.thumb.tip_to_palm >= 0.44

    # ✅ Numbers use ONLY the 4 non-thumb fingers to prevent BRAVO/ONE confusion
    def detect_numbers_1_to_5(self, landmarks: Any) -> Optional[str]:
        """
        Returns: ONE/TWO/THREE/FOUR/FIVE or None

        Rule:
          - FIVE = open palm (separate detector)
          - ONE..FOUR are based ONLY on index/middle/ring/pinky:
              count of extended among those 4
              AND all other non-thumb fingers must be curled
          - thumb is ignored for numbers
        """
        lm = self._lm(landmarks)

        if self.detect_open_palm(landmarks):
            return "FIVE"

        ext_i = self.finger_extended(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        ext_m = self.finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        ext_r = self.finger_extended(lm, RING_TIP, RING_PIP, RING_MCP)
        ext_p = self.finger_extended(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP)

        cur_i = self.finger_curled(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        cur_m = self.finger_curled(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        cur_r = self.finger_curled(lm, RING_TIP, RING_PIP, RING_MCP)
        cur_p = self.finger_curled(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP)

        ext = [ext_i, ext_m, ext_r, ext_p]
        n = sum(1 for v in ext if v)

        # Require that the NOT-extended fingers are curled (clean digit)
        def others_curled_ok() -> bool:
            ok = True
            if not ext_i:
                ok = ok and cur_i
            if not ext_m:
                ok = ok and cur_m
            if not ext_r:
                ok = ok and cur_r
            if not ext_p:
                ok = ok and cur_p
            return ok

        if n in (1, 2, 3, 4) and others_curled_ok():
            return {1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR"}[n]

        return None

    def detect_L(self, landmarks: Any) -> bool:
        lm = self._lm(landmarks)

        idx_ext = self.finger_extended(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        mid_ext = self.finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        ring_ext = self.finger_extended(lm, RING_TIP, RING_PIP, RING_MCP)
        pink_ext = self.finger_extended(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP)
        th_ext = self.thumb_extended(lm)

        if not (th_ext and idx_ext and mid_ext and ring_ext and pink_ext):
            return False

        i_vec = _vec2(lm[INDEX_MCP], lm[INDEX_TIP])
        t_vec = _vec2(lm[THUMB_MCP], lm[THUMB_TIP])
        ni = _norm2(i_vec)
        nt = _norm2(t_vec)
        if ni < 1e-9 or nt < 1e-9:
            return False

        cosv = abs(_dot2(i_vec, t_vec) / (ni * nt))
        return cosv <= self.cfg.l_perp_cos_max

    def detect_thumbs_up(self, landmarks: Any) -> bool:
        """
        ✅ BRAVO / THUMBS_UP (strict):
          - thumb is clearly UP (y ordering + above wrist)
          - thumb is far from wrist (dominant)
          - index/middle/ring/pinky are curled (STRICT)
          - explicit exclusion: if index looks extended => NOT BRAVO (prevents ONE confusion)
        """
        lm = self._lm(landmarks)
        scale = self._scale(lm)

        # Thumb "up" geometry using y order (smaller y is up)
        tip = lm[THUMB_TIP]
        ip = lm[THUMB_IP]
        mcp = lm[THUMB_MCP]
        wrist = lm[WRIST]

        # must be pointing upward-ish: tip above ip above mcp
        if not (tip.y < ip.y < mcp.y):
            return False

        # thumb tip must be above wrist by margin
        if (wrist.y - tip.y) <= (self.cfg.thumb_up_y_margin * scale):
            return False

        # thumb must be far from wrist (avoid false positives)
        if (self._d(tip, wrist) / scale) < self.cfg.bravo_thumb_far:
            return False

        # Other fingers must be curled (and index must NOT be extended)
        idx_ext = self.finger_extended(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        if idx_ext:
            return False  # explicit block against ONE

        idx_c = self.finger_curled(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        mid_c = self.finger_curled(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        ring_c = self.finger_curled(lm, RING_TIP, RING_PIP, RING_MCP)
        pink_c = self.finger_curled(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP)

        if self.cfg.bravo_require_strict_curl:
            return idx_c and mid_c and ring_c and pink_c

        return (idx_c + mid_c + ring_c + pink_c) >= 3  # type: ignore


_DEFAULT = HandGestures()

def detect_open_palm(landmarks: Any) -> bool:
    return _DEFAULT.detect_open_palm(landmarks)

def detect_number(landmarks: Any) -> Optional[str]:
    return _DEFAULT.detect_numbers_1_to_5(landmarks)

def detect_L(landmarks: Any) -> bool:
    return _DEFAULT.detect_L(landmarks)

def detect_bravo(landmarks: Any) -> bool:
    return _DEFAULT.detect_thumbs_up(landmarks)
