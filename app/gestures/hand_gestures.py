from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from app.gestures.features import (
    get_landmarks,
    summarize_hand_pose,
    thumb_is_extended,
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
INDEX_DIP = 7
INDEX_TIP = 8

# Middle
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12

# Ring
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16

# Pinky
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
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
    open_palm_min_adjacent_tip_x_ratio: float = 0.28
    open_palm_min_avg_tip_x_ratio: float = 0.24
    open_palm_min_cluster_ratio: float = 0.72
    open_palm_min_tip_span_ratio: float = 1.18
    open_palm_thumb_index_min_ratio: float = 0.84
    open_palm_thumb_min_tip_to_palm: float = 0.60

    # Closed palm
    closed_palm_min_chain_count: int = 3
    closed_palm_max_parallel_angle_deg: float = 35.0
    closed_palm_max_adjacent_tip_x_ratio: float = 0.39
    closed_palm_max_avg_tip_x_ratio: float = 0.30
    closed_palm_max_cluster_ratio: float = 0.90
    closed_palm_max_tip_span_ratio: float = 0.90
    closed_palm_thumb_min_tip_to_palm: float = 0.30
    closed_palm_thumb_max_tip_to_palm: float = 1.18
    closed_palm_thumb_index_max_ratio: float = 1.32

    # Peace sign: two raised fingers with ring/pinky folded and a compact thumb.
    two_finger_folded_y_margin: float = 0.0
    peace_min_angle_deg: float = 10.0
    peace_min_tip_gap_ratio: float = 0.28
    peace_thumb_ring_min_gap_ratio: float = 0.24
    peace_thumb_max_tip_to_palm: float = 0.78

    # Shaka: thumb + pinky extended with the middle three fingers curled.
    shaka_min_thumb_tip_distance: float = 0.55
    shaka_min_thumb_pinky_gap_ratio: float = 0.95

    # L gesture perpendicular check
    l_perp_cos_max: float = 0.45

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

    def _palm_width(self, lm) -> float:
        return max(self._d(lm[INDEX_MCP], lm[PINKY_MCP]), self.cfg.min_scale)

    def _adjacent_tip_x_ratios(self, lm, palm_width: float) -> list[float]:
        pairs = [
            (INDEX_TIP, MIDDLE_TIP),
            (MIDDLE_TIP, RING_TIP),
            (RING_TIP, PINKY_TIP),
        ]
        return [abs(lm[a].x - lm[b].x) / palm_width for a, b in pairs]

    def _non_thumb_fingers_extended(self, lm) -> bool:
        wrist = lm[WRIST]
        return (
            self._finger_chain_extended(lm, wrist, INDEX_MCP, INDEX_PIP, INDEX_TIP)
            and self._finger_chain_extended(lm, wrist, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP)
            and self._finger_chain_extended(lm, wrist, RING_MCP, RING_PIP, RING_TIP)
            and self._finger_chain_extended(lm, wrist, PINKY_MCP, PINKY_PIP, PINKY_TIP)
        )

    def _non_thumb_chain_count(self, lm) -> int:
        wrist = lm[WRIST]
        return sum(
            1 for mcp, pip, tip in (
                (INDEX_MCP, INDEX_PIP, INDEX_TIP),
                (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
                (RING_MCP, RING_PIP, RING_TIP),
                (PINKY_MCP, PINKY_PIP, PINKY_TIP),
            )
            if self._finger_chain_extended(lm, wrist, mcp, pip, tip)
        )

    def _finger_chain_extended(self, lm, wrist, mcp: int, pip: int, tip: int) -> bool:
        wrist_to_tip = self._d(wrist, lm[tip])
        wrist_to_pip = self._d(wrist, lm[pip])
        wrist_to_mcp = self._d(wrist, lm[mcp])
        return (
            wrist_to_tip > wrist_to_pip > wrist_to_mcp
            and lm[tip].y < lm[pip].y < lm[mcp].y
        )

    def _finger_vector(self, lm, mcp: int, tip: int) -> Tuple[float, float]:
        return _vec2(lm[mcp], lm[tip])

    def _angle_between(self, u: Tuple[float, float], v: Tuple[float, float]) -> float:
        nu = _norm2(u)
        nv = _norm2(v)
        if nu < 1e-9 or nv < 1e-9:
            return 180.0
        cosv = max(-1.0, min(1.0, _dot2(u, v) / (nu * nv)))
        return math.degrees(math.acos(cosv))

    def _adjacent_finger_angles(self, lm) -> list[float]:
        vectors = [
            self._finger_vector(lm, INDEX_MCP, INDEX_TIP),
            self._finger_vector(lm, MIDDLE_MCP, MIDDLE_TIP),
            self._finger_vector(lm, RING_MCP, RING_TIP),
            self._finger_vector(lm, PINKY_MCP, PINKY_TIP),
        ]
        return [
            self._angle_between(vectors[0], vectors[1]),
            self._angle_between(vectors[1], vectors[2]),
            self._angle_between(vectors[2], vectors[3]),
        ]

    def _tip_cluster_ratio(self, lm) -> float:
        tip_x = [lm[INDEX_TIP].x, lm[MIDDLE_TIP].x, lm[RING_TIP].x, lm[PINKY_TIP].x]
        tip_y = [lm[INDEX_TIP].y, lm[MIDDLE_TIP].y, lm[RING_TIP].y, lm[PINKY_TIP].y]
        mcp_y = [lm[INDEX_MCP].y, lm[MIDDLE_MCP].y, lm[RING_MCP].y, lm[PINKY_MCP].y]
        cluster_width = max(tip_x) - min(tip_x)
        cluster_height = max(sum(mcp_y) / len(mcp_y) - sum(tip_y) / len(tip_y), self.cfg.min_scale)
        return cluster_width / cluster_height

    def _tip_span_ratio(self, lm, palm_width: float) -> float:
        tip_x = [lm[INDEX_TIP].x, lm[MIDDLE_TIP].x, lm[RING_TIP].x, lm[PINKY_TIP].x]
        return (max(tip_x) - min(tip_x)) / palm_width

    def _finger_folded_toward_palm(self, lm, tip: int, pip: int, mcp: int) -> bool:
        return (
            lm[tip].y > (lm[pip].y + self.cfg.two_finger_folded_y_margin)
            and not self._finger_chain_extended(lm, lm[WRIST], mcp, pip, tip)
        )

    def _finger_raised(self, lm, tip: int, pip: int, mcp: int) -> bool:
        """
        Raised finger used by V-sign style gestures.
        Accept a clean extension chain, but also allow slight landmark noise as long
        as the finger is vertically ordered upward and not classified as curled.
        """
        return (
            self._finger_chain_extended(lm, lm[WRIST], mcp, pip, tip)
            or (
                lm[tip].y < lm[pip].y < lm[mcp].y
                and self.finger_extended(lm, tip, pip, mcp)
                and not self.finger_curled(lm, tip, pip, mcp)
            )
        )

    def _finger_folded_for_peace(self, lm, tip: int, pip: int, mcp: int) -> bool:
        """
        Folded finger for peace-sign recognition.
        Accept either a clear curl or the simpler image-space cue that the tip has
        dropped below the PIP joint.
        """
        return (
            self.finger_curled(lm, tip, pip, mcp)
            or self._finger_folded_toward_palm(lm, tip, pip, mcp)
            or lm[tip].y > lm[pip].y
        )

    def _two_finger_pose_metrics(self, landmarks: Any) -> Optional[dict[str, float]]:
        """
        Shared geometry for gestures with index and middle raised while ring and
        pinky remain folded.
        """
        lm = self._lm(landmarks)
        palm_width = self._palm_width(lm)

        if not self._finger_raised(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP):
            return None
        if not self._finger_raised(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP):
            return None
        if not self._finger_folded_for_peace(lm, RING_TIP, RING_PIP, RING_MCP):
            return None
        if not self._finger_folded_for_peace(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP):
            return None

        index_vector = self._finger_vector(lm, INDEX_MCP, INDEX_TIP)
        middle_vector = self._finger_vector(lm, MIDDLE_MCP, MIDDLE_TIP)
        return {
            "angle_deg": self._angle_between(index_vector, middle_vector),
            "tip_gap_ratio": self._d(lm[INDEX_TIP], lm[MIDDLE_TIP]) / palm_width,
        }

    def _closed_palm_metrics(self, landmarks: Any) -> tuple[bool, int, list[float], float, float, float, float]:
        lm = self._lm(landmarks)
        summary = summarize_hand_pose(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)
        palm_width = self._palm_width(lm)
        adjacent_tip_x_ratios = self._adjacent_tip_x_ratios(lm, palm_width)
        avg_tip_x_ratio = sum(adjacent_tip_x_ratios) / len(adjacent_tip_x_ratios)
        cluster_ratio = self._tip_cluster_ratio(lm)
        tip_span_ratio = self._tip_span_ratio(lm, palm_width)
        adjacent_finger_angles = self._adjacent_finger_angles(lm)
        thumb_to_index_ratio = self._d(lm[THUMB_TIP], lm[INDEX_MCP]) / palm_width
        chain_count = self._non_thumb_chain_count(lm)
        passed = (
            chain_count >= self.cfg.closed_palm_min_chain_count
            and max(adjacent_finger_angles) <= self.cfg.closed_palm_max_parallel_angle_deg
            and max(adjacent_tip_x_ratios) <= self.cfg.closed_palm_max_adjacent_tip_x_ratio
            and avg_tip_x_ratio <= self.cfg.closed_palm_max_avg_tip_x_ratio
            and cluster_ratio <= self.cfg.closed_palm_max_cluster_ratio
            and tip_span_ratio <= self.cfg.closed_palm_max_tip_span_ratio
            and self.cfg.closed_palm_thumb_min_tip_to_palm <= summary.thumb.tip_to_palm <= self.cfg.closed_palm_thumb_max_tip_to_palm
            and thumb_to_index_ratio <= self.cfg.closed_palm_thumb_index_max_ratio
        )
        return (
            passed,
            chain_count,
            adjacent_finger_angles,
            max(adjacent_tip_x_ratios),
            avg_tip_x_ratio,
            cluster_ratio,
            tip_span_ratio,
        )

    # -------- gestures --------
    def detect_open_palm(self, landmarks: Any) -> bool:
        lm = self._lm(landmarks)
        summary = summarize_hand_pose(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)
        palm_width = self._palm_width(lm)
        adjacent_tip_x_ratios = self._adjacent_tip_x_ratios(lm, palm_width)
        avg_tip_x_ratio = sum(adjacent_tip_x_ratios) / len(adjacent_tip_x_ratios)
        cluster_ratio = self._tip_cluster_ratio(lm)
        tip_span_ratio = self._tip_span_ratio(lm, palm_width)
        thumb_to_index_ratio = self._d(lm[THUMB_TIP], lm[INDEX_MCP]) / palm_width

        if summary.extended_count < self.cfg.open_palm_min_extended:
            return False
        if summary.curled_count > self.cfg.open_palm_max_curled:
            return False
        if summary.near_palm_count > self.cfg.open_palm_max_near_palm:
            return False
        if not self._non_thumb_fingers_extended(lm):
            return False
        if max(adjacent_tip_x_ratios) < self.cfg.open_palm_min_adjacent_tip_x_ratio:
            return False
        if avg_tip_x_ratio < self.cfg.open_palm_min_avg_tip_x_ratio:
            return False
        if cluster_ratio < self.cfg.open_palm_min_cluster_ratio:
            return False
        if tip_span_ratio < self.cfg.open_palm_min_tip_span_ratio:
            return False
        if thumb_to_index_ratio < self.cfg.open_palm_thumb_index_min_ratio:
            return False

        if self.cfg.open_palm_require_thumb:
            return thumb_is_extended(summary.thumb, min_tip_to_palm=self.cfg.thumb_ext_thresh)

        # Relaxed thumb rule for natural open hands: thumb may be splayed sideways
        # instead of fully extended, but it should not be folded tightly into the palm.
        return summary.thumb.tip_to_palm >= self.cfg.open_palm_thumb_min_tip_to_palm

    def detect_closed_palm(self, landmarks: Any) -> bool:
        passed, _, _, _, _, _, _ = self._closed_palm_metrics(landmarks)
        return passed

    def detect_peace_sign(self, landmarks: Any) -> bool:
        """
        PEACE_SIGN:
        - index and middle are extended upward
        - ring and pinky are folded inward
        - thumb stays compact near the folded fingers
        - index and middle visibly diverge to form a V

        This gesture is accepted only when both the finger-pair angle and the
        fingertip gap exceed their spread thresholds.
        """
        metrics = self._two_finger_pose_metrics(landmarks)
        if metrics is None:
            return False
        lm = self._lm(landmarks)
        palm_width = self._palm_width(lm)
        thumb_ring_gap_ratio = self._d(lm[THUMB_TIP], lm[RING_TIP]) / palm_width
        thumb_tip_to_palm = self._d(lm[THUMB_TIP], self._palm_center(lm)) / self._scale(lm)
        return (
            metrics["angle_deg"] >= self.cfg.peace_min_angle_deg
            and metrics["tip_gap_ratio"] >= self.cfg.peace_min_tip_gap_ratio
            and thumb_ring_gap_ratio >= self.cfg.peace_thumb_ring_min_gap_ratio
            and thumb_tip_to_palm <= self.cfg.peace_thumb_max_tip_to_palm
        )

    def detect_shaka(self, landmarks: Any) -> bool:
        """
        SHAKA:
        - thumb and pinky are clearly extended
        - index/middle/ring stay curled
        - no thumb-finger pinch contact

        The rule intentionally avoids palm-vs-back orientation assumptions so the
        pose can be recognized from either side of the hand.
        """
        lm = self._lm(landmarks)
        summary = summarize_hand_pose(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)
        palm_width = self._palm_width(lm)
        thumb_pinky_gap_ratio = self._d(lm[THUMB_TIP], lm[PINKY_TIP]) / palm_width

        index_curled = (
            self.finger_curled(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
            or self._finger_folded_toward_palm(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        )
        middle_curled = (
            self.finger_curled(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
            or self._finger_folded_toward_palm(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        )
        ring_curled = (
            self.finger_curled(lm, RING_TIP, RING_PIP, RING_MCP)
            or self._finger_folded_toward_palm(lm, RING_TIP, RING_PIP, RING_MCP)
        )
        pinky_extended = self._finger_raised(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP)

        return (
            self.thumb_extended(lm)
            and pinky_extended
            and index_curled
            and middle_curled
            and ring_curled
            and summary.min_thumb_tip_distance >= self.cfg.shaka_min_thumb_tip_distance
            and thumb_pinky_gap_ratio >= self.cfg.shaka_min_thumb_pinky_gap_ratio
        )

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
        if self.detect_peace_sign(landmarks) or self.detect_shaka(landmarks):
            return None

        ext_i = self._finger_raised(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        ext_m = self._finger_raised(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        ext_r = self._finger_raised(lm, RING_TIP, RING_PIP, RING_MCP)
        ext_p = self._finger_raised(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP)

        cur_i = self._finger_folded_toward_palm(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP) or self.finger_curled(lm, INDEX_TIP, INDEX_PIP, INDEX_MCP)
        cur_m = self._finger_folded_toward_palm(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP) or self.finger_curled(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        cur_r = self._finger_folded_toward_palm(lm, RING_TIP, RING_PIP, RING_MCP) or self.finger_curled(lm, RING_TIP, RING_PIP, RING_MCP)
        cur_p = self._finger_folded_toward_palm(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP) or self.finger_curled(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP)

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
        mid_curled = self.finger_curled(lm, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP)
        ring_curled = self.finger_curled(lm, RING_TIP, RING_PIP, RING_MCP)
        pink_curled = self.finger_curled(lm, PINKY_TIP, PINKY_PIP, PINKY_MCP)
        th_ext = self.thumb_extended(lm)

        if not (th_ext and idx_ext and mid_curled and ring_curled and pink_curled):
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
        # Canonical thumbs-up / BRAVO detection now lives in app.gestures.bravo.
        from app.gestures.bravo import detect_bravo as _detect_bravo

        return _detect_bravo(landmarks)


_DEFAULT = HandGestures()

def detect_open_palm(landmarks: Any) -> bool:
    return _DEFAULT.detect_open_palm(landmarks)

def detect_closed_palm(landmarks: Any) -> bool:
    return _DEFAULT.detect_closed_palm(landmarks)

def detect_peace_sign(landmarks: Any) -> bool:
    return _DEFAULT.detect_peace_sign(landmarks)

def detect_shaka(landmarks: Any) -> bool:
    return _DEFAULT.detect_shaka(landmarks)

def detect_number(landmarks: Any) -> Optional[str]:
    return _DEFAULT.detect_numbers_1_to_5(landmarks)

def detect_L(landmarks: Any) -> bool:
    return _DEFAULT.detect_L(landmarks)

def detect_bravo(landmarks: Any) -> bool:
    return _DEFAULT.detect_thumbs_up(landmarks)
