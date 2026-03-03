from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, FrozenSet, Dict, List, Tuple

from app.gestures.features import (
    THUMB_TIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_DIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_DIP,
    MIDDLE_TIP,
    RING_MCP,
    RING_PIP,
    RING_DIP,
    RING_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_DIP,
    PINKY_TIP,
    distance,
    finger_is_curled,
    finger_is_extended,
    finger_metrics,
    get_landmarks,
    hand_scale,
    summarize_hand_pose,
    thumb_metrics,
)


@dataclass
class PinchCfg:
    # --- Singles ---
    index_on: float = 0.46
    index_off: float = 0.58

    middle_on: float = 0.46
    middle_off: float = 0.58

    ring_on: float = 0.47
    ring_off: float = 0.60

    pinky_on: float = 0.48
    pinky_off: float = 0.62

    # --- Chords release hysteresis ---
    chord_off: float = 0.64

    # ✅ IMRP must be "strong" on non-pinky fingers too (prevents pinky->IMRP confusion)
    imrp_nonpinky_max: float = 0.44  # I/M/R must all be below this to allow IMRP
    imrp_pinky_max: float = 0.46     # pinky must be below this to allow IMRP

    # ✅ If pinky is clearly the closest pinch by this margin, prefer PINCH_PINKY
    pinky_priority_margin: float = 0.06

    # Exclusion / separation
    thumb_opposition_min: float = 0.42
    target_tip_to_palm_min: float = 0.40
    nearest_margin: float = 0.08
    very_strong_on: float = 0.30
    dip_weight: float = 0.35
    palm_weight: float = 0.18
    support_extended_tip_to_palm_min: float = 0.70
    support_extended_curl_ratio_min: float = 1.03
    support_extended_min_count_single: int = 2
    support_extended_min_count_imr: int = 1
    im_ring_clear_off: float = 0.56
    im_ring_extended_tip_to_palm_min: float = 0.72
    im_ring_extended_curl_ratio_min: float = 1.04
    pinky_extended_tip_to_palm_min: float = 0.72
    pinky_extended_curl_ratio_min: float = 1.04
    pinky_imrp_max_tip_to_palm: float = 0.68
    global_fist_curled: int = 3
    global_fist_near_palm: int = 4
    global_open_extended: int = 4
    global_open_min_thumb_dist: float = 0.34

    # Temporal stability
    ema_alpha: float = 0.70
    hold_on: int = 1
    hold_off: int = 2

    use_3d: bool = True
    min_scale: float = 1e-4


class PinchDetector:
    """
    Detects:
      Singles: PINCH_INDEX / PINCH_MIDDLE / PINCH_RING / PINCH_PINKY
      Chords : PINCH_IM / PINCH_IMR / PINCH_IMRP

    Key improvements:
      - IMRP has a strict "strong pinch" gate
      - pinky gets priority when it is clearly the closest pinch
    """
    def __init__(self, cfg: PinchCfg = PinchCfg()):
        self.cfg = cfg
        self.reset()

        self._chords: List[Tuple[FrozenSet[str], str]] = [
            (frozenset({"I", "M", "R", "P"}), "PINCH_IMRP"),
            (frozenset({"I", "M", "R"}), "PINCH_IMR"),
            (frozenset({"I", "M"}), "PINCH_IM"),
        ]

    def reset(self) -> None:
        self._ema: Dict[str, Optional[float]] = {"I": None, "M": None, "R": None, "P": None}
        self._state: Optional[str] = None
        self._on: int = 0
        self._off: int = 0

    def _ema_update(self, k: str, raw: float) -> float:
        prev = self._ema[k]
        if prev is None:
            self._ema[k] = raw
        else:
            a = self.cfg.ema_alpha
            self._ema[k] = a * raw + (1.0 - a) * prev
        return self._ema[k]  # type: ignore

    def update(self, landmarks: Any) -> Optional[str]:
        lm = get_landmarks(landmarks)
        summary = summarize_hand_pose(lm, use_3d=self.cfg.use_3d, min_scale=self.cfg.min_scale)
        scale = summary.scale

        thumb = lm[THUMB_TIP]

        di = self._ema_update("I", distance(thumb, lm[INDEX_TIP], use_3d=self.cfg.use_3d) / scale)
        dm = self._ema_update("M", distance(thumb, lm[MIDDLE_TIP], use_3d=self.cfg.use_3d) / scale)
        dr = self._ema_update("R", distance(thumb, lm[RING_TIP], use_3d=self.cfg.use_3d) / scale)
        dp = self._ema_update("P", distance(thumb, lm[PINKY_TIP], use_3d=self.cfg.use_3d) / scale)

        ddi = distance(thumb, lm[INDEX_DIP], use_3d=self.cfg.use_3d) / scale
        ddm = distance(thumb, lm[MIDDLE_DIP], use_3d=self.cfg.use_3d) / scale
        ddr = distance(thumb, lm[RING_DIP], use_3d=self.cfg.use_3d) / scale
        ddp = distance(thumb, lm[PINKY_DIP], use_3d=self.cfg.use_3d) / scale

        dmap = {"I": di, "M": dm, "R": dr, "P": dp}
        score_map = {
            "I": di + self.cfg.dip_weight * ddi,
            "M": dm + self.cfg.dip_weight * ddm,
            "R": dr + self.cfg.dip_weight * ddr,
            "P": dp + self.cfg.dip_weight * ddp,
        }
        ordered = sorted(score_map.items(), key=lambda kv: kv[1])
        best_key, best_score = ordered[0]
        second_score = ordered[1][1]

        thumb_state = summary.thumb
        finger_states = summary.fingers
        curled_count = summary.curled_count
        support_extended = {
            key: finger_is_extended(
                finger_states[key],
                min_tip_to_palm=self.cfg.support_extended_tip_to_palm_min,
                min_curl_ratio=self.cfg.support_extended_curl_ratio_min,
            )
            for key in ("I", "M", "R", "P")
        }
        pinky_extended = finger_is_extended(
            finger_states["P"],
            min_tip_to_palm=self.cfg.pinky_extended_tip_to_palm_min,
            min_curl_ratio=self.cfg.pinky_extended_curl_ratio_min,
        )
        pinky_not_extended = finger_states["P"].tip_to_palm <= self.cfg.pinky_imrp_max_tip_to_palm or not pinky_extended
        ring_clearly_not_participating = (
            dr >= self.cfg.im_ring_clear_off
            and finger_is_extended(
                finger_states["R"],
                min_tip_to_palm=self.cfg.im_ring_extended_tip_to_palm_min,
                min_curl_ratio=self.cfg.im_ring_extended_curl_ratio_min,
            )
        )

        # Pinch should not win on clearly global open hands or fully closed hands.
        if summary.extended_count >= self.cfg.global_open_extended and summary.min_thumb_tip_distance > self.cfg.global_open_min_thumb_dist:
            cand = None
        elif summary.extended_count == 0 and curled_count >= self.cfg.global_fist_curled and summary.near_palm_count >= self.cfg.global_fist_near_palm:
            cand = None
        else:
            cand = None

            # Block fist-like closures unless the thumb/finger contact is very strong and isolated.
            fist_like = (
                thumb_state.tip_to_palm < self.cfg.thumb_opposition_min
                and curled_count >= self.cfg.global_fist_curled
                and summary.near_palm_count >= self.cfg.global_fist_near_palm
            )
            if fist_like and not (dmap[best_key] < self.cfg.very_strong_on and (second_score - best_score) >= self.cfg.nearest_margin):
                cand = None

            def eligible(key: str, dist_value: float, on_threshold: float) -> bool:
                return (
                    dist_value < on_threshold
                    and (
                        finger_states[key].tip_to_palm >= self.cfg.target_tip_to_palm_min
                        or dist_value < self.cfg.very_strong_on
                    )
                )

            # --- Singles ON flags (strict) ---
            i_on = eligible("I", di, self.cfg.index_on)
            m_on = eligible("M", dm, self.cfg.middle_on)
            r_on = eligible("R", dr, self.cfg.ring_on)
            p_on = eligible("P", dp, self.cfg.pinky_on)

            single_support = {
                "I": sum(1 for k in ("M", "R", "P") if support_extended[k]),
                "M": sum(1 for k in ("I", "R", "P") if support_extended[k]),
                "R": sum(1 for k in ("I", "M", "P") if support_extended[k]),
                "P": sum(1 for k in ("I", "M", "R") if support_extended[k]),
            }

            # ✅ Pinky priority rule:
            # If pinky is strongly the closest, prefer PINCH_PINKY and do NOT upgrade to IMRP.
            others_min = min(di, dm, dr)
            if (
                p_on
                and single_support["P"] >= self.cfg.support_extended_min_count_single
                and (dp + self.cfg.pinky_priority_margin) < others_min
            ):
                cand = "PINCH_PINKY"
            else:
                # --- Chords first (with strict IMRP gate) ---
                # IMRP requires all four ON and pinky not extended.
                if i_on and m_on and r_on and p_on:
                    if (
                        (max(di, dm, dr) < self.cfg.imrp_nonpinky_max)
                        and (dp < self.cfg.imrp_pinky_max)
                        and pinky_not_extended
                    ):
                        cand = "PINCH_IMRP"

                # IMR requires a visibly extended pinky, not a participating pinky.
                if (
                    cand is None
                    and (i_on and m_on and r_on)
                    and pinky_extended
                    and not p_on
                    and single_support["P"] >= self.cfg.support_extended_min_count_imr
                ):
                    cand = "PINCH_IMR"
                if cand is None and (i_on and m_on) and ring_clearly_not_participating:
                    cand = "PINCH_IM"

                # --- Singles fallback (closest wins only if clearly isolated) ---
                if cand is None:
                    active = []
                    if i_on and single_support["I"] >= self.cfg.support_extended_min_count_single:
                        active.append(("PINCH_INDEX", score_map["I"] + self.cfg.palm_weight * finger_states["I"].tip_to_palm))
                    if m_on and single_support["M"] >= self.cfg.support_extended_min_count_single:
                        active.append(("PINCH_MIDDLE", score_map["M"] + self.cfg.palm_weight * finger_states["M"].tip_to_palm))
                    if r_on and single_support["R"] >= self.cfg.support_extended_min_count_single:
                        active.append(("PINCH_RING", score_map["R"] + self.cfg.palm_weight * finger_states["R"].tip_to_palm))
                    if p_on and single_support["P"] >= self.cfg.support_extended_min_count_single:
                        active.append(("PINCH_PINKY", score_map["P"] + self.cfg.palm_weight * finger_states["P"].tip_to_palm))

                    if active:
                        active.sort(key=lambda x: x[1])
                        if len(active) == 1 or (active[1][1] - active[0][1]) >= self.cfg.nearest_margin:
                            cand = active[0][0]

        # --- State machine (hysteresis + hold) ---
        if self._state is None:
            if cand is not None:
                self._on += 1
                if self._on >= self.cfg.hold_on:
                    self._state = cand
                    self._off = 0
            else:
                self._on = 0
        else:
            still = False

            if self._state.startswith("PINCH_IM"):
                chord_req = {
                    "PINCH_IM": {"I", "M"},
                    "PINCH_IMR": {"I", "M", "R"},
                    "PINCH_IMRP": {"I", "M", "R", "P"},
                }.get(self._state)

                if chord_req is not None:
                    still = all(dmap[k] < self.cfg.chord_off for k in chord_req)
            else:
                still = {
                    "PINCH_INDEX": di < self.cfg.index_off,
                    "PINCH_MIDDLE": dm < self.cfg.middle_off,
                    "PINCH_RING": dr < self.cfg.ring_off,
                    "PINCH_PINKY": dp < self.cfg.pinky_off,
                }.get(self._state, False)

            if still:
                self._off = 0
            else:
                self._off += 1
                if self._off >= self.cfg.hold_off:
                    self._state = None
                    self._on = 0

        return self._state


_DEFAULT_DETECTOR = PinchDetector()


def detect_pinch_type(landmarks: Any) -> Optional[str]:
    return _DEFAULT_DETECTOR.update(landmarks)


def reset_pinch() -> None:
    _DEFAULT_DETECTOR.reset()
