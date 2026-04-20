from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from app.config import CONFIG, PresentationToolConfig

if TYPE_CHECKING:
    from app.gestures.suite import GestureSuiteOut


def _debug_draw(message: str) -> None:
    print(f"[DRAW] {message}", flush=True)


@dataclass(frozen=True)
class PresentationEraseEvent:
    action_name: str | None
    reason: str
    fist_pressed: bool
    duration_ms: int | None = None


class PresentationEraseHandler:
    """
    Timed FIST handler for draw-mode erase actions.

    Quick FIST release undoes one stroke. A sustained hold clears all
    annotations once, then waits for release without re-triggering.
    """

    def __init__(
        self,
        *,
        cfg: PresentationToolConfig | None = None,
        clock: Callable[[], float] | None = None,
    ):
        self.cfg = cfg or CONFIG.presentation_tools
        self._clock = clock or time.monotonic
        self.reset()

    def reset(self) -> None:
        self._pressed = False
        self._pressed_at = 0.0
        self._clear_sent = False
        self._visible_frames = 0

    def update(
        self,
        *,
        suite_out: GestureSuiteOut | None,
        hand_present: bool,
        enabled: bool,
    ) -> PresentationEraseEvent:
        if not enabled:
            self.reset()
            return PresentationEraseEvent(None, "disabled", False, None)

        now = self._clock()
        fist_visible = bool(hand_present and suite_out is not None and self._is_fist_visible(suite_out))
        fist_up = bool((suite_out is not None and suite_out.up == "FIST") or (self._pressed and not fist_visible))

        if fist_visible and self._pressed:
            self._visible_frames += 1

        if fist_visible and not self._pressed:
            self._pressed = True
            self._pressed_at = now
            self._clear_sent = False
            self._visible_frames = 1
            _debug_draw("FIST pressed")
            return PresentationEraseEvent(None, "fist_pressed", True, 0)

        if self._pressed and not self._clear_sent:
            duration_ms = self._duration_ms(now)
            if (
                fist_visible
                and duration_ms >= max(1, int(self.cfg.draw_clear_hold_ms))
                and self._visible_frames >= max(1, int(getattr(self.cfg, "draw_clear_confirm_frames", 1)))
            ):
                self._clear_sent = True
                _debug_draw("Action: CLEAR_ALL")
                return PresentationEraseEvent(
                    "PRESENTATION_DRAW_CLEAR",
                    "clear_all",
                    True,
                    duration_ms,
                )

        if self._pressed and fist_up:
            duration_ms = self._duration_ms(now)
            _debug_draw(f"FIST released (duration: {duration_ms} ms)")
            if (
                not self._clear_sent
                and duration_ms <= max(1, int(self.cfg.draw_undo_max_ms))
                and (
                    duration_ms >= max(0, int(self.cfg.draw_undo_min_ms))
                    or self._visible_frames >= max(1, int(self.cfg.draw_undo_min_detected_frames))
                )
            ):
                self.reset()
                _debug_draw("Action: UNDO_ONE")
                return PresentationEraseEvent(
                    "PRESENTATION_DRAW_UNDO",
                    "undo_one",
                    False,
                    duration_ms,
                )
            self.reset()
            return PresentationEraseEvent(None, "fist_released", False, duration_ms)

        if self._pressed:
            return PresentationEraseEvent(None, "fist_held", True, self._duration_ms(now))

        return PresentationEraseEvent(None, "idle", False, None)

    @staticmethod
    def _is_fist_visible(suite_out: GestureSuiteOut) -> bool:
        raw_candidates = getattr(suite_out, "raw_candidates", None)
        if raw_candidates is not None:
            return "FIST" in raw_candidates
        candidates = getattr(suite_out, "candidates", None)
        if candidates is not None:
            return "FIST" in candidates
        return any(
            label == "FIST"
            for label in (
                suite_out.chosen,
                suite_out.down,
            )
        )

    def _duration_ms(self, now: float) -> int:
        return int(round(max(0.0, now - self._pressed_at) * 1000.0))
