from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.config import CONFIG, PresentationRuntimeConfig
from app.modes.presentation import PRESENTATION_PLAYBACK_BINDINGS

if TYPE_CHECKING:
    from app.gestures.suite import GestureSuiteOut


@dataclass(frozen=True)
class PresentationGestureSignal:
    gesture_label: str | None
    event_label: str | None
    active_frames: int
    threshold_frames: int | None
    reason: str

    def status_text(self) -> str:
        label = self.gesture_label or "-"
        if self.threshold_frames is None:
            return f"PRS:{label}:{self.reason}"
        return f"PRS:{label}:{self.active_frames}/{self.threshold_frames}:{self.reason}"


class PresentationGestureInterpreter:
    """
    Presentation-local event gate.
    It converts held presentation gestures into one-shot playback events and
    makes session-control gestures more deliberate than slide navigation.
    """

    def __init__(self, *, cfg: PresentationRuntimeConfig | None = None):
        self.cfg = cfg or CONFIG.presentation_runtime
        self.reset()

    def reset(self) -> None:
        self._active_label: str | None = None
        self._active_frames = 0
        self._release_frames = 0
        self._emitted_for_active = False

    def update(
        self,
        *,
        suite_out: GestureSuiteOut | None,
        hand_present: bool,
    ) -> PresentationGestureSignal:
        gesture_label = None
        if hand_present and suite_out is not None and suite_out.eligible in PRESENTATION_PLAYBACK_BINDINGS:
            gesture_label = suite_out.eligible

        if gesture_label is None:
            return self._handle_release()

        if gesture_label != self._active_label:
            self._active_label = gesture_label
            self._active_frames = 1
            self._release_frames = 0
            self._emitted_for_active = False
        else:
            self._active_frames += 1
            self._release_frames = 0

        threshold = self._threshold_for(gesture_label)
        if self._emitted_for_active:
            return PresentationGestureSignal(
                gesture_label=gesture_label,
                event_label=None,
                active_frames=self._active_frames,
                threshold_frames=threshold,
                reason="held_after_emit",
            )

        if self._active_frames < threshold:
            return PresentationGestureSignal(
                gesture_label=gesture_label,
                event_label=None,
                active_frames=self._active_frames,
                threshold_frames=threshold,
                reason="pending_confirm",
            )

        self._emitted_for_active = True
        return PresentationGestureSignal(
            gesture_label=gesture_label,
            event_label=gesture_label,
            active_frames=self._active_frames,
            threshold_frames=threshold,
            reason="emitted",
        )

    def _handle_release(self) -> PresentationGestureSignal:
        if self._active_label is None:
            return PresentationGestureSignal(
                gesture_label=None,
                event_label=None,
                active_frames=0,
                threshold_frames=None,
                reason="idle",
            )

        self._release_frames += 1
        if self._release_frames < max(1, int(self.cfg.release_grace_frames)):
            return PresentationGestureSignal(
                gesture_label=None,
                event_label=None,
                active_frames=self._active_frames,
                threshold_frames=self._threshold_for(self._active_label),
                reason="release_grace",
            )

        self.reset()
        return PresentationGestureSignal(
            gesture_label=None,
            event_label=None,
            active_frames=0,
            threshold_frames=None,
            reason="released",
        )

    def _threshold_for(self, gesture_label: str) -> int:
        binding = PRESENTATION_PLAYBACK_BINDINGS[gesture_label]
        if binding.capability == "navigation":
            return max(1, int(self.cfg.navigation_confirm_frames))
        return max(1, int(self.cfg.session_control_confirm_frames))
