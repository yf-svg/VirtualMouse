from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.security.auth import GestureAuthCfg

if TYPE_CHECKING:
    from app.gestures.suite import GestureSuiteOut


@dataclass(frozen=True)
class AuthRuntimeOut:
    detected_gesture: str | None
    event_label: str | None


@dataclass(frozen=True)
class AuthRuntimeCfg:
    expected_confirm_frames: int = 2
    control_confirm_frames: int = 2
    wrong_input_confirm_frames: int = 3
    release_zone_frames: int = 2
    missing_hand_release_frames: int = 3


class AuthGestureInterpreter:
    """
    Auth-local adapter that keeps auth input conservative without changing the
    global runtime gesture pipeline.

    Responsibilities:
      - expose an auth-only detected label for the overlay
      - debounce auth events before they reach GestureAuth.update(...)
      - keep transient auth-pose transitions from becoming immediate resets
    """

    def __init__(
        self,
        *,
        auth_cfg: GestureAuthCfg | None = None,
        cfg: AuthRuntimeCfg | None = None,
    ):
        self.auth_cfg = auth_cfg or GestureAuthCfg()
        self.cfg = cfg or AuthRuntimeCfg()
        self._sequence_labels = frozenset(self.auth_cfg.sequence)
        self._reset_labels = frozenset(self.auth_cfg.reset_gestures)
        self._approve_labels = frozenset(self.auth_cfg.approve_gestures)
        self._back_labels = frozenset(self.auth_cfg.back_gestures)
        self._auth_labels = self._sequence_labels | self._reset_labels | self._approve_labels | self._back_labels
        self.reset()

    def reset(self) -> None:
        self._eligible_label: str | None = None
        self._eligible_frames = 0
        self._emitted_label: str | None = None
        self._digit_release_ready = True
        self._missing_hand_frames = 0
        self._release_zone_frames = 0

    def update(
        self,
        *,
        suite_out: Any | None,
        expected_next: str | None,
        hand_present: bool = True,
    ) -> AuthRuntimeOut:
        detected = self._detected_auth_label(suite_out)
        eligible = suite_out.eligible if suite_out is not None and suite_out.eligible in self._auth_labels else None
        eligible = self._filter_auth_event_label(eligible=eligible, expected_next=expected_next)

        if not hand_present:
            self._eligible_label = None
            self._eligible_frames = 0
            self._missing_hand_frames += 1
            self._advance_release_zone(self.cfg.missing_hand_release_frames)
            return AuthRuntimeOut(detected_gesture=None, event_label=None)

        self._missing_hand_frames = 0
        if eligible is None:
            self._eligible_label = None
            self._eligible_frames = 0
            self._advance_release_zone(self.cfg.release_zone_frames)
            return AuthRuntimeOut(detected_gesture=detected, event_label=None)

        self._release_zone_frames = 0
        if eligible not in self._sequence_labels:
            self._digit_release_ready = True

        if eligible == self._eligible_label:
            self._eligible_frames += 1
        else:
            self._eligible_label = eligible
            self._eligible_frames = 1
            self._emitted_label = None

        if eligible == self._emitted_label:
            return AuthRuntimeOut(detected_gesture=detected, event_label=None)

        if (
            expected_next in self._sequence_labels
            and eligible == expected_next
            and not self._digit_release_ready
        ):
            return AuthRuntimeOut(detected_gesture=detected, event_label=None)

        required_frames = self._required_frames(label=eligible, expected_next=expected_next)
        if self._eligible_frames < required_frames:
            return AuthRuntimeOut(detected_gesture=detected, event_label=None)

        self._emitted_label = eligible
        if eligible in self._sequence_labels:
            self._digit_release_ready = False
        return AuthRuntimeOut(detected_gesture=detected, event_label=eligible)

    def _filter_auth_event_label(self, *, eligible: str | None, expected_next: str | None) -> str | None:
        if eligible is None:
            return None
        if expected_next in self._approve_labels and eligible not in (self._approve_labels | self._back_labels | self._reset_labels):
            return None
        if eligible in self._approve_labels and expected_next not in self._approve_labels:
            return None
        return eligible

    def _required_frames(self, *, label: str, expected_next: str | None) -> int:
        if expected_next is not None and label == expected_next:
            return self.cfg.expected_confirm_frames
        if label in (self._reset_labels | self._back_labels):
            return self.cfg.control_confirm_frames
        return self.cfg.wrong_input_confirm_frames

    def _detected_auth_label(self, suite_out: Any | None) -> str | None:
        if suite_out is None:
            return None
        for label in (suite_out.eligible, suite_out.stable, suite_out.chosen):
            if label in self._auth_labels:
                return label
        return None

    def _advance_release_zone(self, required_frames: int) -> None:
        self._release_zone_frames += 1
        if self._release_zone_frames >= required_frames:
            self._emitted_label = None
            self._digit_release_ready = True
