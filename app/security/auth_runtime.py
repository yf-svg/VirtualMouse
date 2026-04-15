from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.gestures.sets.auth_set import auth_alias_map_for_cfg
from app.security.auth import AuthInputState, GestureAuthCfg

if TYPE_CHECKING:
    from app.gestures.suite import GestureSuiteOut


@dataclass(frozen=True)
class AuthRuntimeOut:
    detected_gesture: str | None
    event_label: str | None


@dataclass(frozen=True)
class AuthRuntimeCfg:
    digit_confirm_frames: int = 2
    control_confirm_frames: int = 2
    release_zone_frames: int = 2
    missing_hand_release_frames: int = 3


class AuthGestureInterpreter:
    """
    Auth-local adapter that turns the live auth gesture stream into discrete,
    keypad-style auth events without changing the global runtime gesture path.

    Responsibilities:
      - expose an auth-only detected label for the overlay
      - debounce auth events before they reach GestureAuth.update(...)
      - require a release zone between committed digit presses
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
        self._sequence_aliases = auth_alias_map_for_cfg(self.auth_cfg)
        self._auth_labels = (
            self._sequence_labels
            | self._reset_labels
            | self._approve_labels
            | self._back_labels
            | frozenset(self._sequence_aliases)
        )
        self.reset()

    def reset(self) -> None:
        self._eligible_label: str | None = None
        self._eligible_frames = 0
        self._emitted_label: str | None = None
        self._digit_release_ready = True
        self._release_zone_frames = 0

    def update(
        self,
        *,
        suite_out: Any | None,
        auth_state: AuthInputState,
        hand_present: bool = True,
    ) -> AuthRuntimeOut:
        detected = self._detected_auth_label(suite_out)
        eligible = self._normalize_auth_label(suite_out.eligible if suite_out is not None else None)
        eligible = self._filter_auth_event_label(eligible=eligible, auth_state=auth_state)

        if not hand_present:
            self._eligible_label = None
            self._eligible_frames = 0
            self._advance_release_zone(self.cfg.missing_hand_release_frames)
            return AuthRuntimeOut(detected_gesture=None, event_label=None)

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

        if eligible in self._sequence_labels and not self._digit_release_ready:
            return AuthRuntimeOut(detected_gesture=detected, event_label=None)

        required_frames = self._required_frames(label=eligible)
        if self._eligible_frames < required_frames:
            return AuthRuntimeOut(detected_gesture=detected, event_label=None)

        self._emitted_label = eligible
        if eligible in self._sequence_labels:
            self._digit_release_ready = False
        return AuthRuntimeOut(detected_gesture=detected, event_label=eligible)

    def _filter_auth_event_label(self, *, eligible: str | None, auth_state: AuthInputState) -> str | None:
        if eligible is None:
            return None
        if auth_state.locked_out:
            return None
        if eligible in self._sequence_labels and not auth_state.accepting_digits:
            return None
        if eligible in self._approve_labels and not auth_state.ready_to_submit:
            return None
        return eligible

    def _required_frames(self, *, label: str) -> int:
        if label in self._sequence_labels:
            return self.cfg.digit_confirm_frames
        if label in (self._reset_labels | self._back_labels):
            return self.cfg.control_confirm_frames
        return self.cfg.control_confirm_frames

    def _detected_auth_label(self, suite_out: Any | None) -> str | None:
        if suite_out is None:
            return None
        for label in (suite_out.eligible, suite_out.stable, suite_out.chosen):
            normalized = self._normalize_auth_label(label)
            if normalized is not None:
                return normalized
        return None

    def _normalize_auth_label(self, label: str | None) -> str | None:
        if label is None:
            return None
        if label in self._sequence_aliases:
            return self._sequence_aliases[label]
        if label in self._auth_labels:
            return label
        return None

    def _advance_release_zone(self, required_frames: int) -> None:
        self._release_zone_frames += 1
        if self._release_zone_frames >= required_frames:
            self._emitted_label = None
            self._digit_release_ready = True
