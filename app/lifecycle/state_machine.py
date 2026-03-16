from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.constants import AppState


_ALLOWED_TRANSITIONS: Final[dict[AppState, frozenset[AppState]]] = {
    AppState.IDLE_LOCKED: frozenset({AppState.AUTHENTICATING, AppState.EXITING}),
    AppState.AUTHENTICATING: frozenset({
        AppState.IDLE_LOCKED,
        AppState.ACTIVE_GENERAL,
        AppState.ACTIVE_PRESENTATION,
        AppState.EXITING,
    }),
    AppState.ACTIVE_GENERAL: frozenset({
        AppState.SLEEP,
        AppState.ACTIVE_PRESENTATION,
        AppState.IDLE_LOCKED,
        AppState.EXITING,
    }),
    AppState.ACTIVE_PRESENTATION: frozenset({
        AppState.ACTIVE_GENERAL,
        AppState.SLEEP,
        AppState.IDLE_LOCKED,
        AppState.EXITING,
    }),
    AppState.SLEEP: frozenset({AppState.IDLE_LOCKED, AppState.AUTHENTICATING, AppState.EXITING}),
    AppState.EXITING: frozenset(),
}


@dataclass
class StateMachine:
    _state: AppState = AppState.IDLE_LOCKED

    @property
    def state(self) -> AppState:
        return self._state

    def can_transition(self, new_state: AppState) -> bool:
        if new_state == self._state:
            return True
        return new_state in _ALLOWED_TRANSITIONS[self._state]

    def set_state(self, new_state: AppState) -> None:
        if not self.can_transition(new_state):
            raise ValueError(f"Invalid state transition: {self._state.value} -> {new_state.value}")
        self._state = new_state

    def reset(self) -> None:
        self._state = AppState.IDLE_LOCKED
