from __future__ import annotations

import traceback

from app.constants import AppState
from app.lifecycle.runtime_loop import run_loop
from app.lifecycle.state_machine import StateMachine


def main() -> None:
    sm = StateMachine()
    exit_code = 0
    print(f"[App] Starting in state={sm.state.value}")

    try:
        run_loop(initial_state=sm.state)
    except KeyboardInterrupt:
        exit_code = 130
        print("[App] Interrupted by user")
    except Exception as exc:
        exit_code = 1
        print(f"[App] Fatal error: {exc}")
        traceback.print_exc()
    finally:
        if sm.can_transition(AppState.EXITING):
            sm.set_state(AppState.EXITING)
        print(f"[App] Shutting down in state={sm.state.value}")

    if exit_code:
        raise SystemExit(exit_code)
