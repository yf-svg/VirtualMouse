from app.constants import AppState
from app.lifecycle.state_machine import StateMachine
from app.lifecycle.runtime_loop import run_loop


def main() -> None:
    sm = StateMachine()
    print(f"Starting {sm.state.value} ...")
    run_loop(initial_state=sm.state)