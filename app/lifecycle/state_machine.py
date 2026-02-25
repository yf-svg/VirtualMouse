from app.constants import AppState


class StateMachine:
    def __init__(self):
        self.state = AppState.IDLE_LOCKED

    def set_state(self, new_state: AppState) -> None:
        self.state = new_state