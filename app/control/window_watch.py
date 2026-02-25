class WindowWatch:
    """
    App-based presentation mode permissioning.
    Phase 4: detect presentation app open/active and signal mode switching.
    """
    def __init__(self):
        raise NotImplementedError("Implemented in Phase 4")

    def presentation_allowed(self) -> bool:
        raise NotImplementedError("Implemented in Phase 4")