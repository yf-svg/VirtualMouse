# from __future__ import annotations

# import math
# import time

# # MediaPipe indices
# THUMB_TIP = 4
# INDEX_TIP = 8
# MIDDLE_TIP = 12


# def _dist(a, b) -> float:
#     return math.hypot(a.x - b.x, a.y - b.y)


# class PinchProjector:
#     """
#     Visual-only pinch constraint projection.
#     When pinch is active, gently pulls the two tip landmarks toward each other so the
#     skeleton visually "closes" the pinch.

#     Includes hysteresis (hold) to avoid snap flicker.
#     """

#     def __init__(
#         self,
#         pull_strength: float = 0.65,     # 0..1 how strongly to pull tips together
#         max_project_dist: float = 0.08,  # only project if tips are within this distance
#         hold_ms: int = 120,              # keep pinch active for this long after it disappears
#     ):
#         self.pull_strength = float(pull_strength)
#         self.max_project_dist = float(max_project_dist)
#         self.hold_s = float(hold_ms) / 1000.0

#         self._active_type: str | None = None
#         self._last_seen_t = 0.0

#     def update(self, pinch_type: str | None) -> None:
#         now = time.perf_counter()
#         if pinch_type in ("PINCH_INDEX", "PINCH_MIDDLE"):
#             self._active_type = pinch_type
#             self._last_seen_t = now
#             return

#         # pinch not seen: hold for a short time to avoid flicker
#         if self._active_type is not None and (now - self._last_seen_t) <= self.hold_s:
#             return

#         self._active_type = None

#     def apply(self, landmarks):
#         """
#         Apply projection to a landmarks proto (returns a new proto).
#         Safe to call every frame. If no pinch active, returns copy unchanged.
#         """
#         out = type(landmarks)()
#         out.CopyFrom(landmarks)

#         if self._active_type is None:
#             return out

#         lm = out.landmark
#         thumb = lm[THUMB_TIP]
#         other_idx = INDEX_TIP if self._active_type == "PINCH_INDEX" else MIDDLE_TIP
#         other = lm[other_idx]

#         d = _dist(thumb, other)
#         if d <= 1e-6 or d > self.max_project_dist:
#             return out

#         # Midpoint
#         mx = 0.5 * (thumb.x + other.x)
#         my = 0.5 * (thumb.y + other.y)

#         # Pull both tips toward midpoint
#         k = self.pull_strength
#         thumb.x = (1.0 - k) * thumb.x + k * mx
#         thumb.y = (1.0 - k) * thumb.y + k * my

#         other.x = (1.0 - k) * other.x + k * mx
#         other.y = (1.0 - k) * other.y + k * my

#         return out
