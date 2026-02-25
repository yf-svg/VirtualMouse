 
# # finger_state.py

# class FingerState:
#     """
#        Robust finger state detection with handedness safety.
       
#     """

#     def __init__(self):
#         self.tips = [4, 8, 12, 16, 20]
#         self.refs = [2, 6, 10, 14, 18]

#     # --------------------------------------------------
#     def get_finger_states(self, lm):
#         fingers = []

#         # Thumb (adaptive to hand orientation)
#         if lm[self.tips[0]].x > lm[self.refs[0]].x:
#             fingers.append(1)
#         else:
#             fingers.append(0)

#         # Other fingers
#         for i in range(1, 5):
#             fingers.append(
#                 1 if lm[self.tips[i]].y < lm[self.refs[i]].y else 0
#             )

#         return fingers
 
def extract_features(landmarks):
    raise NotImplementedError("Implemented in Phase 2")