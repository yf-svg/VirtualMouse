# Simple moving average smoother for 2D points
from collections import deque

class MovingAverageSmoother:
	def __init__(self, window_size=5):
		self.window_size = window_size
		self.x_vals = deque(maxlen=window_size)
		self.y_vals = deque(maxlen=window_size)

	def smooth(self, x, y):
		self.x_vals.append(x)
		self.y_vals.append(y)
		avg_x = sum(self.x_vals) / len(self.x_vals)
		avg_y = sum(self.y_vals) / len(self.y_vals)
		return avg_x, avg_y
