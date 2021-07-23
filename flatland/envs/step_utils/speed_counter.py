import numpy as np
from flatland.envs.agent_utils import TrainState

class SpeedTracker:
	def __init__(self, speed):
		self.speed = speed
		self.max_count = int(np.ceil(1/speed))

	def update_counter(self, state):
		if state == TrainState.MOVING:
			self.counter += 1
			self.counter = self.counter % self.max_count

	@property
	def is_cell_exit(self):
		return self.counter == 0
	
	@property
	def is_cell_entry(self):
		return self.counter == self.max_count - 1