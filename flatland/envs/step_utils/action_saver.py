from flatland.envs.agent_utils import TrainState
from flatland.envs.rail_env import RailEnvActions

class ActionSaver:
	def __init__(self):
		self.saved_action = None
	
	@property
	def is_action_saved(self):
		return not RailEnvActions.is_moving_action(self.saved_action)

	def save_action_if_allowed(self, action):
		if not self.is_action_saved and RailEnvActions.is_moving_action(action):
			self.saved_action = action

	def clear_saved_action(self):
		self.saved_action = None

