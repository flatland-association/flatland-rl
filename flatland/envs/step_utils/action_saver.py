from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

class ActionSaver:
	def __init__(self):
		self.saved_action = None
	
	@property
	def is_action_saved(self):
		return self.saved_action is not None

	def save_action_if_allowed(self, action, state):
		if not self.is_action_saved and \
           RailEnvActions.is_moving_action(action) and \
           not TrainState.is_malfunction_state(state):
			self.saved_action = action

	def clear_saved_action(self):
		self.saved_action = None

