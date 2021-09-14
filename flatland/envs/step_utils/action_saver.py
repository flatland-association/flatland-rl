from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

class ActionSaver:
    def __init__(self):
        self.saved_action = None

    @property
    def is_action_saved(self):
        return self.saved_action is not None
    
    def __repr__(self):
        return f"is_action_saved: {self.is_action_saved}, saved_action: {str(self.saved_action)}"


    def save_action_if_allowed(self, action, state):
        """
        Save the action if all conditions are met
            1. It is a movement based action -> Forward, Left, Right
            2. Action is not already saved 
            3. Agent is not already done
        """
        if action.is_moving_action() and not self.is_action_saved and not state == TrainState.DONE:
            self.saved_action = action

    def clear_saved_action(self):
        self.saved_action = None

    def to_dict(self):
        return {"saved_action": self.saved_action}
    
    def from_dict(self, load_dict):
        self.saved_action = load_dict['saved_action']
    
    def __eq__(self, other):
        return self.saved_action == other.saved_action


