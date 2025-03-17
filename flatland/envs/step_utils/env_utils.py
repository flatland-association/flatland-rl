from dataclasses import dataclass
from typing import Tuple

from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils import transition_utils
from flatland.utils.decorators import enable_infrastructure_lru_cache


@dataclass(repr=True)
class AgentTransitionData:
    """ Class for keeping track of temporary agent data for position update """
    position : Tuple[int, int]
    direction : Grid4Transitions
    preprocessed_action : RailEnvActions

@enable_infrastructure_lru_cache(maxsize=1_000_000)
def apply_action_independent(action, rail, position, direction):
    """ Apply the action on the train regardless of locations of other trains
        Checks for valid cells to move and valid rail transitions
        ---------------------------------------------------------------------
        Parameters: action - Action to execute
                    rail - Flatland env.rail object
                    position - current position of the train
                    direction - current direction of the train
        ---------------------------------------------------------------------
        Returns: new_position - New position after applying the action
                    new_direction - New direction after applying the action
    """

    new_direction, _ = transition_utils.check_action(action, position, direction, rail)
    new_position = get_new_position(position, new_direction)
    return new_position, new_direction

def state_position_sync_check(state, position, i_agent):
    """ Check for whether on map and off map states are matching with position """
    if state.is_on_map_state() and position is None:
        raise ValueError("Agent ID {} Agent State {} is on map Agent Position {} if off map ".format(
                        i_agent, str(state), str(position) ))
    elif state.is_off_map_state() and position is not None:
        raise ValueError("Agent ID {} Agent State {} is off map Agent Position {} if on map ".format(
                        i_agent, str(state), str(position) ))
