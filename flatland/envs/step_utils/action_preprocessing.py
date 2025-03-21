from functools import lru_cache
from typing import Any, Tuple

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.transition_utils import check_action_on_agent
from flatland.utils.decorators import enable_infrastructure_lru_cache


@lru_cache()
def process_illegal_action(action: Any) -> RailEnvActions:
    """
    Returns the action if valid (either int value or in RailEnvActions), returns RailEnvActions.DO_NOTHING otherwise.
    """
    if not RailEnvActions.is_action_valid(action):
        return RailEnvActions.DO_NOTHING
    else:
        return RailEnvActions(action)


@enable_infrastructure_lru_cache()
def preprocess_left_right_action(action: RailEnvActions, rail: GridTransitionMap, position: Tuple[int, int], direction: Grid4TransitionsEnum):
    """
    LEFT/RIGHT is converted to FORWARD if left/right is not available.
    """
    if action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT] and not check_valid_action(action, rail, position, direction):
        action = RailEnvActions.MOVE_FORWARD
    return action


def check_valid_action(action: RailEnvActions, rail: GridTransitionMap, position: Tuple[int, int], direction: Grid4TransitionsEnum):
    """
    Checks whether action at position and direction leads to a valid new position in the grid.

    Fails if the grid is not valid or if MOVE_FORWARD in a symmetric switch or MOVE_LEFT in straight element.

    Parameters
    ----------
    action : RailEnvActions
    rail : GridTransitionMap
    position : Tuple[int, int]
    direction: Grid4TransitionsEnum

    Returns
    -------
    bool
    """
    new_cell_valid, _, _, transition_valid = check_action_on_agent(action, rail, position, direction)
    return new_cell_valid and transition_valid
