from typing import Tuple

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero
from flatland.envs.rail_env_action import RailEnvActions
from flatland.utils.decorators import enable_infrastructure_lru_cache


@enable_infrastructure_lru_cache(maxsize=1_000_000)
def check_action_on_agent(
    action: RailEnvActions, rail: GridTransitionMap, position: Tuple[int, int], direction: Grid4TransitionsEnum) -> Tuple[bool, int, Tuple[int, int], bool]:
    """
    Gets new position and direction for the action.

    Parameters
    ----------
    action : RailEnvActions
    rail : GridTransitionMap
    position: Tuple[int,int]
    direction : Grid4TransitionsEnum

    Returns
    -------
    new_cell_valid : bool
        whether the new position and new direction are valid in the grid
    new_direction : int
    new_position : Tuple[int,int]
    transition_valid : bool
        whether the transition from old to new position and direction is defined in the grid

    """
    possible_transitions = rail.get_transitions(*position, direction)
    num_transitions = fast_count_nonzero(possible_transitions)

    new_direction = direction

    # simple and symmetric switches, slips, ...
    if action == RailEnvActions.MOVE_LEFT:
        new_direction = direction - 1

    elif action == RailEnvActions.MOVE_RIGHT:
        new_direction = direction + 1

    new_direction %= 4

    # dead-end, straight line or curved line: take only available transition
    if (action == RailEnvActions.MOVE_FORWARD or action == RailEnvActions.DO_NOTHING) and num_transitions == 1:
        new_direction = fast_argmax(possible_transitions)

    new_position = get_new_position(position, new_direction)

    new_cell_valid = check_bounds(new_position, rail.height, rail.width) and rail.get_full_transitions(*new_position) > 0
    transition_valid = rail.get_transition((*position, direction), new_direction)

    return new_cell_valid, new_direction, new_position, transition_valid


def check_bounds(position, height, width):
    return position[0] >= 0 and position[1] >= 0 and position[0] < height and position[1] < width


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
