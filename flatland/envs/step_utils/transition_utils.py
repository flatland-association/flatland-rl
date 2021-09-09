from typing import Tuple
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env_action import RailEnvActions


def check_action(action, position, direction, rail):
    """

    Parameters
    ----------
    agent : EnvAgent
    action : RailEnvActions

    Returns
    -------
    Tuple[Grid4TransitionsEnum,Tuple[int,int]]



    """
    transition_valid = None
    possible_transitions = rail.get_transitions(*position, direction)
    num_transitions = fast_count_nonzero(possible_transitions)
	
    new_direction = direction
    if action == RailEnvActions.MOVE_LEFT:
        new_direction = direction - 1
        if num_transitions <= 1:
            transition_valid = False

    elif action == RailEnvActions.MOVE_RIGHT:
        new_direction = direction + 1
        if num_transitions <= 1:
            transition_valid = False

    new_direction %= 4  # Dipam : Why?

    if action == RailEnvActions.MOVE_FORWARD and num_transitions == 1:
        # - dead-end, straight line or curved line;
        # new_direction will be the only valid transition
        # - take only available transition
        new_direction = fast_argmax(possible_transitions)
        transition_valid = True
    return new_direction, transition_valid


def check_action_on_agent(action, rail, position, direction):
    """
    Parameters
    ----------
    action : RailEnvActions
    agent : EnvAgent

    Returns
    -------
    bool
        Is it a legal move?
        1) transition allows the new_direction in the cell,
        2) the new cell is not empty (case 0),
        3) the cell is free, i.e., no agent is currently in that cell


    """
    # compute number of possible transitions in the current
    # cell used to check for invalid actions
    new_direction, transition_valid = check_action(action, position, direction, rail)
    new_position = get_new_position(position, new_direction)

    new_cell_valid = check_bounds(new_position, rail.height, rail.width) and \
                     rail.get_full_transitions(*new_position) > 0

    # If transition validity hasn't been checked yet.
    if transition_valid is None:
        transition_valid = rail.get_transition( (*position, direction), new_direction)

    return new_cell_valid, new_direction, new_position, transition_valid


def check_valid_action(action, rail, position, direction):
	new_cell_valid, _, _, transition_valid = check_action_on_agent(action, rail, position, direction)
	action_is_valid = new_cell_valid and transition_valid
	return action_is_valid

def fast_argmax(possible_transitions: Tuple[int, int, int, int]) -> bool:
    if possible_transitions[0] == 1:
        return 0
    if possible_transitions[1] == 1:
        return 1
    if possible_transitions[2] == 1:
        return 2
    return 3

def fast_count_nonzero(possible_transitions: Tuple[int, int, int, int]):
    return possible_transitions[0] + possible_transitions[1] + possible_transitions[2] + possible_transitions[3]

def check_bounds(position, height, width):
    return position[0] >= 0 and position[1] >= 0 and position[0] < height and position[1] < width
 