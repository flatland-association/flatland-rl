from dataclasses import dataclass
from typing import Tuple

from flatland.core.grid.grid4 import Grid4Transitions, Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import IntVector2D
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import StateTransitionSignals
from flatland.utils.decorators import enable_infrastructure_lru_cache


@dataclass(repr=True)
class AgentTransitionData:
    """ Class for keeping track of temporary agent data for position update """
    speed: float
    new_position: Tuple[int, int]
    new_direction: Grid4Transitions
    new_speed: float
    new_position_level_free: float
    preprocessed_action: RailEnvActions
    agent_position_level_free: Tuple[int, int]
    state_transition_signal: StateTransitionSignals


@enable_infrastructure_lru_cache(maxsize=1_000_000)
def apply_action_independent(action: RailEnvActions, rail: GridTransitionMap, position: IntVector2D, direction: Grid4TransitionsEnum):
    """ Apply the action on the train regardless of locations of other trains.
        Checks for valid cells to move and valid rail transitions.

        Parameters
        ----------
        action : RailEnvActions
            Action to execute
        rail : GridTransitionMap
            Flatland env.rail object
        position : IntVector2D
            current position of the train
        direction : int
            current direction of the train

        Returns
        -------
        new_position
            New position after applying the action
        new_direction
            New direction after applying the action
    """

    def check_action_new(action: RailEnvActions, position: IntVector2D, direction: int, rail, ):
        """
        Returns

        Parameters
        ----------
        agent : EnvAgent
        action : RailEnvActions

        Returns
        -------
        Tuple[Grid4TransitionsEnum, bool]
            the new direction and whether the the action was valid
        """
        possible_transitions = rail.get_transitions(*position, direction)
        num_transitions = fast_count_nonzero(possible_transitions)

        if num_transitions == 1:
            # - dead-end, straight line or curved line;
            # new_direction will be the only valid transition
            # - take only available transition
            new_direction = fast_argmax(possible_transitions)
            transition_valid = True
            return new_direction, transition_valid

        if action == RailEnvActions.MOVE_LEFT:
            new_direction = (direction - 1) % 4
            if possible_transitions[new_direction]:
                return new_direction, True

        elif action == RailEnvActions.MOVE_RIGHT:
            new_direction = (direction + 1) % 4
            if possible_transitions[new_direction]:
                return new_direction, True

        return direction, False

    new_direction, _ = check_action_new(action, position, direction, rail)
    new_position = get_new_position(position, new_direction)
    return new_position, new_direction
