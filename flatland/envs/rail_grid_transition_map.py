from functools import lru_cache
from typing import Set
from typing import Tuple

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import IntVector2D
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.core.transitions import Transitions
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_action import RailEnvNextAction


class RailGridTransitionMap(GridTransitionMap):

    def __init__(self, width, height, transitions: Transitions = RailEnvTransitions(), random_seed=None, grid: np.ndarray = None):
        super().__init__(width, height, transitions, random_seed, grid)

    @lru_cache
    def get_valid_move_actions_(self, agent_direction: Grid4TransitionsEnum, agent_position: Tuple[int, int]) -> Set[RailEnvNextAction]:
        """
        Get the valid move actions (forward, left, right) for an agent.

        TODO The implementation could probably be more efficient and more elegant,
          but given the few calls this has no priority now.

        Parameters
        ----------
        agent_direction : Grid4TransitionsEnum
        agent_position: Tuple[int,int]


        Returns
        -------
        Set of `RailEnvNextAction` (tuples of (action,position,direction))
            Possible move actions (forward,left,right) and the next position/direction they lead to.
            It is not checked that the next cell is free.
        """
        valid_actions: Set[RailEnvNextAction] = []
        possible_transitions = self.get_transitions(*agent_position, agent_direction)
        num_transitions = fast_count_nonzero(possible_transitions)
        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if self.is_dead_end(agent_position):
            action = RailEnvActions.MOVE_FORWARD
            exit_direction = (agent_direction + 2) % 4
            if possible_transitions[exit_direction]:
                new_position = get_new_position(agent_position, exit_direction)
                valid_actions = [(RailEnvNextAction(action, new_position, exit_direction))]
        elif num_transitions == 1:
            action = RailEnvActions.MOVE_FORWARD
            for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[new_direction]:
                    new_position = get_new_position(agent_position, new_direction)
                    valid_actions = [(RailEnvNextAction(action, new_position, new_direction))]
        else:
            for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[new_direction]:
                    if new_direction == agent_direction:
                        action = RailEnvActions.MOVE_FORWARD
                    elif new_direction == (agent_direction + 1) % 4:
                        action = RailEnvActions.MOVE_RIGHT
                    elif new_direction == (agent_direction - 1) % 4:
                        action = RailEnvActions.MOVE_LEFT
                    else:
                        raise Exception("Illegal state")

                    new_position = get_new_position(agent_position, new_direction)
                    valid_actions.append(RailEnvNextAction(action, new_position, new_direction))
        return valid_actions

    @lru_cache(maxsize=1_000_000)
    def check_action_on_agent(self, action: RailEnvActions, position: Tuple[int, int], direction: Grid4TransitionsEnum) -> Tuple[
        bool, int, Tuple[int, int], bool]:
        """
        Gets new position and direction for the action.

        Parameters
        ----------
        action : RailEnvActions
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
        possible_transitions = self.get_transitions(*position, direction)
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

        new_cell_valid = self.check_bounds(new_position) and self.get_full_transitions(*new_position) > 0
        transition_valid = self.get_transition((*position, direction), new_direction)

        return new_cell_valid, new_direction, new_position, transition_valid

    def check_bounds(self, position):
        return position[0] >= 0 and position[1] >= 0 and position[0] < self.height and position[1] < self.width

    @lru_cache()
    def check_valid_action(self, action: RailEnvActions, position: Tuple[int, int], direction: Grid4TransitionsEnum):
        """
        Checks whether action at position and direction leads to a valid new position in the grid.

        Fails if the grid is not valid or if MOVE_FORWARD in a symmetric switch or MOVE_LEFT in straight element.

        Parameters
        ----------
        action : RailEnvActions
        position : Tuple[int, int]
        direction: Grid4TransitionsEnum

        Returns
        -------
        bool
        """
        new_cell_valid, _, _, transition_valid = self.check_action_on_agent(action, position, direction)
        return new_cell_valid and transition_valid

    def _check_action_new(self, action: RailEnvActions, position: IntVector2D, direction: int):
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
        possible_transitions = self.get_transitions(*position, direction)
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

    @lru_cache(maxsize=1_000_000)
    def apply_action_independent(self, action: RailEnvActions, position: IntVector2D, direction: Grid4TransitionsEnum):
        """ Apply the action on the train regardless of locations of other trains.
            Checks for valid cells to move and valid rail transitions.

            Parameters
            ----------
            action : RailEnvActions
                Action to execute
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
        new_direction, _ = self._check_action_new(action, position, direction)
        new_position = get_new_position(position, new_direction)
        return new_position, new_direction

    @lru_cache()
    def preprocess_left_right_action(self, action: RailEnvActions, position: Tuple[int, int], direction: Grid4TransitionsEnum):
        """
        LEFT/RIGHT is converted to FORWARD if left/right is not available.
        """
        if action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT] and not self.check_valid_action(action, position, direction):
            # TODO revise design: this may accelerate!
            action = RailEnvActions.MOVE_FORWARD
        return action
