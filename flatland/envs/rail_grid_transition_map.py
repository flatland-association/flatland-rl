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

    def check_bounds(self, position):
        return position[0] >= 0 and position[1] >= 0 and position[0] < self.height and position[1] < self.width

    @lru_cache(maxsize=1_000_000)
    def _check_action_new(self, action: RailEnvActions, position: IntVector2D, direction: int):
        """
        Checks whether action at position and direction leads to a valid new position in the grid.

        Sets action to MOVE_FORWARD if MOVE_LEFT/MOVE_RIGHT is provided but transition is not possible.
        Sets action to STOPPED if MOVE_FORWARD or DO_NOTING is provided but going into symmetric switch (facing the switch).

        Parameters
        ----------
        action : RailEnvActions
        position: IntVector2D
        direction: int

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
            if action == RailEnvActions.MOVE_LEFT and new_direction != (direction - 1) % 4:
                action = RailEnvActions.MOVE_FORWARD
                return new_direction, False, action

            elif action == RailEnvActions.MOVE_RIGHT and new_direction != (direction + 1) % 4:
                action = RailEnvActions.MOVE_FORWARD
                return new_direction, False, action

            # straight or dead-end
            return new_direction, True, action

        if action == RailEnvActions.MOVE_LEFT:
            new_direction = (direction - 1) % 4
            if possible_transitions[new_direction]:
                return new_direction, True, RailEnvActions.MOVE_LEFT
            elif possible_transitions[direction]:
                return direction, False, RailEnvActions.MOVE_FORWARD

        elif action == RailEnvActions.MOVE_RIGHT:
            new_direction = (direction + 1) % 4
            if possible_transitions[new_direction]:
                return new_direction, True, RailEnvActions.MOVE_RIGHT
            elif possible_transitions[direction]:
                return direction, False, RailEnvActions.MOVE_FORWARD

        elif possible_transitions[direction]:
            return direction, True, action

        return direction, False, RailEnvActions.STOP_MOVING

    @lru_cache(maxsize=1_000_000)
    def check_action_on_agent(self, action: RailEnvActions, position: IntVector2D, direction: Grid4TransitionsEnum):
        """ Apply the action on the train regardless of locations of other trains.
            Checks for valid cells to move and valid rail transitions.

            Parameters
            ----------
            action : RailEnvActions
                Action to execute
            position : IntVector2D
                current position of the train
            direction : Grid4TransitionsEnum
                current direction of the train

            Returns
            -------
            new_cell_valid: bool
                is the new position and direction valid (i.e. is it within bounds and does it have > 0 outgoing transitions)
            new_position
                New position after applying the action
            new_direction
                New direction after applying the action
            transition_valid: bool
                Whether the transition from old and direction is defined in the grid.
                In other words, can the action be applied directly? False if
                - MOVE_FORWARD/DO_NOTHING when entering symmetric switch
                - MOVE_LEFT/MOVE_RIGHT corrected to MOVE_FORWARD in switches and dead-ends
                However, transition_valid for dead-ends and turns either with the correct MOVE_RIGHT/MOVE_LEFT or MOVE_FORWARD/DO_NOTHING.
            preprocessed_action: RailEnvActions
                Corrected action if not transition_valid.

                The preprocessed action has the following semantics:
                - MOVE_LEFT/MOVE_RIGHT: turn left/right without acceleration
                - MOVE_FORWARD: move forward with acceleration (swap direction in dead-end, also works in left/right turns or symmetric-switches non-facing)
                - DO_NOTHING: if already moving, keep moving forward without acceleration (swap direction in dead-end, also works in left/right turns or symmetric-switches non-facing); if stopped, stay stopped.
        """
        new_direction, transition_valid, preprocessed_action = self._check_action_new(action, position, direction)
        new_position = get_new_position(position, new_direction)
        new_cell_valid = self.check_bounds(new_position) and self.get_full_transitions(*new_position) > 0
        return new_cell_valid, new_direction, new_position, transition_valid, preprocessed_action
