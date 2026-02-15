from functools import lru_cache
from typing import Set, List, Optional
from typing import Tuple

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.grid.grid_utils import IntVector2D
from flatland.core.transition_map import GridTransitionMap
from flatland.core.transitions import Transitions
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_env_action import RailEnvNextAction
from flatland.utils.ordered_set import OrderedSet


class RailGridTransitionMap(GridTransitionMap[RailEnvActions]):

    def __init__(self, width, height, transitions: Transitions = RailEnvTransitions(), grid: np.ndarray = None):
        super().__init__(width=width, height=height, transitions=transitions, grid=grid)

    @lru_cache
    def get_valid_move_actions(self, configuration: Tuple[Tuple[int, int], int]) -> Set[RailEnvNextAction]:
        """
        Get the valid move actions (forward, left, right) for an agent.

        Parameters
        ----------
        configuration: Tuple[Tuple[int,int],int]


        Returns
        -------
        Set of `RailEnvNextAction` (tuples of (action,position,direction))
            Possible move actions (forward,left,right) and the next position/direction they lead to.
            It is not checked that the next cell is free.
        """
        position, direction = configuration

        valid_actions: Set[RailEnvNextAction] = []
        for action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT]:
            new_direction, transition_valid, preprocessed_action, _ = self._check_action_new(action, position, direction)
            new_position = get_new_position(position, new_direction)
            # TODO this is wrong?! should also include whether transitions from direction are defined!
            if transition_valid:
                valid_actions.append(RailEnvNextAction(action, (new_position, new_direction)))
        return valid_actions

    @lru_cache
    def get_successor_configurations(self, configuration: Tuple[Tuple[int, int], int]) -> Set[Tuple[Tuple[int, int], int]]:
        position, direction = configuration
        successors = OrderedSet()
        for action in [RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT]:
            new_direction, transition_valid, preprocessed_action, _ = self._check_action_new(action, position, direction)
            # TODO this is wrong?! should also include whether transitions from direction are defined!
            new_position = get_new_position(position, new_direction)
            if transition_valid and self.check_bounds(new_position):
                successors.add((new_position, new_direction))
        return successors

    @lru_cache
    def get_predecessor_configurations(self, configuration: Tuple[Tuple[int, int], int]) -> Set[Tuple[Tuple[int, int], int]]:
        position, direction = configuration
        predecessors = OrderedSet()

        # The agent must land into the current cell with orientation `direction`.
        # This is only possible if the agent has arrived from the cell in the opposite direction!
        possible_directions = [(direction + 2) % 4]

        for neigh_direction in possible_directions:
            new_cell = get_new_position(position, neigh_direction)

            if self.check_bounds(new_cell):

                desired_movement_from_new_cell = (neigh_direction + 2) % 4

                # Check all possible transitions in new_cell
                for agent_orientation in range(4):
                    # Is a transition along movement `desired_movement_from_new_cell' to the current cell possible?
                    is_valid = self.get_transition(((new_cell[0], new_cell[1]), agent_orientation),
                                                   desired_movement_from_new_cell)

                    if is_valid:
                        predecessors.add((new_cell, agent_orientation))
        return predecessors

    @lru_cache
    def is_valid_configuration(self, configuration: Tuple[Tuple[int, int], int]) -> bool:
        position, direction = configuration
        return self.check_bounds(position) and fast_count_nonzero(self.get_transitions(configuration)) > 0

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
        possible_transitions = self.get_transitions((position, direction))
        num_transitions = fast_count_nonzero(possible_transitions)

        if num_transitions == 1:
            # - dead-end, straight line or curved line;
            # new_direction will be the only valid transition
            # - take only available transition
            new_direction = fast_argmax(possible_transitions)
            if action == RailEnvActions.MOVE_LEFT and new_direction != (direction - 1) % 4:
                action = RailEnvActions.MOVE_FORWARD
                return new_direction, False, action, True

            elif action == RailEnvActions.MOVE_RIGHT and new_direction != (direction + 1) % 4:
                action = RailEnvActions.MOVE_FORWARD
                return new_direction, False, action, True

            # straight or dead-end
            return new_direction, True, action, True

        if action == RailEnvActions.MOVE_LEFT:
            new_direction = (direction - 1) % 4
            if possible_transitions[new_direction]:
                return new_direction, True, RailEnvActions.MOVE_LEFT, True
            elif possible_transitions[direction]:
                return direction, False, RailEnvActions.MOVE_FORWARD, True

        elif action == RailEnvActions.MOVE_RIGHT:
            new_direction = (direction + 1) % 4
            if possible_transitions[new_direction]:
                return new_direction, True, RailEnvActions.MOVE_RIGHT, True
            elif possible_transitions[direction]:
                return direction, False, RailEnvActions.MOVE_FORWARD, True
        elif possible_transitions[direction]:
            return direction, True, action, True

        return direction, False, RailEnvActions.STOP_MOVING, False

    # TODO make private and check usages -> prefer use of apply_action_independent
    @lru_cache(maxsize=1_000_000)
    def check_action_on_agent(self, action: RailEnvActions, configuration: Tuple[Tuple[int, int], int]) -> Tuple[
        bool, Tuple[Tuple[int, int], int], bool, RailEnvActions, bool]:
        """

        Returns
        -------
        new_cell_valid: bool
            is the new position and direction valid (i.e. is it within bounds and does it have > 0 outgoing transitions)
        new_position: [ConfigurationType]
            New position after applying the action
        transition_valid: bool
            Whether the transition from old and direction is defined in the grid.
            In other words, can the action be applied directly? False if
            - MOVE_FORWARD/DO_NOTHING when entering symmetric switch
            - MOVE_LEFT/MOVE_RIGHT corrected to MOVE_FORWARD in switches and dead-ends
            However, transition_valid for dead-ends and turns either with the correct MOVE_RIGHT/MOVE_LEFT or MOVE_FORWARD/DO_NOTHING.
        preprocessed_action: [ActionType]
            Corrected action if not transition_valid.

            The preprocessed action has the following semantics:
            - MOVE_LEFT/MOVE_RIGHT: turn left/right without acceleration
            - MOVE_FORWARD: move forward with acceleration (swap direction in dead-end, also works in left/right turns or symmetric-switches non-facing)
            - DO_NOTHING: if already moving, keep moving forward without acceleration (swap direction in dead-end, also works in left/right turns or symmetric-switches non-facing); if stopped, stay stopped.
        """
        position, direction = configuration
        new_direction, transition_valid, preprocessed_action, action_valid = self._check_action_new(action, position, direction)
        new_position = get_new_position(position, new_direction)
        # TODO this is wrong?! should also include whether transitions from direction are defined!
        new_cell_valid = self.check_bounds(new_position) and self.get_full_transitions(*new_position) > 0
        return new_cell_valid, (new_position, new_direction), transition_valid, preprocessed_action, action_valid

    @lru_cache(maxsize=1_000_000)
    def apply_action_independent(self, action: RailEnvActions, configuration: Tuple[Tuple[int, int], int]) -> Optional[
        Tuple[Tuple[Tuple[int, int], int], bool]]:
        new_cell_valid, (new_position, new_direction), transition_valid, preprocessed_action, action_valid = self.check_action_on_agent(action, configuration)
        if action_valid:
            return (new_position, new_direction), transition_valid
        else:
            return None

    def get_valid_directions_on_grid(self, row: int, col: int) -> List[int]:
        """
        Returns directions in which the agent can move
        """
        return self.transitions.get_entry_directions(self.get_full_transitions(row, col))
