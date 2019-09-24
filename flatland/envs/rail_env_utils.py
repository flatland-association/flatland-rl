import math
from typing import Tuple, Set, Dict, List, NamedTuple

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.distance_map import DistanceMap
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvNextAction, RailEnvActions
from flatland.envs.rail_generators import rail_from_file
from flatland.envs.schedule_generators import schedule_from_file
from flatland.utils.ordered_set import OrderedSet

ShortestPathElement = \
    NamedTuple('Path_Element',
               [('position', Tuple[int, int]), ('direction', int), ('next_action_element', RailEnvNextAction)])


def load_flatland_environment_from_file(file_name, load_from_package=None, obs_builder_object=None):
    if obs_builder_object is None:
        obs_builder_object = TreeObsForRailEnv(
            max_depth=2,
            predictor=ShortestPathPredictorForRailEnv(max_depth=10))
    environment = RailEnv(width=1,
                          height=1,
                          rail_generator=rail_from_file(file_name, load_from_package),
                          number_of_agents=1,
                          schedule_generator=schedule_from_file(file_name, load_from_package),
                          obs_builder_object=obs_builder_object)
    return environment


def get_valid_move_actions_(agent_direction: Grid4TransitionsEnum,
                            agent_position: Tuple[int, int],
                            rail: GridTransitionMap) -> Set[RailEnvNextAction]:
    """
    Get the valid move actions (forward, left, right) for an agent.

    Parameters
    ----------
    agent_direction : Grid4TransitionsEnum
    agent_position: Tuple[int,int]
    rail : GridTransitionMap


    Returns
    -------
    Set of `RailEnvNextAction` (tuples of (action,position,direction))
        Possible move actions (forward,left,right) and the next position/direction they lead to.
        It is not checked that the next cell is free.
    """
    valid_actions: Set[RailEnvNextAction] = OrderedSet()
    possible_transitions = rail.get_transitions(*agent_position, agent_direction)
    num_transitions = np.count_nonzero(possible_transitions)
    # Start from the current orientation, and see which transitions are available;
    # organize them as [left, forward, right], relative to the current orientation
    # If only one transition is possible, the forward branch is aligned with it.
    if rail.is_dead_end(agent_position):
        action = RailEnvActions.MOVE_FORWARD
        exit_direction = (agent_direction + 2) % 4
        if possible_transitions[exit_direction]:
            new_position = get_new_position(agent_position, exit_direction)
            valid_actions.add(RailEnvNextAction(action, new_position, exit_direction))
    elif num_transitions == 1:
        action = RailEnvActions.MOVE_FORWARD
        for new_direction in [(agent_direction + i) % 4 for i in range(-1, 2)]:
            if possible_transitions[new_direction]:
                new_position = get_new_position(agent_position, new_direction)
                valid_actions.add(RailEnvNextAction(action, new_position, new_direction))
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
                valid_actions.add(RailEnvNextAction(action, new_position, new_direction))
    return valid_actions


def get_shortest_paths(distance_map: DistanceMap) -> Dict[int, List[ShortestPathElement]]:
    # TODO: do we need to support unreachable targets?
    # TODO refactoring: unify with predictor (support agent.moving and max_depth)
    shortest_paths = dict()
    for a in distance_map.agents:
        position = a.position
        direction = a.direction
        shortest_paths[a.handle] = []
        distance = math.inf
        while (position != a.target):
            next_actions = get_valid_move_actions_(direction, position, distance_map.rail)

            best_next_action = None
            for next_action in next_actions:
                next_action_distance = distance_map.get()[
                    a.handle, next_action.next_position[0], next_action.next_position[1], next_action.next_direction]
                if next_action_distance < distance:
                    best_next_action = next_action
                    distance = next_action_distance

            shortest_paths[a.handle].append(ShortestPathElement(position, direction, best_next_action))

            position = best_next_action.next_position
            direction = best_next_action.next_direction

    return shortest_paths
