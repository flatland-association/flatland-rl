import math
from typing import Dict, List, Optional, NamedTuple, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions
from flatland.utils.ordered_set import OrderedSet

WalkingElement = \
    NamedTuple('WalkingElement',
               [('position', Tuple[int, int]), ('direction', int), ('next_action_element', RailEnvNextAction)])


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


# N.B. get_shortest_paths is not part of distance_map since it refers to RailEnvActions (would lead to circularity!)
def get_shortest_paths(distance_map: DistanceMap, max_depth: Optional[int] = None) \
    -> Dict[int, Optional[List[WalkingElement]]]:
    """
    Computes the shortest path for each agent to its target and the action to be taken to do so.
    The paths are derived from a `DistanceMap`.

    If there is no path (rail disconnected), the path is given as None.
    The agent state (moving or not) and its speed are not taken into account

    Parameters
    ----------
    distance_map

    Returns
    -------
        Dict[int, Optional[List[WalkingElement]]]

    """
    shortest_paths = dict()

    def _shortest_path_for_agent(agent):
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            position = agent.initial_position
        elif agent.status == RailAgentStatus.ACTIVE:
            position = agent.position
        elif agent.status == RailAgentStatus.DONE:
            position = agent.target
        else:
            shortest_paths[agent.handle] = None
            return
        direction = agent.direction
        shortest_paths[agent.handle] = []
        distance = math.inf
        depth = 0
        while (position != agent.target and (max_depth is None or depth < max_depth)):
            next_actions = get_valid_move_actions_(direction, position, distance_map.rail)
            best_next_action = None
            for next_action in next_actions:
                next_action_distance = distance_map.get()[
                    agent.handle, next_action.next_position[0], next_action.next_position[
                        1], next_action.next_direction]
                if next_action_distance < distance:
                    best_next_action = next_action
                    distance = next_action_distance

            shortest_paths[agent.handle].append(WalkingElement(position, direction, best_next_action))
            depth += 1

            # if there is no way to continue, the rail must be disconnected!
            # (or distance map is incorrect)
            if best_next_action is None:
                shortest_paths[agent.handle] = None
                return

            position = best_next_action.next_position
            direction = best_next_action.next_direction
        if max_depth is None or depth < max_depth:
            shortest_paths[agent.handle].append(
                WalkingElement(position, direction,
                               RailEnvNextAction(RailEnvActions.STOP_MOVING, position, direction)))

    for agent in distance_map.agents:
        _shortest_path_for_agent(agent)

    return shortest_paths


def visualize_distance_map(distance_map: DistanceMap, agent_handle: int = 0):
    if agent_handle >= distance_map.get().shape[0]:
        print("Error: agent_handle cannot be larger than actual number of agents")
        return
    # take min value of all 4 directions
    min_distance_map = np.min(distance_map.get(), axis=3)
    plt.imshow(min_distance_map[agent_handle][:][:])
    plt.show()
