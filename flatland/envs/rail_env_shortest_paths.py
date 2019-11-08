import math
from typing import Dict, List, Optional, NamedTuple, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import RailAgentStatus
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions, RailEnv
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
def get_shortest_paths(distance_map: DistanceMap, max_depth: Optional[int] = None, agent_handle: Optional[int] = None) \
    -> Dict[int, Optional[List[WalkingElement]]]:
    """
    Computes the shortest path for each agent to its target and the action to be taken to do so.
    The paths are derived from a `DistanceMap`.

    If there is no path (rail disconnected), the path is given as None.
    The agent state (moving or not) and its speed are not taken into account

    example:
            agent_fixed_travel_paths = get_shortest_paths(env.distance_map, None, agent.handle)
            path = agent_fixed_travel_paths[agent.handle]

    Parameters
    ----------
    distance_map : reference to the distance_map
    max_depth : max path length, if the shortest path is longer, it will be cutted
    agent_handle : if set, the shortest for agent.handle will be returned , otherwise for all agents

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

    if agent_handle is not None:
        _shortest_path_for_agent(distance_map.agents[agent_handle])
    else:
        for agent in distance_map.agents:
            _shortest_path_for_agent(agent)

    return shortest_paths


def get_k_shortest_paths(env: RailEnv,
                         source_position: Tuple[int, int],
                         source_direction: int,
                         target_position=Tuple[int, int],
                         k: int = 1, debug=False) -> List[Tuple[WalkingElement]]:
    """
    Computes the k shortest paths using modified Dijkstra
    following pseudo-code https://en.wikipedia.org/wiki/K_shortest_path_routing
    In contrast to the pseudo-code in wikipedia, we do not a allow for loopy paths.

    Parameters
    ----------
    env :             RailEnv
    source_position:  Tuple[int,int]
    source_direction: int
    target_position:  Tuple[int,int]
    k :               int
        max number of shortest paths
    debug:            bool
        print debug statements

    Returns
    -------
    List[Tuple[WalkingElement]]
        We use tuples since we need the path elements to be hashable.
        The walking elements do not contain any actions.
        We use a list of paths in order to keep the order of length.
    """

    # P: set of shortest paths from s to t
    # P =empty,
    shortest_paths: List[Tuple[WalkingElement]] = []

    # countu: number of shortest paths found to node u
    # countu = 0, for all u in V
    count = {(r, c, d): 0 for r in range(env.height) for c in range(env.width) for d in range(4)}

    # B is a heap data structure containing paths
    heap: Set[Tuple[WalkingElement]] = set()

    # insert path Ps = {s} into B with cost 0
    heap.add((WalkingElement(source_position, source_direction, None),))

    # while B is not empty and countt < K:
    while len(heap) > 0 and len(shortest_paths) < k:
        if debug:
            print("iteration heap={}, shortest_paths={}".format(heap, shortest_paths))
        # – let Pu be the shortest cost path in B with cost C
        c = np.inf
        pu = None
        for path in heap:
            if len(path) < c:
                pu = path
                c = len(path)
        u: WalkingElement = pu[-1]
        if debug:
            print("  looking at pu={}".format(pu))

        #     – B = B − {Pu }
        heap.remove(pu)
        #     – countu = countu + 1

        urcd = (*u.position, u.direction)
        count[urcd] += 1

        # – if u = t then P = P U {Pu}
        if u.position == target_position:
            print(" found of length {} {}".format(len(pu), pu))
            shortest_paths.append(pu)

        # – if countu ≤ K then
        # CAVEAT: do not allow for loopy paths
        elif count[urcd] <= k:
            possible_transitions = env.rail.get_transitions(*urcd)
            if debug:
                print("  looking at neighbors of u={}, transitions are {}".format(u, possible_transitions))
            #     for each vertex v adjacent to u:
            for new_direction in range(4):
                if debug:
                    print("        looking at new_direction={}".format(new_direction))
                if possible_transitions[new_direction]:
                    new_position = get_new_position(u.position, new_direction)
                    if debug:
                        print("        looking at neighbor v={}".format((*new_position, new_direction)))

                    v = WalkingElement(position=new_position, direction=new_direction, next_action_element=None)
                    # CAVEAT: do not allow for loopy paths
                    if v in pu:
                        continue

                    # – let Pv be a new path with cost C + w(u, v) formed by concatenating edge (u, v) to path Pu
                    pv = pu + (v,)
                    #     – insert Pv into B
                    heap.add(pv)

    # return P
    return shortest_paths


def visualize_distance_map(distance_map: DistanceMap, agent_handle: int = 0):
    if agent_handle >= distance_map.get().shape[0]:
        print("Error: agent_handle cannot be larger than actual number of agents")
        return
    # take min value of all 4 directions
    min_distance_map = np.min(distance_map.get(), axis=3)
    plt.imshow(min_distance_map[agent_handle][:][:])
    plt.show()
