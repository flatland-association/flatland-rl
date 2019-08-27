"""Agent generators (railway undertaking, "EVU")."""
from typing import Tuple, List, Callable, Mapping, Optional, Any

import msgpack
import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgentStatic

AgentPosition = Tuple[int, int]
AgentGeneratorProduct = Tuple[List[AgentPosition], List[AgentPosition], List[AgentPosition], List[float]]
AgentGenerator = Callable[[GridTransitionMap, int, Optional[Any]], AgentGeneratorProduct]


def speed_initialization_helper(nb_agents: int, speed_ratio_map: Mapping[float, float] = None) -> List[float]:
    """
    Parameters
    -------
    nb_agents : int
        The number of agents to generate a speed for
    speed_ratio_map : Mapping[float,float]
        A map of speeds mappint to their ratio of appearance. The ratios must sum up to 1.

    Returns
    -------
    List[float]
        A list of size nb_agents of speeds with the corresponding probabilistic ratios.
    """
    if speed_ratio_map is None:
        return [1.0] * nb_agents

    nb_classes = len(speed_ratio_map.keys())
    speed_ratio_map_as_list: List[Tuple[float, float]] = list(speed_ratio_map.items())
    speed_ratios = list(map(lambda t: t[1], speed_ratio_map_as_list))
    speeds = list(map(lambda t: t[0], speed_ratio_map_as_list))
    return list(map(lambda index: speeds[index], np.random.choice(nb_classes, nb_agents, p=speed_ratios)))


def complex_rail_generator_agents_placer(speed_ratio_map: Mapping[float, float] = None) -> AgentGenerator:
    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None):
        start_goal = hints['start_goal']
        start_dir = hints['start_dir']
        agents_position = [sg[0] for sg in start_goal[:num_agents]]
        agents_target = [sg[1] for sg in start_goal[:num_agents]]
        agents_direction = start_dir[:num_agents]

        if speed_ratio_map:
            speeds = speed_initialization_helper(num_agents, speed_ratio_map)
        else:
            speeds = [1.0] * len(agents_position)

        return agents_position, agents_direction, agents_target, speeds

    return generator


def get_rnd_agents_pos_tgt_dir_on_rail(speed_ratio_map: Mapping[float, float] = None) -> AgentGenerator:
    """
    Given a `rail' GridTransitionMap, return a random placement of agents (initial position, direction and target).

    Parameters
    -------
        rail : GridTransitionMap
            The railway to place agents on.
        num_agents : int
            The number of agents to generate a speed for
        speed_ratio_map : Mapping[float,float]
            A map of speeds mappint to their ratio of appearance. The ratios must sum up to 1.
    Returns
    -------
        Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
        initial positions, directions, targets speeds
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None):
        def _path_exists(rail, start, direction, end):
            # BFS - Check if a path exists between the 2 nodes

            visited = set()
            stack = [(start, direction)]
            while stack:
                node = stack.pop()
                if node[0][0] == end[0] and node[0][1] == end[1]:
                    return 1
                if node not in visited:
                    visited.add(node)
                    moves = rail.get_transitions(node[0][0], node[0][1], node[1])
                    for move_index in range(4):
                        if moves[move_index]:
                            stack.append((get_new_position(node[0], move_index),
                                          move_index))

                    # If cell is a dead-end, append previous node with reversed
                    # orientation!
                    nbits = 0
                    tmp = rail.get_full_transitions(node[0][0], node[0][1])
                    while tmp > 0:
                        nbits += (tmp & 1)
                        tmp = tmp >> 1
                    if nbits == 1:
                        stack.append((node[0], (node[1] + 2) % 4))

            return 0

        valid_positions = []
        for r in range(rail.height):
            for c in range(rail.width):
                if rail.get_full_transitions(r, c) > 0:
                    valid_positions.append((r, c))
        if len(valid_positions) == 0:
            return [], [], [], []
        re_generate = True
        while re_generate:
            agents_position = [
                valid_positions[i] for i in
                np.random.choice(len(valid_positions), num_agents)]
            agents_target = [
                valid_positions[i] for i in
                np.random.choice(len(valid_positions), num_agents)]

            # agents_direction must be a direction for which a solution is
            # guaranteed.
            agents_direction = [0] * num_agents
            re_generate = False
            for i in range(num_agents):
                valid_movements = []
                for direction in range(4):
                    position = agents_position[i]
                    moves = rail.get_transitions(position[0], position[1], direction)
                    for move_index in range(4):
                        if moves[move_index]:
                            valid_movements.append((direction, move_index))

                valid_starting_directions = []
                for m in valid_movements:
                    new_position = get_new_position(agents_position[i], m[1])
                    if m[0] not in valid_starting_directions and _path_exists(rail, new_position, m[0],
                                                                              agents_target[i]):
                        valid_starting_directions.append(m[0])

                if len(valid_starting_directions) == 0:
                    re_generate = True
                else:
                    agents_direction[i] = valid_starting_directions[
                        np.random.choice(len(valid_starting_directions), 1)[0]]

        agents_speed = speed_initialization_helper(num_agents, speed_ratio_map)
        return agents_position, agents_direction, agents_target, agents_speed

    return generator


def agents_from_file(filename) -> AgentGenerator:
    """
    Utility to load pickle file

    Parameters
    -------
    input_file : Pickle file generated by env.save() or editor

    Returns
    -------
    Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
        initial positions, directions, targets speeds
    """

    def generator(rail: GridTransitionMap, num_agents: int, hints: Any = None):
        with open(filename, "rb") as file_in:
            load_data = file_in.read()
        data = msgpack.unpackb(load_data, use_list=False)

        # agents are always reset as not moving
        agents_static = [EnvAgentStatic(d[0], d[1], d[2], moving=False) for d in data[b"agents_static"]]
        # setup with loaded data
        agents_position = [a.position for a in agents_static]
        agents_direction = [a.direction for a in agents_static]
        agents_target = [a.target for a in agents_static]

        return agents_position, agents_direction, agents_target, [1.0] * len(agents_position)

    return generator
