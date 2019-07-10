"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""

import numpy as np

from flatland.core.grid.grid4_astar import a_star
from flatland.core.grid.grid4_utils import get_direction, mirror, get_new_position


def connect_rail(rail_trans, rail_array, start, end):
    """
    Creates a new path [start,end] in rail_array, based on rail_trans.
    """
    # in the worst case we will need to do a A* search, so we might as well set that up
    path = a_star(rail_trans, rail_array, start, end)
    if len(path) < 2:
        return []
    current_dir = get_direction(path[0], path[1])
    end_pos = path[-1]
    for index in range(len(path) - 1):
        current_pos = path[index]
        new_pos = path[index + 1]
        new_dir = get_direction(current_pos, new_pos)

        new_trans = rail_array[current_pos]
        if index == 0:
            if new_trans == 0:
                # end-point
                # need to flip direction because of how end points are defined
                new_trans = rail_trans.set_transition(new_trans, mirror(current_dir), new_dir, 1)
            else:
                # into existing rail
                new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
        else:
            # set the forward path
            new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
            # set the backwards path
            new_trans = rail_trans.set_transition(new_trans, mirror(new_dir), mirror(current_dir), 1)
        rail_array[current_pos] = new_trans

        if new_pos == end_pos:
            # setup end pos setup
            new_trans_e = rail_array[end_pos]
            if new_trans_e == 0:
                # end-point
                new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, mirror(new_dir), 1)
            else:
                # into existing rail
                new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, new_dir, 1)
            rail_array[end_pos] = new_trans_e

        current_dir = new_dir
    return path


def get_rnd_agents_pos_tgt_dir_on_rail(rail, num_agents):
    """
    Given a `rail' GridTransitionMap, return a random placement of agents (initial position, direction and target).

    TODO: add extensive documentation, as users may need this function to simplify their custom level generators.
    """

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
                if m[0] not in valid_starting_directions and _path_exists(rail, new_position, m[0], agents_target[i]):
                    valid_starting_directions.append(m[0])

            if len(valid_starting_directions) == 0:
                re_generate = True
            else:
                agents_direction[i] = valid_starting_directions[np.random.choice(len(valid_starting_directions), 1)[0]]

    return agents_position, agents_direction, agents_target
