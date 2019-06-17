"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""

import numpy as np

from flatland.core.transitions import Grid4TransitionsEnum


def get_direction(pos1, pos2):
    """
    Assumes pos1 and pos2 are adjacent location on grid.
    Returns direction (int) that can be used with transitions.
    """
    diff_0 = pos2[0] - pos1[0]
    diff_1 = pos2[1] - pos1[1]
    if diff_0 < 0:
        return 0
    if diff_0 > 0:
        return 2
    if diff_1 > 0:
        return 1
    if diff_1 < 0:
        return 3
    return 0


def mirror(dir):
    return (dir + 2) % 4


def validate_new_transition(rail_trans, rail_array, prev_pos, current_pos, new_pos, end_pos):
    # start by getting direction used to get to current node
    # and direction from current node to possible child node
    new_dir = get_direction(current_pos, new_pos)
    if prev_pos is not None:
        current_dir = get_direction(prev_pos, current_pos)
    else:
        current_dir = new_dir
    # create new transition that would go to child
    new_trans = rail_array[current_pos]
    if prev_pos is None:
        if new_trans == 0:
            # need to flip direction because of how end points are defined
            new_trans = rail_trans.set_transition(new_trans, mirror(current_dir), new_dir, 1)
        else:
            # check if matches existing layout
            new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
    else:
        # set the forward path
        new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
        # set the backwards path
        new_trans = rail_trans.set_transition(new_trans, mirror(new_dir), mirror(current_dir), 1)
    if new_pos == end_pos:
        # need to validate end pos setup as well
        new_trans_e = rail_array[end_pos]
        if new_trans_e == 0:
            # need to flip direction because of how end points are defined
            new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, mirror(new_dir), 1)
        else:
            # check if matches existing layout
            new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, new_dir, 1)

        if not rail_trans.is_valid(new_trans_e):
            return False

    # is transition is valid?
    return rail_trans.is_valid(new_trans)


def position_to_coordinate(depth, position):
    """
         [ (0,0) (0,1) ..  (0,w)
           (1,0) (1,1)     (1,w)
           ...
           (d,0) (d,1)     (d,w) ]

         -->

         [ 0      1    ..   w
           w+1    w+2  ..   2w
           ...
           d*w+1  d*w+

    :param depth:
    :param position:
    :return:
    """
    coords = ()
    for p in position:
        coords = coords + ((int(p) % depth, int(p) // depth),)  # changed x_dim to y_dim
    return coords


def coordinate_to_position(depth, coords):
    """
    Helper function to

    :param depth:
    :param coords:
    :return:
    """
    position = np.empty(len(coords), dtype=int)
    idx = 0
    for t in coords:
        position[idx] = int(t[1] * depth + t[0])
        idx += 1
    return position


class AStarNode():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, pos=None):
        self.parent = parent
        self.pos = pos
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)

    def update_if_better(self, other):
        if other.g < self.g:
            self.parent = other.parent
            self.g = other.g
            self.h = other.h
            self.f = other.f


def a_star(rail_trans, rail_array, start, end):
    """
    Returns a list of tuples as a path from the given start to end.
    If no path is found, returns path to closest point to end.
    """
    rail_shape = rail_array.shape
    start_node = AStarNode(None, start)
    end_node = AStarNode(None, end)
    open_nodes = set()
    closed_nodes = set()
    open_nodes.add(start_node)

    while len(open_nodes) > 0:
        # get node with current shortest est. path (lowest f)
        current_node = None
        for item in open_nodes:
            if current_node is None:
                current_node = item
                continue
            if item.f < current_node.f:
                current_node = item

        # pop current off open list, add to closed list
        open_nodes.remove(current_node)
        closed_nodes.add(current_node)

        # found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.pos)
                current = current.parent
            # return reversed path
            return path[::-1]

        # generate children
        children = []
        if current_node.parent is not None:
            prev_pos = current_node.parent.pos
        else:
            prev_pos = None
        for new_pos in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_pos = (current_node.pos[0] + new_pos[0], current_node.pos[1] + new_pos[1])
            if node_pos[0] >= rail_shape[0] or node_pos[0] < 0 or node_pos[1] >= rail_shape[1] or node_pos[1] < 0:
                continue

            # validate positions
            if not validate_new_transition(rail_trans, rail_array, prev_pos, current_node.pos, node_pos, end_node.pos):
                continue

            # create new node
            new_node = AStarNode(current_node, node_pos)
            children.append(new_node)

        # loop through children
        for child in children:
            # already in closed list?
            if child in closed_nodes:
                continue

            # create the f, g, and h values
            child.g = current_node.g + 1
            # this heuristic favors diagonal paths:
            # child.h = ((child.pos[0] - end_node.pos[0]) ** 2) + ((child.pos[1] - end_node.pos[1]) ** 2) \#  noqa: E800
            # this heuristic avoids diagonal paths
            child.h = abs(child.pos[0] - end_node.pos[0]) + abs(child.pos[1] - end_node.pos[1])
            child.f = child.g + child.h

            # already in the open list?
            if child in open_nodes:
                continue

            # add the child to the open list
            open_nodes.add(child)

        # no full path found
        if len(open_nodes) == 0:
            return []


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


def distance_on_rail(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_new_position(position, movement):
    """ Utility function that converts a compass movement over a 2D grid to new positions (r, c). """
    if movement == Grid4TransitionsEnum.NORTH:
        return (position[0] - 1, position[1])
    elif movement == Grid4TransitionsEnum.EAST:
        return (position[0], position[1] + 1)
    elif movement == Grid4TransitionsEnum.SOUTH:
        return (position[0] + 1, position[1])
    elif movement == Grid4TransitionsEnum.WEST:
        return (position[0], position[1] - 1)


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
                moves = rail.get_transitions((node[0][0], node[0][1], node[1]))
                for move_index in range(4):
                    if moves[move_index]:
                        stack.append((get_new_position(node[0], move_index),
                                      move_index))

                # If cell is a dead-end, append previous node with reversed
                # orientation!
                nbits = 0
                tmp = rail.get_transitions((node[0][0], node[0][1]))
                while tmp > 0:
                    nbits += (tmp & 1)
                    tmp = tmp >> 1
                if nbits == 1:
                    stack.append((node[0], (node[1] + 2) % 4))

        return 0

    valid_positions = []
    for r in range(rail.height):
        for c in range(rail.width):
            if rail.get_transitions((r, c)) > 0:
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
                moves = rail.get_transitions((position[0], position[1], direction))
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
