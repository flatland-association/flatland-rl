
"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""
# import numpy as np

# from flatland.core.env import Environment
# from flatland.core.env_observation_builder import TreeObsForRailEnv

# from flatland.core.transitions import Grid8Transitions, RailEnvTransitions
# from flatland.core.transition_map import GridTransitionMap


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

    def update_if_better(self, other):
        if other.g < self.g:
            self.parent = other.parent
            self.g = other.g
            self.h = other.h
            self.f = other.f


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
            # new_trans = rail_trans.set_transition(new_trans, mirror(new_dir), mirror(current_dir), 1)
            # rail_trans.print(new_trans)
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
            # new_trans_e = rail_trans.set_transition(new_trans_e, mirror(new_dir), mirror(new_dir), 1)
            # print("end:", end_pos, current_pos)
            # rail_trans.print(new_trans_e)

        # print("========> end trans")
        # rail_trans.print(new_trans_e)
        if not rail_trans.is_valid(new_trans_e):
            # print("end failed", end_pos, current_pos)
            return False
        # else:
        #    print("end ok!", end_pos, current_pos)

    # is transition is valid?
    # print("=======> trans")
    # rail_trans.print(new_trans)
    return rail_trans.is_valid(new_trans)


def a_star(rail_trans, rail_array, start, end):
    """
    Returns a list of tuples as a path from the given start to end.
    If no path is found, returns path to closest point to end.
    """
    rail_shape = rail_array.shape
    start_node = AStarNode(None, start)
    end_node = AStarNode(None, end)
    open_list = []
    closed_list = []

    open_list.append(start_node)

    # this could be optimized
    def is_node_in_list(node, the_list):
        for o_node in the_list:
            if node == o_node:
                return o_node
        return None

    while len(open_list) > 0:
        # get node with current shortest est. path (lowest f)
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # print("a*:", current_node.pos)
        # for cn in closed_list:
        #    print("closed:", cn.pos)

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
            if node_pos[0] >= rail_shape[0] or \
                    node_pos[0] < 0 or \
                    node_pos[1] >= rail_shape[1] or \
                    node_pos[1] < 0:
                continue

            # validate positions
            # debug: avoid all current rails
            # if rail_array.item(node_pos) != 0:
            #    continue

            # validate positions
            if not validate_new_transition(rail_trans, rail_array, prev_pos, current_node.pos, node_pos, end_node.pos):
                # print("A*: transition invalid")
                continue

            # create new node
            new_node = AStarNode(current_node, node_pos)
            children.append(new_node)

        # loop through children
        for child in children:
            # already in closed list?
            closed_node = is_node_in_list(child, closed_list)
            if closed_node is not None:
                continue

            # create the f, g, and h values
            child.g = current_node.g + 1
            # this heuristic favors diagonal paths
            # child.h = ((child.pos[0] - end_node.pos[0]) ** 2) + \
            #           ((child.pos[1] - end_node.pos[1]) ** 2)
            # this heuristic avoids diagonal paths
            child.h = abs(child.pos[0] - end_node.pos[0]) + abs(child.pos[1] - end_node.pos[1])
            child.f = child.g + child.h

            # already in the open list?
            open_node = is_node_in_list(child, open_list)
            if open_node is not None:
                open_node.update_if_better(child)
                continue

            # add the child to the open list
            open_list.append(child)

        # no full path found, return partial path
        if len(open_list) == 0:
            path = []
            current = current_node
            while current is not None:
                path.append(current.pos)
                current = current.parent
            # return reversed path
            print("partial:", start, end, path[::-1])
            return path[::-1]


def connect_rail(rail_trans, rail_array, start, end):
    """
    Creates a new path [start,end] in rail_array, based on rail_trans.
    """
    # in the worst case we will need to do a A* search, so we might as well set that up
    path = a_star(rail_trans, rail_array, start, end)
    # print("connecting path", path)
    if len(path) < 2:
        return
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
                # new_trans = rail_trans.set_transition(new_trans, mirror(new_dir), mirror(current_dir), 1)
                pass
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
                # new_trans_e = rail_trans.set_transition(new_trans_e, mirror(new_dir), mirror(new_dir), 1)
            rail_array[end_pos] = new_trans_e

        current_dir = new_dir


def distance_on_rail(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

