"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""

from flatland.core.grid.grid4_astar import a_star
from flatland.core.grid.grid4_utils import get_direction, mirror


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


def connect_nodes(rail_trans, rail_array, start, end):
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
                # don't set any transition at node yet
                new_trans = 0
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
                # don't set any transition at node yet

                new_trans_e = 0
            else:
                # into existing rail
                new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, new_dir, 1)
            rail_array[end_pos] = new_trans_e

        current_dir = new_dir
    return path


def connect_from_nodes(rail_trans, rail_array, start, end):
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
                new_trans = 0
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


def connect_to_nodes(rail_trans, rail_array, start, end):
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
                new_trans_e = 0
            else:
                # into existing rail
                new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, new_dir, 1)
            rail_array[end_pos] = new_trans_e

        current_dir = new_dir
    return path
