"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""

from flatland.core.grid.grid4_astar import a_star
from flatland.core.grid.grid4_utils import get_direction, mirror
from flatland.core.grid.grid_utils import IntVector2D
from flatland.core.transition_map import GridTransitionMap, RailEnvTransitions


def connect_basic_operation(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap,
                            start: IntVector2D,
                            end: IntVector2D,
                            flip_start_node_trans=False,
                            flip_end_node_trans=False):
    """
    Creates a new path [start,end] in grid_map, based on rail_trans.
    """
    # in the worst case we will need to do a A* search, so we might as well set that up
    path = a_star(rail_trans, grid_map, start, end)
    if len(path) < 2:
        return []
    current_dir = get_direction(path[0], path[1])
    end_pos = path[-1]
    for index in range(len(path) - 1):
        current_pos = path[index]
        new_pos = path[index + 1]
        new_dir = get_direction(current_pos, new_pos)

        new_trans = grid_map.grid[current_pos]
        if index == 0:
            if new_trans == 0:
                # end-point
                if flip_start_node_trans:
                    # need to flip direction because of how end points are defined
                    new_trans = rail_trans.set_transition(new_trans, mirror(current_dir), new_dir, 1)
                else:
                    new_trans = 0
            else:
                # into existing rail
                new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
        else:
            # set the forward path
            new_trans = rail_trans.set_transition(new_trans, current_dir, new_dir, 1)
            # set the backwards path
            new_trans = rail_trans.set_transition(new_trans, mirror(new_dir), mirror(current_dir), 1)
        grid_map.grid[current_pos] = new_trans

        if new_pos == end_pos:
            # setup end pos setup
            new_trans_e = grid_map.grid[end_pos]
            if new_trans_e == 0:
                # end-point
                if flip_end_node_trans:
                    new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, mirror(new_dir), 1)
                else:
                    new_trans_e = 0
            else:
                # into existing rail
                new_trans_e = rail_trans.set_transition(new_trans_e, new_dir, new_dir, 1)
            grid_map.grid[end_pos] = new_trans_e

        current_dir = new_dir
    return path


def connect_rail(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap, start: IntVector2D, end: IntVector2D):
    return connect_basic_operation(rail_trans, grid_map, start, end, True, True)


def connect_nodes(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap, start: IntVector2D, end: IntVector2D):
    return connect_basic_operation(rail_trans, grid_map, start, end, False, False)


def connect_from_nodes(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap, start: IntVector2D,
                       end: IntVector2D):
    return connect_basic_operation(rail_trans, grid_map, start, end, False, True)


def connect_to_nodes(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap, start: IntVector2D, end: IntVector2D):
    return connect_basic_operation(rail_trans, grid_map, start, end, True, False)
