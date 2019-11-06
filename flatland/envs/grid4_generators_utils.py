"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_astar import a_star
from flatland.core.grid.grid4_utils import get_direction, mirror, direction_to_point, get_new_position
from flatland.core.grid.grid_utils import IntVector2D, IntVector2DDistance, IntVector2DArray
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.transition_map import GridTransitionMap, RailEnvTransitions


def connect_rail_in_grid_map(grid_map: GridTransitionMap, start: IntVector2D, end: IntVector2D,
                             rail_trans: RailEnvTransitions,
                             a_star_distance_function: IntVector2DDistance = Vec2d.get_manhattan_distance,
                             flip_start_node_trans: bool = False, flip_end_node_trans: bool = False,
                             respect_transition_validity: bool = True, forbidden_cells: IntVector2DArray = None,
                             avoid_rail=False) -> IntVector2DArray:
    """
        Creates a new path [start,end] in `grid_map.grid`, based on rail_trans, and
    returns the path created as a list of positions.
    :param avoid_rail:
    :param rail_trans: basic rail transition object
    :param grid_map: grid map
    :param start: start position of rail
    :param end: end position of rail
    :param flip_start_node_trans: make valid start position by adding dead-end, empty start if False
    :param flip_end_node_trans: make valid end position by adding dead-end, empty end if False
    :param respect_transition_validity: Only draw rail maps if legal rail elements can be use, False, draw line without
    respecting rail transitions.
    :param a_star_distance_function: Define what distance function a-star should use
    :param forbidden_cells: cells to avoid when drawing rail. Rail cannot go through this list of cells
    :return: List of cells in the path
    """

    path: IntVector2DArray = a_star(grid_map, start, end, a_star_distance_function, avoid_rail,
                                    respect_transition_validity,
                                    forbidden_cells)
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


def connect_straight_line_in_grid_map(grid_map: GridTransitionMap, start: IntVector2D,
                                      end: IntVector2D, rail_trans: RailEnvTransitions) -> IntVector2DArray:
    """
    Generates a straight rail line from start cell to end cell.
    Diagonal lines are not allowed
    :param rail_trans:
    :param grid_map:
    :param start: Cell coordinates for start of line
    :param end: Cell coordinates for end of line
    :return: A list of all cells in the path
    """

    if not (start[0] == end[0] or start[1] == end[1]):
        print("No straight line possible!")
        return []

    direction = direction_to_point(start, end)

    if direction is Grid4TransitionsEnum.NORTH or direction is Grid4TransitionsEnum.SOUTH:
        start_row = min(start[0], end[0])
        end_row = max(start[0], end[0]) + 1
        rows = np.arange(start_row, end_row)
        length = np.abs(end[0] - start[0]) + 1
        cols = np.repeat(start[1], length)

    else:  # Grid4TransitionsEnum.EAST or Grid4TransitionsEnum.WEST
        start_col = min(start[1], end[1])
        end_col = max(start[1], end[1]) + 1
        cols = np.arange(start_col, end_col)
        length = np.abs(end[1] - start[1]) + 1
        rows = np.repeat(start[0], length)

    path = list(zip(rows, cols))

    for cell in path:
        transition = grid_map.grid[cell]
        transition = rail_trans.set_transition(transition, direction, direction, 1)
        transition = rail_trans.set_transition(transition, mirror(direction), mirror(direction), 1)
        grid_map.grid[cell] = transition

    return path


def fix_inner_nodes(grid_map: GridTransitionMap, inner_node_pos: IntVector2D, rail_trans: RailEnvTransitions):
    """
    Fix inner city nodes by connecting it to its neighbouring parallel track
    :param grid_map:
    :param inner_node_pos: inner city node to fix
    :param rail_trans:
    :return:
    """
    corner_directions = []
    for direction in range(4):
        tmp_pos = get_new_position(inner_node_pos, direction)
        if grid_map.grid[tmp_pos] > 0:
            corner_directions.append(direction)
    if len(corner_directions) == 2:
        transition = 0
        transition = rail_trans.set_transition(transition, mirror(corner_directions[0]), corner_directions[1], 1)
        transition = rail_trans.set_transition(transition, mirror(corner_directions[1]), corner_directions[0], 1)
        grid_map.grid[inner_node_pos] = transition
        tmp_pos = get_new_position(inner_node_pos, corner_directions[0])
        transition = grid_map.grid[tmp_pos]
        transition = rail_trans.set_transition(transition, corner_directions[0], mirror(corner_directions[0]), 1)
        grid_map.grid[tmp_pos] = transition
        tmp_pos = get_new_position(inner_node_pos, corner_directions[1])
        transition = grid_map.grid[tmp_pos]
        transition = rail_trans.set_transition(transition, corner_directions[1], mirror(corner_directions[1]),
                                               1)
        grid_map.grid[tmp_pos] = transition
    return


def align_cell_to_city(city_center, city_orientation, cell):
    """
    Alig all cells to face the city center along the city orientation
    @param city_center: Center needed for orientation
    @param city_orientation: Orientation of the city
    @param cell: Cell we would like to orient
    :@return: Orientation of cell towards city center along axis of city orientation
    """
    if city_orientation % 2 == 0:
        return int(2 * np.clip(cell[0] - city_center[0], 0, 1))
    else:
        return int(2 * np.clip(city_center[1] - cell[1], 0, 1)) + 1
