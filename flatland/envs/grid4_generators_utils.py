"""
Definition of the RailEnv environment and related level-generation functions.

Generator functions are functions that take width, height and num_resets as arguments and return
a GridTransitionMap object.
"""

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_astar import a_star
from flatland.core.grid.grid4_utils import get_direction, mirror, get_new_position, directions_of_vector
from flatland.core.grid.grid_utils import IntVector2D, IntVector2DDistance, IntVector2DArray
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.transition_map import GridTransitionMap, RailEnvTransitions


def connect_rail(rail_trans: RailEnvTransitions, grid_map: GridTransitionMap, start: IntVector2D, end: IntVector2D,
                 a_star_distance_function: IntVector2DDistance = Vec2d.get_manhattan_distance,
                 flip_start_node_trans=False, flip_end_node_trans=False, respect_transition_validity=True,
                 forbidden_cells=None) -> IntVector2DArray:
    """
        Creates a new path [start,end] in `grid_map.grid`, based on rail_trans, and
    returns the path created as a list of positions.
    :param rail_trans: basic rail transition object
    :param grid_map: grid map
    :param start: start position of rail
    :param end: end position of rail
    :param flip_start_node_trans: make valid start position by adding dead-end, empty start if False
    :param flip_end_node_trans: make valid end position by adding dead-end, empty end if False
    :param respect_transition_validity: Only draw rail maps if legal rail elements can be use, False, draw line without respecting rail transitions.
    :param a_star_distance_function: Define what distance function a-star should use
    :param forbidden_cells: cells to avoid when drawing rail. Rail cannot go through this list of cells
    :return: List of cells in the path
    """

    # in the worst case we will need to do a A* search, so we might as well set that up
    path: IntVector2DArray = a_star(grid_map, start, end, a_star_distance_function, respect_transition_validity,
                                    forbidden_cells)
    # path:  IntVector2DArray = quick_path(grid_map, start, end, forbidden_cells=forbidden_cells, openend=False)
    if len(path) < 2:
        print("No path found", path)
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


def connect_straigt_line(rail_trans, grid_map, start, end, openend=False):
    """
    Generates a straight rail line from start cell to end cell.
    Diagonal lines are not allowed
    :param rail_trans:
    :param grid_map:
    :param start: Cell coordinates for start of line
    :param end: Cell coordinates for end of line
    :param openend: If True then the transition at start and end is set to 0: An empty cell
    :return: A list of all cells in the path
    """

    # Assert that a straight line is possible
    if not (start[0] == end[0] or start[1] == end[1]):
        print("No straight line possible!")
        return []
    current_cell = start
    path = [current_cell]
    new_trans = grid_map.grid[current_cell]
    direction = (np.clip(end[0] - start[0], -1, 1), np.clip(end[1] - start[1], -1, 1))
    if direction[0] == 0:
        if direction[1] > 0:
            direction_int = 1
        else:
            direction_int = 3
    else:
        if direction[0] > 0:
            direction_int = 2
        else:
            direction_int = 0
    new_trans = rail_trans.set_transition(new_trans, direction_int, direction_int, 1)
    new_trans = rail_trans.set_transition(new_trans, mirror(direction_int), mirror(direction_int), 1)
    grid_map.grid[current_cell] = new_trans
    if openend:
        grid_map.grid[current_cell] = 0
    # Set path
    while current_cell != end:
        current_cell = tuple(map(lambda x, y: x + y, current_cell, direction))
        new_trans = grid_map.grid[current_cell]
        new_trans = rail_trans.set_transition(new_trans, direction_int, direction_int, 1)
        new_trans = rail_trans.set_transition(new_trans, mirror(direction_int), mirror(direction_int), 1)
        grid_map.grid[current_cell] = new_trans
        if current_cell == end and openend:
            grid_map.grid[current_cell] = 0
        path.append(current_cell)
    return path


def quick_path(grid_map, start, end, forbidden_cells=[], openend=False):
    """
    Quick path connecting algorithm with simple heuristic to allways follow largest value of vector towards target.
    When obstacle is encountereed second direction of vector is chosen.
    """
    # Helper function to make legal steps
    (height, width) = np.shape(grid_map.grid)

    def _next_legal_step(position, old_direction, target):
        if old_direction is not None:
            mirror_direction = Grid4TransitionsEnum(mirror(old_direction))
        else:
            mirror_direction = 4

        closest_direction, second_closest_direction = directions_of_vector(current_cell, target)

        if closest_direction == mirror_direction:
            closest_direction = second_closest_direction

        next_position = get_new_position(position, closest_direction)
        direction_tries = 1

        # Necessary to overcome city boarder
        if next_position == target:
            return next_position, closest_direction

        while (not np.array_equal(next_position, np.clip(next_position, [0, 0],
                                                         [height - 1,
                                                          width - 1])) or next_position in forbidden_cells):

            if direction_tries > 1:
                closest_direction = (closest_direction + 1) % 4
                if closest_direction == mirror_direction:
                    closest_direction = (closest_direction + 1) % 4
                if direction_tries > 3:
                    return None, None
            else:
                closest_direction = second_closest_direction
                if closest_direction == mirror_direction:
                    closest_direction = (closest_direction + 1) % 4

            next_position = get_new_position(position, closest_direction)
            direction_tries += 1
        return next_position, closest_direction

    current_cell = start
    path = [current_cell]
    current_direction = None

    while current_cell != end:
        # Make legal step towards the target
        current_cell, current_direction = _next_legal_step(current_cell, current_direction, end)

        if current_cell is not None:
            path.append(current_cell)

    return path
