from functools import lru_cache
from typing import Set, TYPE_CHECKING

import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import IntVector2D

if TYPE_CHECKING:
    from flatland.core.transition_map import GridTransitionMap


def get_direction(pos1: IntVector2D, pos2: IntVector2D) -> Grid4TransitionsEnum:
    """
    Assumes pos1 and pos2 are adjacent location on grid.
    Returns direction (int) that can be used with transitions.
    """
    diff_0 = pos2[0] - pos1[0]
    diff_1 = pos2[1] - pos1[1]
    if diff_0 < 0:
        return Grid4TransitionsEnum.NORTH
    if diff_0 > 0:
        return Grid4TransitionsEnum.SOUTH
    if diff_1 > 0:
        return Grid4TransitionsEnum.EAST
    if diff_1 < 0:
        return Grid4TransitionsEnum.WEST
    raise Exception("Could not determine direction {}->{}".format(pos1, pos2))


def is_neighbor_cell(pos1: IntVector2D, pos2: IntVector2D) -> bool:
    """
    Check whether pos1 and pos2 are adjacent to each other, top/bottom or left/right, no diagonal.
    """
    diff_0 = pos2[0] - pos1[0]
    diff_1 = pos2[1] - pos1[1]
    return abs(diff_0) + abs(diff_1) == 1


@lru_cache(maxsize=4)
def mirror(dir):
    return (dir + 2) % 4


MOVEMENT_ARRAY = [(-1, 0), (0, 1), (1, 0), (0, -1)]


@lru_cache(maxsize=1_000_000)
def get_new_position(position, movement):
    """
    Get new (r,c) when exiting in direction movement.
    """
    m = MOVEMENT_ARRAY[movement]
    return (position[0] + m[0], position[1] + m[1])


@lru_cache(maxsize=1_000_000)
def get_old_position(position, movement):
    """
    Get old (r,c) when entering in direction movement.
    """
    m = MOVEMENT_ARRAY[mirror(movement)]
    return (position[0] + m[0], position[1] + m[1])


def direction_to_point(pos1: IntVector2D, pos2: IntVector2D) -> Grid4TransitionsEnum:
    """
    Returns the closest direction orientation of position 2 relative to position 1
    :param pos1: position we are interested in
    :param pos2: position we want to know it is facing
    :return: direction NESW as int N:0 E:1 S:2 W:3
    """
    diff_vec = np.array((pos1[0] - pos2[0], pos1[1] - pos2[1]))
    axis = np.argmax(np.power(diff_vec, 2))
    direction = np.sign(diff_vec[axis])
    if axis == 0:
        if direction > 0:
            return Grid4TransitionsEnum.NORTH
        else:
            return Grid4TransitionsEnum.SOUTH
    else:
        if direction > 0:
            return Grid4TransitionsEnum.WEST
        else:
            return Grid4TransitionsEnum.EAST


def find_connected_cells(grid_map: "GridTransitionMap", open_set: Set[IntVector2D],
                         forbidden_cells: Set[IntVector2D] = None) -> Set[IntVector2D]:
    """
    Flood-fill from a set of starting cells (open set) to find all cells connected to them in the
    grid via valid transitions, without passing through forbidden_cells.

    Parameters
    ----------
    grid_map : GridTransitionMap
        Grid Map to search in.
    open_set : Set[IntVector2D]
        Starting cells (row,column) to search from. Always included in the result, even if also
        listed in `forbidden_cells`.
    forbidden_cells : Optional[Set[IntVector2D]]
        Set of cells the search must not pass through. Used to avoid certain areas of the Grid map.

    Returns
    -------
    Set[IntVector2D]
        Set of all cells (row,column) connected to the open set via valid transitions, excluding
        any `forbidden_cells` other than the starting cells themselves.
    """
    if forbidden_cells is None:
        forbidden_cells = set()
    closed_set: Set[IntVector2D] = set(open_set)
    open_list = list(open_set)
    while len(open_list) > 0:
        current_pos = open_list.pop()
        for from_cell, to_cell in grid_map.get_neighbor_pairs(current_pos):
            for neighbor_pos in (from_cell, to_cell):
                if neighbor_pos in closed_set:
                    continue
                if neighbor_pos in forbidden_cells:
                    continue
                if not grid_map.check_bounds(neighbor_pos):
                    continue
                closed_set.add(neighbor_pos)
                open_list.append(neighbor_pos)
    return closed_set
