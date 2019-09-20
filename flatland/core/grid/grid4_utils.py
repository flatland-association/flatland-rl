from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import IntVector2D


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


def mirror(dir):
    return (dir + 2) % 4


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
