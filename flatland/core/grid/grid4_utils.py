import numpy as np

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

MOVEMENT_ARRAY = [(-1, 0), (0, 1), (1, 0), (0, -1)]
def get_new_position(position, movement):
	return (position[0] + MOVEMENT_ARRAY[movement][0], position[1] + MOVEMENT_ARRAY[movement][1])


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
