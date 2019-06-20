import numpy as np
import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import position_to_coordinate, coordinate_to_position
from flatland.core.grid.grid4_utils import get_direction

depth_to_test = 5
positions_to_test = [0, 5, 1, 6, 20, 30]
coordinates_to_test = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 4], [0, 6]]


def test_position_to_coordinate():
    actual_coordinates = position_to_coordinate(depth_to_test, positions_to_test)
    expected_coordinates = coordinates_to_test
    assert np.array_equal(actual_coordinates, expected_coordinates), \
        "converted positions {}, expected {}".format(actual_coordinates, expected_coordinates)


def test_coordinate_to_position():
    actual_positions = coordinate_to_position(depth_to_test, coordinates_to_test)
    expected_positions = positions_to_test
    assert np.array_equal(actual_positions, expected_positions), \
        "converted positions {}, expected {}".format(actual_positions, expected_positions)


def test_get_direction():
    assert get_direction((0, 0), (0, 1)) == Grid4TransitionsEnum.EAST
    assert get_direction((0, 0), (0, 2)) == Grid4TransitionsEnum.EAST
    assert get_direction((0, 0), (1, 0)) == Grid4TransitionsEnum.SOUTH
    assert get_direction((1, 0), (0, 0)) == Grid4TransitionsEnum.NORTH
    assert get_direction((1, 0), (0, 0)) == Grid4TransitionsEnum.NORTH
    with pytest.raises(Exception, match="Could not determine direction"):
        get_direction((0, 0), (0, 0)) == Grid4TransitionsEnum.NORTH
