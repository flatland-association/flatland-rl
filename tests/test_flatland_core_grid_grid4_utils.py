import numpy as np
import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_direction, find_connected_cells
from flatland.core.grid.grid_utils import position_to_coordinate, coordinate_to_position
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum
from flatland.envs.rail_env_utils import load_flatland_environment_from_file

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
        get_direction((0, 0), (0, 0))


def test_load():
    load_flatland_environment_from_file('test_001.pkl', 'env_data.tests')


def _make_horizontal_line(width=7, height=4, row=1, start_col=1, end_col=5):
    rail = GridTransitionMap(width, height)
    for c in range(start_col, end_col + 1):
        rail.set_transitions((row, c), RailEnvTransitionsEnum.horizontal_straight)
    return rail


def test_find_connected_cells():
    rail = _make_horizontal_line()
    # disconnected segment elsewhere in the grid - must not be picked up
    rail.set_transitions((3, 1), RailEnvTransitionsEnum.horizontal_straight)

    connected = find_connected_cells(rail, open_set={(1, 3)})

    assert connected == {(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)}


def test_find_connected_cells_multiple_open_set_cells():
    rail = _make_horizontal_line()
    rail.set_transitions((3, 1), RailEnvTransitionsEnum.horizontal_straight)

    connected = find_connected_cells(rail, open_set={(1, 3), (3, 1)})

    assert connected == {(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                         (3, 0), (3, 1), (3, 2)}


def test_find_connected_cells_respects_forbidden_cells():
    rail = _make_horizontal_line()

    # forbidding (1, 2) must cut off everything to its left
    connected = find_connected_cells(rail, open_set={(1, 3)}, forbidden_cells={(1, 2)})

    assert connected == {(1, 3), (1, 4), (1, 5), (1, 6)}


def test_find_connected_cells_open_set_cell_always_included_even_if_forbidden():
    rail = _make_horizontal_line()

    # a cell in open_set must be included (and still explored) even if also forbidden
    connected = find_connected_cells(rail, open_set={(1, 3)}, forbidden_cells={(1, 3)})

    assert (1, 3) in connected
    assert connected == {(1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)}
