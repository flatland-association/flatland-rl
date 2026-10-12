import numpy as np
import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum, RailEnvTransitions
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


# TODO this test might be flawed: check_action_on_agent takes the current cell and not the cell to be entered!
@pytest.mark.parametrize(
    "elem, direction, expected_left, expected_forward,expected_right,expected_do_nothing",
    [pytest.param(*v, id=f"{v[0].name}")
     for v in [
         # switch left facing
         (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST,
          (False, ((-1, 0), Grid4TransitionsEnum.NORTH), True, RailEnvActions.MOVE_LEFT, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), True, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), False, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), True, RailEnvActions.DO_NOTHING, True),
          ),
         # switch left non-facing
         (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST,
          (False, ((0, -1), Grid4TransitionsEnum.WEST), False, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, -1), Grid4TransitionsEnum.WEST), True, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, -1), Grid4TransitionsEnum.WEST), False, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, -1), Grid4TransitionsEnum.WEST), True, RailEnvActions.DO_NOTHING, True),
          ),
         # dead-end
         (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST,
          (False, ((0, 1), Grid4TransitionsEnum.EAST), False, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), True, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), False, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), True, RailEnvActions.DO_NOTHING, True),
          ),
         # straight
         (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST,
          (False, ((0, 1), Grid4TransitionsEnum.EAST), False, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), True, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), False, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), True, RailEnvActions.DO_NOTHING, True),
          ),
         # symmetric switch facing
         (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST,
          (False, ((-1, 0), Grid4TransitionsEnum.NORTH), True, RailEnvActions.MOVE_LEFT, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), False, RailEnvActions.STOP_MOVING, False),
          (False, ((1, 0), Grid4TransitionsEnum.SOUTH), True, RailEnvActions.MOVE_RIGHT, True),
          (False, ((0, 1), Grid4TransitionsEnum.EAST), False, RailEnvActions.STOP_MOVING, False),
          ),
         # symmetric switch non-facing (same as right-turn)
         (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH,
          (False, ((0, -1), Grid4TransitionsEnum.WEST), False, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, -1), Grid4TransitionsEnum.WEST), True, RailEnvActions.MOVE_FORWARD, True),
          (False, ((0, -1), Grid4TransitionsEnum.WEST), True, RailEnvActions.MOVE_RIGHT, True),
          (False, ((0, -1), Grid4TransitionsEnum.WEST), True, RailEnvActions.DO_NOTHING, True),
          ),
         # right turn (both forward and right are valid transitions)
         (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST,
          (False, ((1, 0), Grid4TransitionsEnum.SOUTH), False, RailEnvActions.MOVE_FORWARD, True),
          (False, ((1, 0), Grid4TransitionsEnum.SOUTH), True, RailEnvActions.MOVE_FORWARD, True),
          (False, ((1, 0), Grid4TransitionsEnum.SOUTH), True, RailEnvActions.MOVE_RIGHT, True),
          (False, ((1, 0), Grid4TransitionsEnum.SOUTH), True, RailEnvActions.DO_NOTHING, True),
          ),
     ]]
)
def test_check_action_on_agent(elem, direction, expected_left, expected_forward, expected_right, expected_do_nothing):
    rail = RailGridTransitionMap(1, 1, RailEnvTransitions())
    rail.set_transitions((0, 0), elem)

    print(rail.get_transitions((((0, 0), direction))))
    assert rail._check_action_on_agent(RailEnvActions.MOVE_LEFT, ((0, 0), direction)) == expected_left
    assert rail._check_action_on_agent(RailEnvActions.MOVE_FORWARD, ((0, 0), direction)) == expected_forward
    assert rail._check_action_on_agent(RailEnvActions.MOVE_RIGHT, ((0, 0), direction)) == expected_right
    assert rail._check_action_on_agent(RailEnvActions.DO_NOTHING, ((0, 0), direction)) == expected_do_nothing


# TODO this test might be flawed: check_action_on_agent takes the current cell and not the cell to be entered!
def test_check_action_on_agent_horizontal_straight():
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((1, 2), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((0, 1), RailEnvTransitionsEnum.vertical_straight)

    new_cell_valid, (new_position, new_direction), transition_valid, _, _ = rail._check_action_on_agent(
        RailEnvActions.MOVE_FORWARD, ((1, 1), Grid4TransitionsEnum.EAST))
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert transition_valid

    new_cell_valid, (new_position, new_direction), transition_valid, _, _ = rail._check_action_on_agent(
        RailEnvActions.MOVE_LEFT, ((1, 1), Grid4TransitionsEnum.EAST))
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert not transition_valid


# TODO this test might be flawed: check_action_on_agent takes the current cell and not the cell to be entered!
def test_check_action_on_agent_symmetric_switch_from_west():
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.symmetric_switch_from_west)
    rail.set_transitions((0, 1), RailEnvTransitionsEnum.vertical_straight)
    rail.set_transitions((1, 2), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((2, 1), RailEnvTransitionsEnum.vertical_straight)

    new_cell_valid, (new_position, new_direction), transition_valid, _, _ = rail._check_action_on_agent(
        RailEnvActions.MOVE_RIGHT, ((1, 1), Grid4TransitionsEnum.EAST))
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.SOUTH
    assert new_position == (2, 1)
    assert transition_valid

    new_cell_valid, (new_position, new_direction), transition_valid, _, _ = rail._check_action_on_agent(
        RailEnvActions.MOVE_FORWARD, ((1, 1), Grid4TransitionsEnum.EAST))
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert not transition_valid

    new_cell_valid, (new_position, new_direction), transition_valid, _, _ = rail._check_action_on_agent(
        RailEnvActions.MOVE_LEFT, ((1, 1), Grid4TransitionsEnum.EAST))
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.NORTH
    assert new_position == (0, 1)
    assert transition_valid

    new_cell_valid, (new_position, new_direction), transition_valid, _, _ = rail._check_action_on_agent(
        RailEnvActions.DO_NOTHING, ((1, 1), Grid4TransitionsEnum.EAST))
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert not transition_valid


@pytest.mark.parametrize(
    "elem, direction, action, expected", [
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, None),
        (RailEnvTransitionsEnum.dead_end_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.dead_end_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.dead_end_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.dead_end_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.dead_end_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.dead_end_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.dead_end_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.dead_end_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.dead_end_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.dead_end_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.dead_end_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.dead_end_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.dead_end_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.dead_end_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.dead_end_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((2, 1), 2)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, ((1, 2), 1)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, ((1, 0), 3)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, ((0, 1), 0)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, ((1, 0), 3)),
    ]
)
def test_action_independent(elem, direction, action, expected, regenerate=False):
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((0, 1), RailEnvTransitionsEnum.dead_end_from_south)
    rail.set_transitions((1, 2), RailEnvTransitionsEnum.dead_end_from_west)
    rail.set_transitions((2, 1), RailEnvTransitionsEnum.dead_end_from_north)
    rail.set_transitions((1, 0), RailEnvTransitionsEnum.dead_end_from_east)
    rail.set_transitions((1, 1), elem)
    assert rail.apply_action_independent(action, ((1, 1), direction)) == expected

    if regenerate:
        for elem in RailEnvTransitionsEnum:

            rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
            rail.set_transitions((0, 1), RailEnvTransitionsEnum.dead_end_from_south)
            rail.set_transitions((1, 2), RailEnvTransitionsEnum.dead_end_from_west)
            rail.set_transitions((2, 1), RailEnvTransitionsEnum.dead_end_from_north)
            rail.set_transitions((1, 0), RailEnvTransitionsEnum.dead_end_from_east)
            rail.set_transitions((1, 1), elem)
            for d in range(4):
                if not any(rail.get_transitions(((1, 1), d))):
                    continue
                for a in RailEnvActions:
                    t = rail.apply_action_independent(a, ((1, 1), d))
                    print(f"(RailEnvTransitionsEnum.{elem.name},Grid4TransitionsEnum.{Grid4TransitionsEnum(d).name},{a},{t}),")


@pytest.mark.parametrize(
    "elem, expected",
    [pytest.param(*v, id=f"{v[0].name}")
     for v in [
         (RailEnvTransitionsEnum.simple_switch_east_left, [False, True, True, True]),
         (RailEnvTransitionsEnum.dead_end_from_east, [False, False, False, True]),
         (RailEnvTransitionsEnum.horizontal_straight, [False, True, False, True]),
         (RailEnvTransitionsEnum.symmetric_switch_from_west, [True, True, True, False]),
         (RailEnvTransitionsEnum.right_turn_from_west, [True, True, False, False]),
     ]]
)
def test_get_valid_directions_on_grid(elem, expected):
    rail = RailGridTransitionMap(1, 1, RailEnvTransitions())
    rail.set_transitions((0, 0), elem)
    assert rail.get_valid_directions_on_grid(0, 0) == expected


@pytest.mark.parametrize(
    "entry_point, expected",
    [
        pytest.param(((0, 0), Grid4TransitionsEnum.EAST), True, id="in_bounds_valid_transition"),
        pytest.param(((0, 0), Grid4TransitionsEnum.NORTH), False, id="in_bounds_no_transition"),
        pytest.param(((-1, 0), Grid4TransitionsEnum.EAST), False, id="out_of_bounds_negative_row"),
        pytest.param(((0, -1), Grid4TransitionsEnum.EAST), False, id="out_of_bounds_negative_column"),
        pytest.param(((3, 0), Grid4TransitionsEnum.EAST), False, id="out_of_bounds_row_at_height"),
        pytest.param(((0, 3), Grid4TransitionsEnum.EAST), False, id="out_of_bounds_column_at_width"),
    ]
)
def test_is_valid_entry_point_out_of_bounds(entry_point, expected):
    """Regression test: is_valid_entry_point must reject entry_points whose position falls
    outside the grid, not just cells with no outgoing transitions."""
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((0, 0), RailEnvTransitionsEnum.horizontal_straight)
    assert rail.is_valid_entry_point(entry_point) == expected


_NON_EMPTY_RAIL_ENV_TRANSITIONS = [t for t in RailEnvTransitionsEnum if t != RailEnvTransitionsEnum.empty]

# at a symmetric switch, entering head-on (facing the fork) makes MOVE_FORWARD invalid by design -- only
# MOVE_LEFT/MOVE_RIGHT are valid there (see test_apply_action_independent_only_left_right_valid_at_symmetric_switch).
_SYMMETRIC_SWITCH_FACING_ENTRIES = {
    (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST),
    (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH),
    (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST),
    (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH),
}


@pytest.mark.parametrize("rail_env_transition", _NON_EMPTY_RAIL_ENV_TRANSITIONS,
                         ids=[t.name for t in _NON_EMPTY_RAIL_ENV_TRANSITIONS])
@pytest.mark.parametrize("direction", list(Grid4TransitionsEnum), ids=[d.name for d in Grid4TransitionsEnum])
def test_apply_action_independent_not_none_for_every_entry_side(rail_env_transition, direction):
    """Show that all actions are valid except for L/R on symmetric switches facing."""
    if (rail_env_transition, direction) in _SYMMETRIC_SWITCH_FACING_ENTRIES:
        pytest.skip(f"{rail_env_transition.name} facing {direction.name} disallows MOVE_FORWARD by design")

    transitions = RailEnvTransitions()
    center = (1, 1)
    grid = np.zeros((3, 3), dtype=np.uint16)
    grid[center] = rail_env_transition
    for d in Grid4TransitionsEnum:
        if not any(transitions.get_transitions(int(rail_env_transition), d)):
            continue  # element has no transition for this incoming orientation -- no neighbor to connect
        previous_cell = get_new_position(center, (d + 2) % 4)
        is_vertical = d in (Grid4TransitionsEnum.NORTH, Grid4TransitionsEnum.SOUTH)
        sender = RailEnvTransitionsEnum.vertical_straight if is_vertical else RailEnvTransitionsEnum.horizontal_straight
        grid[previous_cell] = sender

    rail = RailGridTransitionMap(width=3, height=3, transitions=transitions)
    rail.grid = grid

    # design: actions applied at cell entry -- like `apply_action_independent(action, agent.next_entry_point)`,
    # derive the entry point at center by applying MOVE_FORWARD from an agent standing ahead of center (the
    # predecessor cell), instead of hand-constructing (center, direction).
    ahead_of_center = (get_new_position(center, (direction + 2) % 4), direction)
    lookahead = rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, ahead_of_center)
    if lookahead is None:
        pytest.skip(f"{rail_env_transition.name} has no valid entry from {direction.name}")
    entry_point = lookahead

    assert all(rail.apply_action_independent(action, entry_point) is not None
               for action in RailEnvActions)


@pytest.mark.parametrize("action", list(RailEnvActions), ids=[a.name for a in RailEnvActions])
def test_apply_action_independent_only_left_right_valid_at_symmetric_switch(action):
    """Document invalid actions L/R for symmetric switch explicitly."""
    transitions = RailEnvTransitions()
    center = (1, 1)
    grid = np.zeros((3, 3), dtype=np.uint16)
    grid[center] = RailEnvTransitionsEnum.symmetric_switch_from_east  # heading west forks N/S
    grid[0, 1] = RailEnvTransitionsEnum.vertical_straight
    grid[2, 1] = RailEnvTransitionsEnum.vertical_straight
    grid[1, 2] = RailEnvTransitionsEnum.horizontal_straight  # ahead of center, agent standing here facing west

    rail = RailGridTransitionMap(width=3, height=3, transitions=transitions)
    rail.grid = grid

    # design: actions applied at cell entry -- like `apply_action_independent(action, agent.next_entry_point)`,
    # derive the entry point at center by applying MOVE_FORWARD from an agent standing ahead of center.
    entry_point = rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, ((1, 2), Grid4TransitionsEnum.WEST))
    result = rail.apply_action_independent(action, entry_point)
    if action in (RailEnvActions.MOVE_LEFT, RailEnvActions.MOVE_RIGHT):
        assert result is not None
    else:
        assert result is None
