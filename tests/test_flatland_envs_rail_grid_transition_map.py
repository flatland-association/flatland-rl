import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.grid.rail_env_grid import RailEnvTransitionsEnum, RailEnvTransitions
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


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
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.vertical_straight, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_north_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_south_left, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_west_left, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.diamond_crossing, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_SW, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_NW, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.single_slip_NE, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.single_slip_SE, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.double_slip_NW_SE, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.double_slip_NE_SW, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_north, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, None),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.symmetric_switch_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, None),
        (RailEnvTransitionsEnum.dead_end_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.dead_end_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.dead_end_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.dead_end_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.dead_end_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.dead_end_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.dead_end_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.dead_end_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.dead_end_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.dead_end_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.dead_end_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.dead_end_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.dead_end_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.dead_end_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.dead_end_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.right_turn_from_south, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.right_turn_from_north, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.right_turn_from_east, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_north_right, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((2, 1), 2), False)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_east_right, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.NORTH, RailEnvActions.DO_NOTHING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_LEFT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.NORTH, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.NORTH, RailEnvActions.STOP_MOVING, (((0, 1), 0), True)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((1, 0), 3), False)),
        (RailEnvTransitionsEnum.simple_switch_south_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((2, 1), 2), True)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.EAST, RailEnvActions.DO_NOTHING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_LEFT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.EAST, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.EAST, RailEnvActions.STOP_MOVING, (((1, 2), 1), True)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.DO_NOTHING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_LEFT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_FORWARD, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.MOVE_RIGHT, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.SOUTH, RailEnvActions.STOP_MOVING, (((1, 2), 1), False)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.WEST, RailEnvActions.DO_NOTHING, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_LEFT, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_FORWARD, (((1, 0), 3), True)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.WEST, RailEnvActions.MOVE_RIGHT, (((0, 1), 0), False)),
        (RailEnvTransitionsEnum.simple_switch_west_right, Grid4TransitionsEnum.WEST, RailEnvActions.STOP_MOVING, (((1, 0), 3), True)),
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
