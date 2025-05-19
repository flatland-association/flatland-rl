import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitionsEnum, RailEnvTransitions
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


@pytest.mark.parametrize(
    "elem, direction, expected_left, expected_forward,expected_right,expected_do_nothing",
    [pytest.param(*v, id=f"{v[0].name}")
     for v in [
         # switch left facing
         (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST,
          (False, Grid4TransitionsEnum.NORTH, (-1, 0), True, RailEnvActions.MOVE_LEFT),
          (False, Grid4TransitionsEnum.EAST, (0, 1), True, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.EAST, (0, 1), False, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.EAST, (0, 1), True, RailEnvActions.DO_NOTHING),
          ),
         # switch left non-facing
         (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST,
          (False, Grid4TransitionsEnum.WEST, (0, -1), False, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.WEST, (0, -1), True, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.WEST, (0, -1), False, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.WEST, (0, -1), True, RailEnvActions.DO_NOTHING),
          ),
         # dead-end
         (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST,
          (False, Grid4TransitionsEnum.EAST, (0, 1), False, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.EAST, (0, 1), True, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.EAST, (0, 1), False, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.EAST, (0, 1), True, RailEnvActions.DO_NOTHING),
          ),
         # straight
         (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST,
          (False, Grid4TransitionsEnum.EAST, (0, 1), False, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.EAST, (0, 1), True, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.EAST, (0, 1), False, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.EAST, (0, 1), True, RailEnvActions.DO_NOTHING),
          ),
         # symmetric switch facing
         (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST,
          (False, Grid4TransitionsEnum.NORTH, (-1, 0), True, RailEnvActions.MOVE_LEFT),
          (False, Grid4TransitionsEnum.EAST, (0, 1), False, RailEnvActions.STOP_MOVING),
          (False, Grid4TransitionsEnum.SOUTH, (1, 0), True, RailEnvActions.MOVE_RIGHT),
          (False, Grid4TransitionsEnum.EAST, (0, 1), False, RailEnvActions.STOP_MOVING),
          ),
         # symmetric switch non-facing (same as right-turn)
         (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.SOUTH,
          (False, Grid4TransitionsEnum.WEST, (0, -1), False, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.WEST, (0, -1), True, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.WEST, (0, -1), True, RailEnvActions.MOVE_RIGHT),
          (False, Grid4TransitionsEnum.WEST, (0, -1), True, RailEnvActions.DO_NOTHING),
          ),
         # right turn (both forward and right are valid transitions)
         (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST,
          (False, Grid4TransitionsEnum.SOUTH, (1, 0), False, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.SOUTH, (1, 0), True, RailEnvActions.MOVE_FORWARD),
          (False, Grid4TransitionsEnum.SOUTH, (1, 0), True, RailEnvActions.MOVE_RIGHT),
          (False, Grid4TransitionsEnum.SOUTH, (1, 0), True, RailEnvActions.DO_NOTHING),
          ),
     ]]
)
def test_check_action_on_agent(elem, direction, expected_left, expected_forward, expected_right, expected_do_nothing):
    rail = RailGridTransitionMap(1, 1, RailEnvTransitions())
    rail.set_transitions((0, 0), elem)

    print(rail.get_transitions(0, 0, direction))
    assert rail.check_action_on_agent(RailEnvActions.MOVE_LEFT, (0, 0), direction) == expected_left
    assert rail.check_action_on_agent(RailEnvActions.MOVE_FORWARD, (0, 0), direction) == expected_forward
    assert rail.check_action_on_agent(RailEnvActions.MOVE_RIGHT, (0, 0), direction) == expected_right
    assert rail.check_action_on_agent(RailEnvActions.DO_NOTHING, (0, 0), direction) == expected_do_nothing


def test_check_action_on_agent_horizontal_straight():
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((1, 2), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((0, 1), RailEnvTransitionsEnum.vertical_straight)

    new_cell_valid, new_direction, new_position, transition_valid, _ = rail.check_action_on_agent(RailEnvActions.MOVE_FORWARD, (1, 1),
                                                                                                  Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert transition_valid

    new_cell_valid, new_direction, new_position, transition_valid, _ = rail.check_action_on_agent(RailEnvActions.MOVE_LEFT, (1, 1), Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert not transition_valid


def test_check_action_on_agent_symmetric_switch_from_west():
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.symmetric_switch_from_west)
    rail.set_transitions((0, 1), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((1, 2), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((2, 1), RailEnvTransitionsEnum.vertical_straight)

    new_cell_valid, new_direction, new_position, transition_valid, _ = rail.check_action_on_agent(RailEnvActions.MOVE_RIGHT, (1, 1), Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.SOUTH
    assert new_position == (2, 1)
    assert transition_valid

    new_cell_valid, new_direction, new_position, transition_valid, _ = rail.check_action_on_agent(RailEnvActions.MOVE_FORWARD, (1, 1),
                                                                                                  Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert not transition_valid

    new_cell_valid, new_direction, new_position, transition_valid, _ = rail.check_action_on_agent(RailEnvActions.MOVE_LEFT, (1, 1), Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.NORTH
    assert new_position == (0, 1)
    assert transition_valid

    new_cell_valid, new_direction, new_position, transition_valid, _ = rail.check_action_on_agent(RailEnvActions.DO_NOTHING, (1, 1), Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert not transition_valid
