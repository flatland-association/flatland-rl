import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitionsEnum, RailEnvTransitions
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


@pytest.mark.parametrize(
    "elem, direction, expected_left, expected_forward,expected_right,expected_do_nothing",
    [pytest.param(*v, id=f"{v[0].name}")
     for v in [
         (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST,
          ((-1, 0), Grid4TransitionsEnum.NORTH),
          ((0, 1), Grid4TransitionsEnum.EAST),
          ((0, 1), Grid4TransitionsEnum.EAST),  # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST),
          ),

         (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST,
          ((0, -1), Grid4TransitionsEnum.WEST),  # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, -1), Grid4TransitionsEnum.WEST),
          ((0, -1), Grid4TransitionsEnum.WEST),
          ((0, -1), Grid4TransitionsEnum.WEST),
          ),

         (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST,
          ((0, 1), Grid4TransitionsEnum.EAST),  # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST),
          ((0, 1), Grid4TransitionsEnum.EAST),  # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST),
          ),
     ]]
)
def test_apply_action_independent(elem, direction, expected_left, expected_forward, expected_right, expected_do_nothing):
    rail = RailGridTransitionMap(1, 1, RailEnvTransitions())
    rail.set_transitions((0, 0), elem)

    print(rail.get_transitions(0, 0, direction))
    assert rail.apply_action_independent(RailEnvActions.MOVE_LEFT, (0, 0), direction) == expected_left
    assert rail.apply_action_independent(RailEnvActions.MOVE_FORWARD, (0, 0), direction) == expected_forward
    assert rail.apply_action_independent(RailEnvActions.MOVE_RIGHT, (0, 0), direction) == expected_right


def test_moving_action_straight():
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.horizontal_straight)

    assert rail.preprocess_left_right_action(RailEnvActions.DO_NOTHING, (1, 1), Grid4TransitionsEnum.EAST) == RailEnvActions.DO_NOTHING
    assert rail.preprocess_left_right_action(RailEnvActions.MOVE_LEFT, (1, 1), Grid4TransitionsEnum.EAST) == RailEnvActions.MOVE_FORWARD
    assert rail.preprocess_left_right_action(RailEnvActions.MOVE_FORWARD, (1, 1), Grid4TransitionsEnum.EAST) == RailEnvActions.MOVE_FORWARD
    assert rail.preprocess_left_right_action(RailEnvActions.MOVE_RIGHT, (1, 1), Grid4TransitionsEnum.EAST) == RailEnvActions.MOVE_FORWARD
    assert rail.preprocess_left_right_action(RailEnvActions.STOP_MOVING, (1, 1), Grid4TransitionsEnum.EAST) == RailEnvActions.STOP_MOVING


def test_moving_action_simple_switch():
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.simple_switch_north_right)
    rail.set_transitions((1, 2,), RailEnvTransitionsEnum.horizontal_straight)
    assert rail.preprocess_left_right_action(RailEnvActions.DO_NOTHING, (1, 1), Grid4TransitionsEnum.NORTH) == RailEnvActions.DO_NOTHING
    assert rail.preprocess_left_right_action(RailEnvActions.MOVE_LEFT, (1, 1), Grid4TransitionsEnum.NORTH) == RailEnvActions.MOVE_FORWARD
    assert rail.preprocess_left_right_action(RailEnvActions.MOVE_FORWARD, (1, 1), Grid4TransitionsEnum.NORTH) == RailEnvActions.MOVE_FORWARD
    assert rail.preprocess_left_right_action(RailEnvActions.MOVE_RIGHT, (1, 1), Grid4TransitionsEnum.NORTH) == RailEnvActions.MOVE_RIGHT
    assert rail.preprocess_left_right_action(RailEnvActions.STOP_MOVING, (1, 1), Grid4TransitionsEnum.NORTH) == RailEnvActions.STOP_MOVING
