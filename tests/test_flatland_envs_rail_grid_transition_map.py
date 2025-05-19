import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitionsEnum, RailEnvTransitions
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


@pytest.mark.parametrize(
    "elem, direction, expected_left, expected_forward,expected_right,expected_do_nothing",
    [pytest.param(*v, id=f"{v[0].name}")
     for v in [
         # switch left
         (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.EAST,
          ((-1, 0), Grid4TransitionsEnum.NORTH, True, RailEnvActions.MOVE_LEFT),
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.MOVE_FORWARD),
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.MOVE_FORWARD),
          # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.MOVE_FORWARD),
          ),

         (RailEnvTransitionsEnum.simple_switch_east_left, Grid4TransitionsEnum.WEST,
          ((0, -1), Grid4TransitionsEnum.WEST, True, RailEnvActions.MOVE_FORWARD),
          # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, -1), Grid4TransitionsEnum.WEST, True, RailEnvActions.MOVE_FORWARD),
          ((0, -1), Grid4TransitionsEnum.WEST, True, RailEnvActions.MOVE_FORWARD),
          ((0, -1), Grid4TransitionsEnum.WEST, True, RailEnvActions.MOVE_FORWARD),
          ),
         # dead-end
         (RailEnvTransitionsEnum.dead_end_from_east, Grid4TransitionsEnum.WEST,
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.MOVE_FORWARD),
          # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.MOVE_FORWARD),
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.MOVE_FORWARD),
          # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.MOVE_FORWARD),
          ),
         # straight
         (RailEnvTransitionsEnum.horizontal_straight, Grid4TransitionsEnum.EAST,
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.MOVE_FORWARD),
          # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.MOVE_FORWARD),
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.MOVE_FORWARD),
          # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST, True, RailEnvActions.DO_NOTHING),  # do not accelerate!
          ),
         # symmetric
         (RailEnvTransitionsEnum.symmetric_switch_from_west, Grid4TransitionsEnum.EAST,
          ((-1, 0), Grid4TransitionsEnum.NORTH, True, RailEnvActions.MOVE_LEFT),  # TODO cell check valid?
          # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST, False, RailEnvActions.STOP_MOVING),
          ((1, 0), Grid4TransitionsEnum.SOUTH, True, RailEnvActions.MOVE_RIGHT),
          # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST, False, RailEnvActions.STOP_MOVING),
          ),
         # symmetric switch
         (RailEnvTransitionsEnum.right_turn_from_west, Grid4TransitionsEnum.EAST,
          ((1, 0), Grid4TransitionsEnum.SOUTH, True, RailEnvActions.MOVE_FORWARD),  # TODO cell check valid? MOVE_FORWARD?
          # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((1, 0), Grid4TransitionsEnum.SOUTH, True, RailEnvActions.MOVE_FORWARD),  # TODO cell check valid? MOVE_FORWARD?
          ((1, 0), Grid4TransitionsEnum.SOUTH, True, RailEnvActions.MOVE_RIGHT),
          # TODO https://github.com/flatland-association/flatland-rl/issues/185 streamline?
          ((0, 1), Grid4TransitionsEnum.EAST, False, RailEnvActions.DO_NOTHING), # TODO cell check valid? DO_NOTHING?
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
