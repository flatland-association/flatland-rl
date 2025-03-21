import pytest

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitionsEnum, RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.env_utils import apply_action_independent


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
    rail = GridTransitionMap(1, 1, RailEnvTransitions())
    rail.set_transitions((0, 0), elem)

    print(rail.get_transitions(0, 0, direction))
    assert apply_action_independent(RailEnvActions.MOVE_LEFT, rail, (0, 0), direction) == expected_left
    assert apply_action_independent(RailEnvActions.MOVE_FORWARD, rail, (0, 0), direction) == expected_forward
    assert apply_action_independent(RailEnvActions.MOVE_RIGHT, rail, (0, 0), direction) == expected_right
