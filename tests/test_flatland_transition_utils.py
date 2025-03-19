from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.transition_utils import check_action_on_agent


def test_check_action_on_agent():
    rail = GridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((1, 2), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((0, 1), RailEnvTransitionsEnum.vertical_straight)

    new_cell_valid, new_direction, new_position, transition_valid = check_action_on_agent(RailEnvActions.MOVE_FORWARD, rail, (1, 1), Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert transition_valid

    new_cell_valid, new_direction, new_position, transition_valid = check_action_on_agent(RailEnvActions.MOVE_LEFT, rail, (1, 1), Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.NORTH
    assert new_position == (0, 1)
    assert not transition_valid
