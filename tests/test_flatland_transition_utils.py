from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


def test_check_action_on_agent_horizontal_straight():
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((1, 2), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((0, 1), RailEnvTransitionsEnum.vertical_straight)

    new_cell_valid, new_direction, new_position, transition_valid = rail.check_action_on_agent(RailEnvActions.MOVE_FORWARD, (1, 1), Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert transition_valid

    new_cell_valid, new_direction, new_position, transition_valid = rail.check_action_on_agent(RailEnvActions.MOVE_LEFT, (1, 1), Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    # TODO should this be valid?
    assert transition_valid


def test_check_action_on_agent_symmetric_switch_from_west():
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.symmetric_switch_from_west)
    rail.set_transitions((0, 1), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((1, 2), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((2, 1), RailEnvTransitionsEnum.vertical_straight)

    new_cell_valid, new_direction, new_position, transition_valid = rail.check_action_on_agent(RailEnvActions.MOVE_RIGHT, (1, 1), Grid4TransitionsEnum.EAST)
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.SOUTH
    assert new_position == (2, 1)
    assert transition_valid

    new_cell_valid, new_direction, new_position, transition_valid = rail.check_action_on_agent(RailEnvActions.MOVE_FORWARD, (1, 1), Grid4TransitionsEnum.EAST)
    # TODO https://github.com/flatland-association/flatland-rl/issues/185 new_cell_valid checks only whether within bounds and not whether the new direction is possible - clean up!
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert not transition_valid

    new_cell_valid, new_direction, new_position, transition_valid = rail.check_action_on_agent(RailEnvActions.MOVE_LEFT, (1, 1), Grid4TransitionsEnum.EAST)
    # TODO https://github.com/flatland-association/flatland-rl/issues/185 new_cell_valid checks only whether within bounds and not whether the new direction is possible - clean up!
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.NORTH
    assert new_position == (0, 1)
    assert transition_valid

    new_cell_valid, new_direction, new_position, transition_valid = rail.check_action_on_agent(RailEnvActions.DO_NOTHING, (1, 1), Grid4TransitionsEnum.EAST)
    # TODO https://github.com/flatland-association/flatland-rl/issues/185 new_cell_valid checks only whether within bounds and not whether the new direction is possible - clean up!
    assert new_cell_valid
    assert new_direction == Grid4TransitionsEnum.EAST
    assert new_position == (1, 2)
    assert not transition_valid
