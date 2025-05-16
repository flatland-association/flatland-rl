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
    assert new_direction == Grid4TransitionsEnum.NORTH
    assert new_position == (0, 1)
    assert not transition_valid


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


def test_check_valid_action_symmetric_switch():
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.symmetric_switch_from_north)
    rail.set_transitions((1, 0), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((1, 2), RailEnvTransitionsEnum.horizontal_straight)

    assert not rail.check_valid_action(RailEnvActions.MOVE_FORWARD, (1, 1), Grid4TransitionsEnum.SOUTH)
    assert not rail.check_valid_action(RailEnvActions.DO_NOTHING, (1, 1), Grid4TransitionsEnum.SOUTH)
    assert rail.check_valid_action(RailEnvActions.MOVE_LEFT, (1, 1), Grid4TransitionsEnum.SOUTH)
    assert rail.check_valid_action(RailEnvActions.MOVE_RIGHT, (1, 1), Grid4TransitionsEnum.SOUTH)


def test_check_valid_action_dead_end():
    rail = RailGridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.dead_end_from_west)
    rail.set_transitions((1, 0), RailEnvTransitionsEnum.horizontal_straight)
    rail.set_transitions((1, 2), RailEnvTransitionsEnum.horizontal_straight)

    assert rail.check_valid_action(RailEnvActions.MOVE_FORWARD, (1, 1), Grid4TransitionsEnum.EAST)
    assert rail.check_valid_action(RailEnvActions.DO_NOTHING, (1, 1), Grid4TransitionsEnum.EAST)
    assert not rail.check_valid_action(RailEnvActions.MOVE_LEFT, (1, 1), Grid4TransitionsEnum.EAST)
    assert not rail.check_valid_action(RailEnvActions.MOVE_RIGHT, (1, 1), Grid4TransitionsEnum.EAST)
