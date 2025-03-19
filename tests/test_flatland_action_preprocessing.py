from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.action_preprocessing import process_illegal_action, preprocess_left_right_action


def test_process_illegal_action():
    assert process_illegal_action(None) == RailEnvActions.DO_NOTHING
    assert process_illegal_action(0) == RailEnvActions.DO_NOTHING
    assert process_illegal_action(RailEnvActions.DO_NOTHING) == RailEnvActions.DO_NOTHING
    assert process_illegal_action("Alice") == RailEnvActions.DO_NOTHING
    assert process_illegal_action("MOVE_LEFT") == RailEnvActions.DO_NOTHING
    assert process_illegal_action(1) == RailEnvActions.MOVE_LEFT


def test_moving_action():
    rail = GridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.horizontal_straight)

    assert preprocess_left_right_action(RailEnvActions.DO_NOTHING, rail, (1, 1), Grid4TransitionsEnum.EAST) == RailEnvActions.DO_NOTHING
    assert preprocess_left_right_action(RailEnvActions.MOVE_LEFT, rail, (1, 1), Grid4TransitionsEnum.EAST) == RailEnvActions.MOVE_FORWARD
    assert preprocess_left_right_action(RailEnvActions.MOVE_FORWARD, rail, (1, 1), Grid4TransitionsEnum.EAST) == RailEnvActions.MOVE_FORWARD
    assert preprocess_left_right_action(RailEnvActions.MOVE_RIGHT, rail, (1, 1), Grid4TransitionsEnum.EAST) == RailEnvActions.MOVE_FORWARD
    assert preprocess_left_right_action(RailEnvActions.STOP_MOVING, rail, (1, 1), Grid4TransitionsEnum.EAST) == RailEnvActions.STOP_MOVING

    rail = GridTransitionMap(3, 3, RailEnvTransitions())
    rail.set_transitions((1, 1,), RailEnvTransitionsEnum.simple_switch_north_right)
    rail.set_transitions((1, 2,), RailEnvTransitionsEnum.horizontal_straight)
    assert preprocess_left_right_action(RailEnvActions.DO_NOTHING, rail, (1, 1), Grid4TransitionsEnum.NORTH) == RailEnvActions.DO_NOTHING
    assert preprocess_left_right_action(RailEnvActions.MOVE_LEFT, rail, (1, 1), Grid4TransitionsEnum.NORTH) == RailEnvActions.MOVE_FORWARD
    assert preprocess_left_right_action(RailEnvActions.MOVE_FORWARD, rail, (1, 1), Grid4TransitionsEnum.NORTH) == RailEnvActions.MOVE_FORWARD
    assert preprocess_left_right_action(RailEnvActions.MOVE_RIGHT, rail, (1, 1), Grid4TransitionsEnum.NORTH) == RailEnvActions.MOVE_RIGHT
    assert preprocess_left_right_action(RailEnvActions.STOP_MOVING, rail, (1, 1), Grid4TransitionsEnum.NORTH) == RailEnvActions.STOP_MOVING
