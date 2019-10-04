import numpy as np

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import DummyPredictorForRailEnv
from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions, RailEnv
from flatland.envs.rail_env_shortest_paths import get_shortest_paths, WalkingElement
from flatland.envs.rail_env_utils import load_flatland_environment_from_file
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_disconnected_simple_rail


def test_get_shortest_paths_unreachable():
    rail, rail_map = make_disconnected_simple_rail()

    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=DummyPredictorForRailEnv(max_depth=10)),
                  )

    # set the initial position
    agent = env.agents_static[0]
    agent.position = (3, 1)  # west dead-end
    agent.initial_position = (3, 1)  # west dead-end
    agent.direction = Grid4TransitionsEnum.WEST
    agent.target = (3, 9)  # east dead-end
    agent.moving = True

    # reset to set agents from agents_static
    env.reset(False, False)

    actual = get_shortest_paths(env.distance_map)
    expected = {0: None}

    assert actual == expected, "actual={},expected={}".format(actual, expected)


def test_get_shortest_paths():
    env = load_flatland_environment_from_file('test_002.pkl', 'env_data.tests')
    actual = get_shortest_paths(env.distance_map)

    expected = {
        0: [
            WalkingElement(position=(1, 1), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(1, 2), next_direction=1)),
            WalkingElement(position=(1, 2), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(1, 3), next_direction=1)),
            WalkingElement(position=(1, 3), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 3), next_direction=2)),
            WalkingElement(position=(2, 3), direction=2,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 4), next_direction=1)),
            WalkingElement(position=(2, 4), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 5), next_direction=1)),
            WalkingElement(position=(2, 5), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 6), next_direction=1)),
            WalkingElement(position=(2, 6), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 7), next_direction=1)),
            WalkingElement(position=(2, 7), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 8), next_direction=1)),
            WalkingElement(position=(2, 8), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 9), next_direction=1)),
            WalkingElement(position=(2, 9), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 10), next_direction=1)),
            WalkingElement(position=(2, 10), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 11), next_direction=1)),
            WalkingElement(position=(2, 11), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 12), next_direction=1)),
            WalkingElement(position=(2, 12), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 13), next_direction=1)),
            WalkingElement(position=(2, 13), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 14), next_direction=1)),
            WalkingElement(position=(2, 14), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 15), next_direction=1)),
            WalkingElement(position=(2, 15), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 16), next_direction=1)),
            WalkingElement(position=(2, 16), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 17), next_direction=1)),
            WalkingElement(position=(2, 17), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 18), next_direction=1)),
            WalkingElement(position=(2, 18), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.STOP_MOVING,
                                                                 next_position=(2, 18), next_direction=1))],
        1: [
            WalkingElement(position=(3, 18), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(3, 17), next_direction=3)),
            WalkingElement(position=(3, 17), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(3, 16), next_direction=3)),
            WalkingElement(position=(3, 16), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 16), next_direction=0)),
            WalkingElement(position=(2, 16), direction=0,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 15), next_direction=3)),
            WalkingElement(position=(2, 15), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 14), next_direction=3)),
            WalkingElement(position=(2, 14), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 13), next_direction=3)),
            WalkingElement(position=(2, 13), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 12), next_direction=3)),
            WalkingElement(position=(2, 12), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 11), next_direction=3)),
            WalkingElement(position=(2, 11), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 10), next_direction=3)),
            WalkingElement(position=(2, 10), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 9), next_direction=3)),
            WalkingElement(position=(2, 9), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 8), next_direction=3)),
            WalkingElement(position=(2, 8), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 7), next_direction=3)),
            WalkingElement(position=(2, 7), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 6), next_direction=3)),
            WalkingElement(position=(2, 6), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 5), next_direction=3)),
            WalkingElement(position=(2, 5), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 4), next_direction=3)),
            WalkingElement(position=(2, 4), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 3), next_direction=3)),
            WalkingElement(position=(2, 3), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 2), next_direction=3)),
            WalkingElement(position=(2, 2), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(2, 1), next_direction=3)),
            WalkingElement(position=(2, 1), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.STOP_MOVING,
                                                                 next_position=(2, 1), next_direction=3))]
    }

    for agent_handle in expected:
        assert np.array_equal(actual[agent_handle], expected[agent_handle]), \
            "[{}] actual={},expected={}".format(agent_handle, actual[agent_handle], expected[agent_handle])


def test_get_shortest_paths_max_depth():
    env = load_flatland_environment_from_file('test_002.pkl', 'env_data.tests')
    actual = get_shortest_paths(env.distance_map, max_depth=2)

    expected = {
        0: [
            WalkingElement(position=(1, 1), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(1, 2), next_direction=1)),
            WalkingElement(position=(1, 2), direction=1,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(1, 3), next_direction=1))
        ],
        1: [
            WalkingElement(position=(3, 18), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(3, 17), next_direction=3)),
            WalkingElement(position=(3, 17), direction=3,
                           next_action_element=RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD,
                                                                 next_position=(3, 16), next_direction=3)),
        ]
    }

    for agent_handle in expected:
        assert np.array_equal(actual[agent_handle], expected[agent_handle]), \
            "[{}] actual={},expected={}".format(agent_handle, actual[agent_handle], expected[agent_handle])
