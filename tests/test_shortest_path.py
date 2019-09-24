import numpy as np

from flatland.envs.rail_env import RailEnvNextAction, RailEnvActions
from flatland.envs.rail_env_utils import load_flatland_environment_from_file, get_shortest_paths


def test_get_shortest_paths():
    env = load_flatland_environment_from_file('test_002.pkl', 'env_data.tests')
    actual = get_shortest_paths(env.distance_map)

    expected = {
        0: [RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(1, 2), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(1, 3), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 3), next_direction=2),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 4), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 5), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 6), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 7), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 8), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 9), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 10), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 11), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 12), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 13), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 14), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 15), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 16), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 17), next_direction=1),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 18), next_direction=1)],
        1: [RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(3, 17), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(3, 16), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 16), next_direction=0),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 15), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 14), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 13), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 12), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 11), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 10), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 9), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 8), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 7), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 6), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 5), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 4), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 3), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 2), next_direction=3),
            RailEnvNextAction(action=RailEnvActions.MOVE_FORWARD, next_position=(2, 1), next_direction=3)]
    }

    for agent_handle in expected:
        assert np.array_equal(actual[agent_handle], expected[agent_handle]), \
            "[{}] actual={},expected={}".format(agent_handle, actual[agent_handle], expected[agent_handle])
