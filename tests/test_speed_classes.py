"""Test speed initialization by a map of speeds and their corresponding ratios."""
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator
from flatland.envs.schedule_generators import speed_initialization_helper, complex_schedule_generator


def test_speed_initialization_helper():
    random_generator = np.random.RandomState()
    random_generator.seed(10)
    speed_ratio_map = {1: 0.3, 2: 0.4, 3: 0.3}
    actual_speeds = speed_initialization_helper(10, speed_ratio_map, np_random=random_generator)

    # seed makes speed_initialization_helper deterministic -> check generated speeds.
    assert actual_speeds == [3, 1, 2, 3, 2, 1, 1, 3, 1, 1]


def test_rail_env_speed_intializer():
    speed_ratio_map = {1: 0.3, 2: 0.4, 3: 0.1, 5: 0.2}

    env = RailEnv(width=50, height=50,
                  rail_generator=complex_rail_generator(nr_start_goal=10, nr_extra=1, min_dist=8, max_dist=99999,
                                                        seed=1), schedule_generator=complex_schedule_generator(),
                  number_of_agents=10)
    env.reset()
    actual_speeds = list(map(lambda agent: agent.speed_data['speed'], env.agents))

    expected_speed_set = set(speed_ratio_map.keys())

    # check that the number of speeds generated is correct
    assert len(actual_speeds) == env.get_num_agents()

    # check that only the speeds defined are generated
    assert all({(actual_speed in expected_speed_set) for actual_speed in actual_speeds})
