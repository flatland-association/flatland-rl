import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail2


def test_random_seeding():
    # Set fixed malfunction duration for this test
    stochastic_data = {'prop_malfunction': 1.,
                       'malfunction_rate': 1000,
                       'min_duration': 3,
                       'max_duration': 3}

    rail, rail_map = make_simple_rail2()

    env = RailEnv(width=25,
                  height=30,
                  rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(),
                  number_of_agents=1,
                  stochastic_data=stochastic_data,  # Malfunction data generator
                  )
    # reset to initialize agents_static
    obs, info = env.reset(True, True, False, random_seed=0)
    env.agents[0].target = (0, 0)
    # Move target to unreachable position in order to not interfere with test
    for idx in range(4):
        env.reset(True, True, False, random_seed=0)
        np.random.seed(0)
        # Test generation print
        print("assert env.agents[0].initial_position == {}".format(env.agents[0].initial_position))

        env.agents[0].target = (0, 0)
        # assert env.agents[0].initial_position == (3, 3)
        for step in range(10):
            actions = {}

            for i in range(len(obs)):
                actions[i] = np.random.randint(4)
            env.step(actions)
        #assert env.agents[0].position == (3, 9)
        # Test generation print
        print("assert  env.agents[0].position == {}".format(env.agents[0].position))
