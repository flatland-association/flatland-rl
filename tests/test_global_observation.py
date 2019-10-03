import numpy as np

from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator


def test_get_global_observation():
    np.random.seed(1)
    number_of_agents = 20

    stochastic_data = {'prop_malfunction': 1.,  # Percentage of defective agents
                       'malfunction_rate': 30,  # Rate of malfunction occurence
                       'min_duration': 3,  # Minimal duration of malfunction
                       'max_duration': 20  # Max duration of malfunction
                       }

    speed_ration_map = {1.: 0.25,  # Fast passenger train
                        1. / 2.: 0.25,  # Fast freight train
                        1. / 3.: 0.25,  # Slow commuter train
                        1. / 4.: 0.25}  # Slow freight train

    env = RailEnv(width=50,
                  height=50,
                  rail_generator=sparse_rail_generator(max_num_cities=6,
                                                       max_rails_between_cities=4,
                                                       seed=15,
                                                       grid_mode=False
                                                       ),
                  schedule_generator=sparse_schedule_generator(speed_ration_map),
                  number_of_agents=number_of_agents, stochastic_data=stochastic_data,  # Malfunction data generator
                  obs_builder_object=GlobalObsForRailEnv())
    obs, all_rewards, done, _ = env.step({0: 0})

    for i in range(len(env.agents)):
        obs_agents_state = obs[i][1]
        obs_targets = obs[i][2]

        nr_agents = np.count_nonzero(obs_targets[:, :, 0])
        nr_agents_other = np.count_nonzero(obs_targets[:, :, 1])
        assert nr_agents == 1
        assert nr_agents_other == (number_of_agents - 1)

        # since the array is initialized with -1 add one in order to used np.count_nonzero
        obs_agents_state += 1
        obs_agents_state_0 = np.count_nonzero(obs_agents_state[:, :, 0])
        obs_agents_state_1 = np.count_nonzero(obs_agents_state[:, :, 1])
        obs_agents_state_2 = np.count_nonzero(obs_agents_state[:, :, 2])
        obs_agents_state_3 = np.count_nonzero(obs_agents_state[:, :, 3])
        assert obs_agents_state_0 == 1
        assert obs_agents_state_1 == (number_of_agents - 1)
        assert obs_agents_state_2 == number_of_agents
        assert obs_agents_state_3 == number_of_agents
