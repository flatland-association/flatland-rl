import numpy as np

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
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

    obs, all_rewards, done, _ = env.step({i: RailEnvActions.MOVE_FORWARD for i in range(number_of_agents)})
    for i in range(len(env.agents)):
        agent: EnvAgent = env.agents[i]
        print("[{}] status={}, position={}, target={}, initial_position={}".format(i, agent.status, agent.position,
                                                                                   agent.target,
                                                                                   agent.initial_position))

    for i, agent in enumerate(env.agents):
        obs_agents_state = obs[i][1]
        obs_targets = obs[i][2]

        #
        nr_agents = np.count_nonzero(obs_targets[:, :, 0])
        assert nr_agents == 1

        for r in range(env.height):
            for c in range(env.width):
                _other_agent_target = 0
                for other_i, other_agent in enumerate(env.agents):
                    if other_agent.target == (r, c):
                        _other_agent_target = 1
                        break
                assert obs_targets[(r, c)][1] == _other_agent_target, "agent {} at {} expected {}".format(i, (r,c), _other_agent_target)
