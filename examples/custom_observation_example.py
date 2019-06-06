import random

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.generators import random_rail_generator
from flatland.envs.rail_env import RailEnv

random.seed(100)
np.random.seed(100)


class CustomObs(ObservationBuilder):
    def __init__(self):
        self.observation_space = [5]

    def reset(self):
        return

    def get(self, handle):
        observation = handle * np.ones((5,))
        return observation


env = RailEnv(width=7,
              height=7,
              rail_generator=random_rail_generator(),
              number_of_agents=3,
              obs_builder_object=CustomObs())

# Print the observation vector for each agents
obs, all_rewards, done, _ = env.step({0: 0})
for i in range(env.get_num_agents()):
    print("Agent ", i, "'s observation: ", obs[i])
