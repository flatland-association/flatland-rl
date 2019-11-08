import random

import numpy as np

from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import random_rail_generator

random.seed(100)
np.random.seed(100)


class SimpleObs(ObservationBuilder):
    """
    Simplest observation builder. The object returns observation vectors with 5 identical components,
    all equal to the ID of the respective agent.
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        return

    def get(self, handle: int = 0) -> np.ndarray:
        observation = handle * np.ones((5,))
        return observation


def main():
    env = RailEnv(width=7, height=7, rail_generator=random_rail_generator(), number_of_agents=3,
                  obs_builder_object=SimpleObs())
    env.reset()

    # Print the observation vector for each agents
    obs, all_rewards, done, _ = env.step({0: 0})
    for i in range(env.get_num_agents()):
        print("Agent ", i, "'s observation: ", obs[i])


if __name__ == '__main__':
    main()
