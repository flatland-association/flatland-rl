import random

import numpy as np

from flatland.core.transition_map import GridTransitionMap
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

random.seed(100)
np.random.seed(100)


def custom_rail_generator():
    def generator(width, height, num_agents=0, num_resets=0):
        rail_trans = RailEnvTransitions()
        grid_map = GridTransitionMap(width=width, height=height, transitions=rail_trans)
        rail_array = grid_map.grid
        rail_array.fill(0)

        agents_positions = []
        agents_direction = []
        agents_target = []

        return grid_map, agents_positions, agents_direction, agents_target

    return generator


env = RailEnv(width=6,
              height=4,
              rail_generator=custom_rail_generator(),
              number_of_agents=1)

env.reset()

env_renderer = RenderTool(env)
env_renderer.renderEnv(show=True)

input("Press Enter to continue...")
