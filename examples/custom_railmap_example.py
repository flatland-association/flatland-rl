import random

import numpy as np

from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
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
        new_tran = rail_trans.set_transition(1, 1, 1, 1)
        print(new_tran)
        agents_positions = []
        agents_direction = []
        agents_target = []
        rail_array[0, 0] = new_tran
        rail_array[0, 1] = new_tran
        return grid_map, agents_positions, agents_direction, agents_target

    return generator


env = RailEnv(width=6,
              height=4,
              rail_generator=custom_rail_generator(),
              number_of_agents=1)

env.reset()

env_renderer = RenderTool(env)
env_renderer.render_env(show=True)

input("Press Enter to continue...")
