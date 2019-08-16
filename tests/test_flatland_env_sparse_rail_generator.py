import time

from flatland.envs.generators import sparse_rail_generator,realistic_rail_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool

import numpy as np

def test_realistic_rail_generator():
    for test_loop in range(5):
        num_agents = np.random.randint(10,30)
        env = RailEnv(width=np.random.randint(40,80),
                      height=np.random.randint(10,20),
                      rail_generator=realistic_rail_generator(nr_start_goal=num_agents+1,seed=test_loop),
                      number_of_agents=num_agents,
                      obs_builder_object=GlobalObsForRailEnv())
        # reset to initialize agents_static
        env_renderer = RenderTool(env, gl="PILSVG", )
        env_renderer.render_env(show=True, show_observations=True, show_predictions=False)
        time.sleep(2)
        env_renderer.close_window()

def test_sparse_rail_generator():

    env = RailEnv(width=20,
                  height=20,
                  rail_generator=sparse_rail_generator(nr_nodes=3, min_node_dist=8,
                                                       node_radius=4),
                  number_of_agents=15,
                  obs_builder_object=GlobalObsForRailEnv())
    # reset to initialize agents_static
    env_renderer = RenderTool(env, gl="PILSVG", )
    env_renderer.render_env(show=True, show_observations=True, show_predictions=False)
    time.sleep(2)
