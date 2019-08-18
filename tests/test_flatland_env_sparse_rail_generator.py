import time

import numpy as np

from flatland.envs.generators import sparse_rail_generator, realistic_rail_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool


def test_realistic_rail_generator():
    for test_loop in range(20):
        num_agents = np.random.randint(10, 30)
        env = RailEnv(width=np.random.randint(40, 80),
                      height=np.random.randint(10, 20),
                      rail_generator=realistic_rail_generator(nr_start_goal=num_agents + 1, seed=test_loop),
                      number_of_agents=num_agents,
                      obs_builder_object=GlobalObsForRailEnv())
        # reset to initialize agents_static
        env_renderer = RenderTool(env, gl="PILSVG", )
        env_renderer.render_env(show=True, show_observations=True, show_predictions=False)
        env_renderer.close_window()


def test_sparse_rail_generator():
    env = RailEnv(width=50,
                  height=50,
                  rail_generator=sparse_rail_generator(num_cities=2,  # Number of cities in map
                                                       num_intersections=3,  # Number of interesections in map
                                                       num_trainstations=5,  # Number of possible start/targets on map
                                                       min_node_dist=10,  # Minimal distance of nodes
                                                       node_radius=2,  # Proximity of stations to city center
                                                       num_neighb=3,  # Number of connections to other cities
                                                       seed=5,  # Random seed
                                                       ),
                  number_of_agents=0,
                  obs_builder_object=GlobalObsForRailEnv())
    # reset to initialize agents_static
    env_renderer = RenderTool(env, gl="PILSVG", )
    env_renderer.render_env(show=True, show_observations=True, show_predictions=False)
    time.sleep(10)
