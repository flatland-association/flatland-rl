import numpy as np

from flatland.envs.generators import sparse_rail_generator, realistic_rail_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool, AgentRenderVariant


def test_realistic_rail_generator(vizualization_folder_name=None):
    for test_loop in range(20):
        num_agents = np.random.randint(10, 30)
        env = RailEnv(width=np.random.randint(40, 80),
                      height=np.random.randint(10, 20),
                      rail_generator=realistic_rail_generator(nr_start_goal=num_agents + 1, seed=test_loop),
                      number_of_agents=num_agents,
                      obs_builder_object=GlobalObsForRailEnv())
        # reset to initialize agents_static
        env_renderer = RenderTool(env, gl="PILSVG", agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND, screen_height=1200,
                                  screen_width=1600)
        env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

def test_sparse_rail_generator():
    env = RailEnv(width=50,
                  height=50,
                  rail_generator=sparse_rail_generator(num_cities=10,  # Number of cities in map
                                                       num_intersections=10,  # Number of interesections in map
                                                       num_trainstations=50,  # Number of possible start/targets on map
                                                       min_node_dist=6,  # Minimal distance of nodes
                                                       node_radius=3,  # Proximity of stations to city center
                                                       num_neighb=3,  # Number of connections to other cities
                                                       seed=5,  # Random seed
                                                       realistic_mode=True  # Ordered distribution of nodes
                                                       ),
                  number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv())
    # reset to initialize agents_static
    env_renderer = RenderTool(env, gl="PILSVG", )
    env_renderer.render_env(show=True, show_observations=True, show_predictions=False)

