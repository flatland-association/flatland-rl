import time

from flatland.envs.generators import sparse_rail_generator,realistic_rail_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool


def test_realistic_rail_generator():

    env = RailEnv(width=40,
                  height=16,
                  rail_generator=realistic_rail_generator(),
                  number_of_agents=15,
                  obs_builder_object=GlobalObsForRailEnv())
    # reset to initialize agents_static
    env_renderer = RenderTool(env, gl="PILSVG", )
    env_renderer.render_env(show=True, show_observations=True, show_predictions=False)
    time.sleep(10)


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
