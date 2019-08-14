import time

from flatland.envs.generators import sparse_rail_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.utils.rendertools import RenderTool


def test_sparse_rail_generator():

    env = RailEnv(width=20,
                  height=20,
                  rail_generator=sparse_rail_generator(nr_train_stations=3, nr_nodes=2, min_node_dist=5,
                                                       node_radius=4),
                  number_of_agents=3,
                  obs_builder_object=GlobalObsForRailEnv())
    # reset to initialize agents_static
    env_renderer = RenderTool(env, gl="PILSVG", )
    env_renderer.render_env(show=True, show_observations=True, show_predictions=False)
    time.sleep(10)
