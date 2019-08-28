from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.utils.rendertools import RenderTool


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
                                                       realistic_mode=False  # Ordered distribution of nodes
                                                       ),
                  schedule_generator=sparse_schedule_generator(),
                  number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv())
    # reset to initialize agents_static
    env_renderer = RenderTool(env, gl="PILSVG", )
    env_renderer.render_env(show=True, show_observations=True, show_predictions=False)
    env_renderer.gl.save_image("./sparse_generator_false.png")
