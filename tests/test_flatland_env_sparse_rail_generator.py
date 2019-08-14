from flatland.envs.generators import sparse_rail_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv


def test_sparse_rail_generator():
    env = RailEnv(width=20,
                  height=20,
                  rail_generator=sparse_rail_generator(nr_train_stations=10, nr_nodes=5, min_node_dist=10,
                                                       node_radius=4),
                  number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv())
    # reset to initialize agents_static
    env.reset()
