import numpy as np

from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators_city_generator import city_generator
from flatland.envs.schedule_generators import city_schedule_generator


def test_city_generator():
    dist_fun = Vec2d.get_manhattan_distance
    env = RailEnv(width=50,
                  height=50,
                  rail_generator=city_generator(num_cities=5,
                                                city_size=10,
                                                allowed_rotation_angles=[90],
                                                max_number_of_station_tracks=4,
                                                nbr_of_switches_per_station_track=2,
                                                connect_max_nbr_of_shortes_city=2,
                                                do_random_connect_stations=False,
                                                a_star_distance_function=dist_fun,
                                                seed=0,
                                                print_out_info=False
                                                ),
                  schedule_generator=city_schedule_generator(),
                  number_of_agents=10,
                  obs_builder_object=GlobalObsForRailEnv())

    # approximative test (replace image comparison)
    assert np.sum(env.rail.grid) == 3642337
    s0 = 0
    s1 = 0
    for a in range(env.get_num_agents()):
        s0 = Vec2d.get_manhattan_distance(env.agents[a].position, (0, 0))
        s1 = Vec2d.get_chebyshev_distance(env.agents[a].position, (0, 0))
    assert s0 == 58, "actual={}".format(s0)
    assert s1 == 38, "actual={}".format(s1)
