import os

import numpy as np

from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators_city_generator import city_generator
from flatland.envs.schedule_generators import city_schedule_generator
from flatland.utils.rendertools import RenderTool, AgentRenderVariant

OUTPUT_DIR = "./../render_output/"
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

for itrials in np.arange(1, 15, 1):
    print(itrials, "generate new city")

    # init seed
    np.random.seed(itrials)

    # select distance function used in a-star path finding
    dist_fun = Vec2d.get_manhattan_distance
    dfsel = (itrials - 1) % 3
    if dfsel == 1:
        dist_fun = Vec2d.get_euclidean_distance
    elif dfsel == 2:
        dist_fun = Vec2d.get_chebyshev_distance

    # create RailEnv and use the city_generator to create a map
    env = RailEnv(width=40 + np.random.choice(100),
                  height=40 + np.random.choice(100),
                  rail_generator=city_generator(num_cities=5 + np.random.choice(10),
                                                city_size=10 + np.random.choice(5),
                                                allowed_rotation_angles=np.arange(0, 360, 6),
                                                max_number_of_station_tracks=4 + np.random.choice(4),
                                                nbr_of_switches_per_station_track=2 + np.random.choice(2),
                                                connect_max_nbr_of_shortes_city=2 + np.random.choice(4),
                                                do_random_connect_stations=itrials % 2 == 0,
                                                a_star_distance_function=dist_fun,
                                                seed=itrials,
                                                print_out_info=False
                                                ),
                  schedule_generator=city_schedule_generator(),
                  number_of_agents=10000,
                  obs_builder_object=GlobalObsForRailEnv())

    # reset to initialize agents_static
    env_renderer = RenderTool(env, gl="PILSVG", screen_width=1400, screen_height=1000,
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX)

    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

    # store rendered file into render_output if the path exists
    env_renderer.gl.save_image(
        os.path.join(
            OUTPUT_DIR,
            "flatland_frame_{:04d}.png".format(itrials)
        ))

    # close the renderer / window
    env_renderer.close_window()
