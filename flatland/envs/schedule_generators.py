import os
import json
import itertools
import warnings
from typing import Tuple, List, Callable, Mapping, Optional, Any
from flatland.envs.schedule_utils import Schedule

import numpy as np
from numpy.random.mtrand import RandomState

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env_shortest_paths import get_shortest_paths


# #### DATA COLLECTION *************************
# import termplotlib as tpl
# import matplotlib.pyplot as plt
# root_path = 'C:\\Users\\nimish\\Programs\\AIcrowd\\flatland\\flatland\\playground'
# dir_name = 'TEMP'
# os.mkdir(os.path.join(root_path, dir_name))

# # Histogram 1
# dist_resolution = 50
# schedule_dist = np.zeros(shape=(dist_resolution))
# # Volume dist
# route_dist = None
# # Dist - shortest path
# shortest_paths_len_dist = []
# # City positions
# city_positions = []
# #### DATA COLLECTION *************************

def schedule_generator(agents: List[EnvAgent], config_speeds: List[float],  distance_map: DistanceMap, 
                            agents_hints: dict, np_random: RandomState = None) -> Schedule:

    # max_episode_steps calculation
    city_positions = agents_hints['city_positions']
    timedelay_factor = 4
    alpha = 2
    max_episode_steps = int(timedelay_factor * alpha * \
        (distance_map.rail.width + distance_map.rail.height + (len(agents) / len(city_positions))))
    
    # Multipliers
    old_max_episode_steps_multiplier = 3.0
    new_max_episode_steps_multiplier = 1.5
    travel_buffer_multiplier = 1.3 # must be strictly lesser than new_max_episode_steps_multiplier
    end_buffer_multiplier = 0.05
    mean_shortest_path_multiplier = 0.2
    
    shortest_paths = get_shortest_paths(distance_map)
    shortest_paths_lengths = [len(v) for k,v in shortest_paths.items()]

    # Find mean_shortest_path_time
    agent_shortest_path_times = []
    for agent in agents:
        speed = agent.speed_data['speed']
        distance = shortest_paths_lengths[agent.handle]
        agent_shortest_path_times.append(int(np.ceil(distance / speed)))

    mean_shortest_path_time = np.mean(agent_shortest_path_times)

    # Deciding on a suitable max_episode_steps
    max_sp_len = max(shortest_paths_lengths) # longest path
    min_speed = min(config_speeds)           # slowest possible speed in config
    
    longest_sp_time = max_sp_len / min_speed
    max_episode_steps_new = int(np.ceil(longest_sp_time * new_max_episode_steps_multiplier))
    
    max_episode_steps_old = int(max_episode_steps * old_max_episode_steps_multiplier)

    max_episode_steps = min(max_episode_steps_new, max_episode_steps_old)
    
    end_buffer = max_episode_steps * end_buffer_multiplier
    latest_arrival_max = max_episode_steps-end_buffer

    # Useless unless needed by returning
    earliest_departures = []
    latest_arrivals = []

    # #### DATA COLLECTION *************************
    # # Create info.txt
    # with open(os.path.join(root_path, dir_name, 'INFO.txt'), 'w') as f:
    #     f.write('COPY FROM main.py')

    # # Volume dist
    # route_dist = np.zeros(shape=(max_episode_steps, distance_map.rail.width, distance_map.rail.height), dtype=np.int8)

    # # City positions
    # # Dummy distance map for shortest path pairs between cities
    # city_positions = agents_hints['city_positions']
    # d_rail = distance_map.rail
    # d_dmap = DistanceMap([], d_rail.height, d_rail.width)
    # d_city_permutations = list(itertools.permutations(city_positions, 2))

    # d_positions = []
    # d_targets = []
    # for position, target in d_city_permutations:
    #     d_positions.append(position)
    #     d_targets.append(target)
    
    # d_schedule = Schedule(d_positions,
    #                       [0] * len(d_positions),
    #                       d_targets,
    #                       [1.0] * len(d_positions),
    #                       [None] * len(d_positions),
    #                       1000)
    
    # d_agents = EnvAgent.from_schedule(d_schedule)
    # d_dmap.reset(d_agents, d_rail)
    # d_map = d_dmap.get()

    # d_data = {
    #     'city_positions': city_positions,
    #     'start': d_positions,
    #     'end': d_targets,
    # }
    # with open(os.path.join(root_path, dir_name, 'city_data.json'), 'w') as f:
    #     json.dump(d_data, f)

    # with open(os.path.join(root_path, dir_name, 'distance_map.npy'), 'wb') as f:
    #     np.save(f, d_map)
    # #### DATA COLLECTION *************************

    for agent in agents:
        agent_shortest_path_time = agent_shortest_path_times[agent.handle]
        agent_travel_time_max = int(np.ceil((agent_shortest_path_time * travel_buffer_multiplier) \
                                            + (mean_shortest_path_time * mean_shortest_path_multiplier)))
        
        departure_window_max = latest_arrival_max - agent_travel_time_max

        earliest_departure = np_random.randint(0, departure_window_max)
        latest_arrival = earliest_departure + agent_travel_time_max
        
        earliest_departures.append(earliest_departure)
        latest_arrivals.append(latest_arrival)

        agent.earliest_departure = earliest_departure
        agent.latest_arrival = latest_arrival

    # #### DATA COLLECTION *************************
    #     # Histogram 1
    #     dist_bounds = get_dist_window(earliest_departure, latest_arrival, latest_arrival_max)
    #     schedule_dist[dist_bounds[0]: dist_bounds[1]] += 1

    #     # Volume dist
    #     for waypoint in agent_shortest_path:
    #         pos = waypoint.position
    #         route_dist[earliest_departure:latest_arrival, pos[0], pos[1]] += 1

    #     # Dist - shortest path
    #     shortest_paths_len_dist.append(agent_shortest_path_len)

    # np.save(os.path.join(root_path, dir_name, 'volume.npy'), route_dist)
    
    # shortest_paths_len_dist.sort()
    # save_sp_fig()
    # #### DATA COLLECTION *************************

    # returns schedule
    return Schedule(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals,
                    max_episode_steps=max_episode_steps)


# #### DATA COLLECTION *************************
# # Histogram 1
# def get_dist_window(departure_t, arrival_t, latest_arrival_max):
#     return (int(np.round(np.interp(departure_t, [0, latest_arrival_max], [0, dist_resolution]))),
#             int(np.round(np.interp(arrival_t, [0, latest_arrival_max], [0, dist_resolution]))))

# def plot_dist():
#     counts, bin_edges = schedule_dist, [i for i in range(0, dist_resolution+1)]
#     fig = tpl.figure()
#     fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
#     fig.show()

# # Shortest path dist
# def save_sp_fig():
#     fig = plt.figure(figsize=(15, 7))
#     plt.bar(np.arange(len(shortest_paths_len_dist)), shortest_paths_len_dist)
#     plt.savefig(os.path.join(root_path, dir_name, 'shortest_paths_sorted.png'))
# #### DATA COLLECTION *************************
