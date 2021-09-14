import os
import json
import itertools
import warnings
from typing import Tuple, List, Callable, Mapping, Optional, Any
from flatland.envs.timetable_utils import Timetable

import numpy as np
from numpy.random.mtrand import RandomState

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env_shortest_paths import get_shortest_paths

def len_handle_none(v):
    if v is not None:
        return len(v)
    else:
        return 0

def timetable_generator(agents: List[EnvAgent], distance_map: DistanceMap, 
                            agents_hints: dict, np_random: RandomState = None) -> Timetable:
    """
    Calculates earliest departure and latest arrival times for the agents
    This is the new addition in Flatland 3
    Also calculates the max episodes steps based on the density of the timetable

    inputs: 
        agents - List of all the agents rail_env.agents
        distance_map - Distance map of positions to tagets of each agent in each direction
        agent_hints - Uses the number of cities
        np_random - RNG state for seeding
    returns:
        Timetable with the latest_arrivals, earliest_departures and max_episdode_steps
    """
    # max_episode_steps calculation
    if agents_hints:
        city_positions = agents_hints['city_positions']
        num_cities = len(city_positions)
    else:
        num_cities = 2

    timedelay_factor = 4
    alpha = 2
    max_episode_steps = int(timedelay_factor * alpha * \
        (distance_map.rail.width + distance_map.rail.height + (len(agents) / num_cities)))
    
    # Multipliers
    old_max_episode_steps_multiplier = 3.0
    new_max_episode_steps_multiplier = 1.5
    travel_buffer_multiplier = 1.3 # must be strictly lesser than new_max_episode_steps_multiplier
    assert new_max_episode_steps_multiplier > travel_buffer_multiplier
    end_buffer_multiplier = 0.05
    mean_shortest_path_multiplier = 0.2
    
    shortest_paths = get_shortest_paths(distance_map)
    shortest_paths_lengths = [len_handle_none(v) for k,v in shortest_paths.items()]

    # Find mean_shortest_path_time
    agent_speeds = [agent.speed_counter.speed for agent in agents]
    agent_shortest_path_times = np.array(shortest_paths_lengths)/ np.array(agent_speeds)
    mean_shortest_path_time = np.mean(agent_shortest_path_times)

    # Deciding on a suitable max_episode_steps
    longest_speed_normalized_time = np.max(agent_shortest_path_times)
    mean_path_delay = mean_shortest_path_time * mean_shortest_path_multiplier
    max_episode_steps_new = int(np.ceil(longest_speed_normalized_time * new_max_episode_steps_multiplier) + mean_path_delay)
    
    max_episode_steps_old = int(max_episode_steps * old_max_episode_steps_multiplier)

    max_episode_steps = min(max_episode_steps_new, max_episode_steps_old)
    
    end_buffer = int(max_episode_steps * end_buffer_multiplier)
    latest_arrival_max = max_episode_steps-end_buffer

    # Useless unless needed by returning
    earliest_departures = []
    latest_arrivals = []

    for agent in agents:
        agent_shortest_path_time = agent_shortest_path_times[agent.handle]
        agent_travel_time_max = int(np.ceil((agent_shortest_path_time * travel_buffer_multiplier) + mean_path_delay))
        
        departure_window_max = max(latest_arrival_max - agent_travel_time_max, 1)
        
        earliest_departure = np_random.randint(0, departure_window_max)
        latest_arrival = earliest_departure + agent_travel_time_max
        
        earliest_departures.append(earliest_departure)
        latest_arrivals.append(latest_arrival)

        agent.earliest_departure = earliest_departure
        agent.latest_arrival = latest_arrival

    return Timetable(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals,
                    max_episode_steps=max_episode_steps)
