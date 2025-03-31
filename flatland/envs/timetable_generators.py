"""Timetable generators: Railway Undertaking (RU) / Eisenbahnverkehrsunternehmen (EVU)."""
import pickle
import warnings
from pathlib import Path
from typing import List

import numpy as np
from numpy.random.mtrand import RandomState

from flatland.envs import persistence
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.timetable_utils import Timetable


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
        distance_map - Distance map of positions to targets of each agent in each direction
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
    num_agents = len(agents)
    max_episode_steps = int(timedelay_factor * alpha * \
                            (distance_map.rail.width + distance_map.rail.height + (num_agents / num_cities)))

    # Multipliers
    old_max_episode_steps_multiplier = 3.0
    new_max_episode_steps_multiplier = 1.5
    travel_buffer_multiplier = 1.3  # must be strictly lesser than new_max_episode_steps_multiplier
    assert new_max_episode_steps_multiplier > travel_buffer_multiplier
    end_buffer_multiplier = 0.05
    mean_shortest_path_multiplier = 0.2

    if len(agents[0].waypoints) > 1:
        # distance for intermediates parts and sum up
        line_length = len(agents[0].waypoints) - 1
        fake_agents = []
        for i in range(line_length):
            for a in agents:
                waypoints = a.waypoints

                fake_agents.append(EnvAgent(
                    handle=i * num_agents + a.handle,
                    initial_position=waypoints[i].position,
                    initial_direction=waypoints[i].direction,
                    position=waypoints[i].position,
                    direction=waypoints[i].direction,
                    target=waypoints[i + 1].position,
                ))
        distance_map_with_intermediates = DistanceMap(fake_agents, distance_map.env_height, distance_map.env_width)
        distance_map_with_intermediates.reset(fake_agents, distance_map.rail)

        shortest_paths = distance_map_with_intermediates.get_shortest_paths()
        shortest_path_segment_lengths = [[] for _ in range(num_agents)]
        for k, v in shortest_paths.items():
            shortest_path_segment_lengths[k % num_agents].append(len_handle_none(v))
        shortest_paths_lengths = [sum(l) for l in shortest_path_segment_lengths]
    else:
        shortest_paths = distance_map.get_shortest_paths()
        shortest_paths_lengths = [len_handle_none(v) for k, v in shortest_paths.items()]
        shortest_path_segment_lengths = [[l] for l in shortest_paths_lengths]

    # Find mean_shortest_path_time
    agent_speeds = [agent.speed_counter.speed for agent in agents]
    agent_shortest_path_times = np.array(shortest_paths_lengths) / np.array(agent_speeds)
    mean_shortest_path_time = np.mean(agent_shortest_path_times)

    # Deciding on a suitable max_episode_steps
    longest_speed_normalized_time = np.max(agent_shortest_path_times)
    mean_path_delay = mean_shortest_path_time * mean_shortest_path_multiplier
    max_episode_steps_new = int(np.ceil(longest_speed_normalized_time * new_max_episode_steps_multiplier) + mean_path_delay)

    max_episode_steps_old = int(max_episode_steps * old_max_episode_steps_multiplier)

    max_episode_steps = min(max_episode_steps_new, max_episode_steps_old)

    end_buffer = int(max_episode_steps * end_buffer_multiplier)
    latest_arrival_max = max_episode_steps - end_buffer

    earliest_departures = []
    latest_arrivals = []

    for agent in agents:
        agent_shortest_path_time = agent_shortest_path_times[agent.handle]
        agent_travel_time_max = int(np.ceil((agent_shortest_path_time * travel_buffer_multiplier) + mean_path_delay))

        departure_window_max = max(latest_arrival_max - agent_travel_time_max, 1)

        earliest_departure = np_random.randint(0, departure_window_max)
        latest_arrival = earliest_departure + agent_travel_time_max

        agent.earliest_departure = earliest_departure
        agent.latest_arrival = latest_arrival
        ed = earliest_departure
        eds = [earliest_departure]
        for l in shortest_path_segment_lengths[agent.handle]:
            ed += l
            eds.append(ed)
        la = latest_arrival
        las = [latest_arrival]
        for l in reversed(shortest_path_segment_lengths[agent.handle]):
            la -= l
            las.insert(0, la)
        eds[-1] = None
        las[0] = None
        earliest_departures.append(eds)
        latest_arrivals.append(las)

    return Timetable(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals,
                     max_episode_steps=max_episode_steps)


def ttgen_flatland2(agents: List[EnvAgent], distance_map: DistanceMap,
                    agents_hints: dict, np_random: RandomState = None) -> Timetable:
    n_max_steps = 1000
    return Timetable(
        earliest_departures=[[0]] * len(agents),
        latest_arrivals=[[n_max_steps]] * len(agents),
        max_episode_steps=n_max_steps)


class FileTimetableGenerator:
    def __init__(self, filename: Path, load_from_package: bool = None):
        self.filename = filename
        self.load_from_package = load_from_package

    def generate(self, *args, **kwargs) -> Timetable:
        if self.load_from_package is not None:
            from importlib_resources import read_binary
            load_data = read_binary(self.load_from_package, self.filename)
        else:
            with open(self.filename, "rb") as file_in:
                load_data = file_in.read()
        return pickle.loads(load_data)

    @staticmethod
    def save(filename: Path, tt: Timetable):
        with open(filename, "wb") as file_out:
            file_out.write(pickle.dumps(tt))

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    @staticmethod
    def wrap(timetable_generator: timetable_generator, tt_pkl: Path) -> timetable_generator:
        def _wrap(*args, **kwargs):
            tt = timetable_generator(*args, **kwargs)
            FileTimetableGenerator.save(tt_pkl, tt)
            return tt

        return _wrap


def timetable_from_file(filename: Path, load_from_package: bool = None) -> timetable_generator:
    """
    Loads timetable from env pickle file.

    Parameters
    ----------
    filename : Pickle file generated by RailEnvPersister.save()

    Returns
    -------
    Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
        initial positions, directions, targets speeds
    """

    def generator(agents: List[EnvAgent], distance_map: DistanceMap, agents_hints: dict, np_random: RandomState = None) -> Timetable:
        env_dict = persistence.RailEnvPersister.load_env_dict(filename, load_from_package=load_from_package)
        agents = env_dict["agents"]

        max_episode_steps = env_dict.get("max_episode_steps", 0)
        if max_episode_steps == 0:
            warnings.warn("This env file has no max_episode_steps (deprecated) - setting to 100")
            max_episode_steps = 100
        earliest_departures = [[a.earliest_departure] for a in agents]
        latest_arrivals = [[a.latest_arrival] for a in agents]
        return Timetable(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals, max_episode_steps=max_episode_steps)

    return generator
