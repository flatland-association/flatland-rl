"""Timetable generators: Railway Undertaking (RU) / Eisenbahnverkehrsunternehmen (EVU)."""
import itertools
import math
import pickle
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
from numpy.random.mtrand import RandomState

from flatland.envs import persistence
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.timetable_utils import Timetable


def _shortest_path_length_or_zero(distance_map: DistanceMap, agent: EnvAgent) -> int:
    length = distance_map.shortest_path_length(agent)
    # whole steps, as timetable entries are steps - rounded up for fractional edge lengths
    return int(math.ceil(length)) if length is not None else 0


def _segment_lengths_through_stops(agents: List[EnvAgent], distance_map: DistanceMap) -> List[List[int]]:
    """
    Segment lengths of each agent's shortest route through its waypoint groups, as `shortest_path_length()` (edge
    lengths plus 1, rounded up to whole steps) per segment.

    The route starts at the initial entry point, serves each intermediate group at one of its entry points and ends at
    the target's position, in any direction. A stop is served at an entry point (position and direction), as the rewards check it, and the
    agent leaves a stop in the direction it arrived in, so the entry points are chosen jointly over the whole route.
    If none of a group's entry points is reachable, the group's positions count in any direction. If none of them is
    reachable either, the segment counts 0, as an unreachable target does.
    """
    rail = distance_map.rail
    # one distance map row per distinct set of entry points to reach (a fake agent targeting them)
    rows = {}
    routes = []
    for a in agents:
        groups = []
        for group in a.waypoints[1:-1]:
            listed = sorted({(wp.position, wp.direction) for wp in group})
            any_direction = sorted({(wp.position, d) for wp in group for d in range(4)
                                    if rail.is_valid_entry_point((wp.position, d))})
            groups.append([[(ep, rows.setdefault(frozenset([ep]), len(rows))) for ep in alternatives]
                           for alternatives in (listed, any_direction)])
        # the last segment ends at the target in any direction, as on the target's position the agent is done
        targets = frozenset((wp.position, d) for wp in a.waypoints[-1] for d in range(4))
        groups.append([[(targets, rows.setdefault(targets, len(rows)))]])
        routes.append(groups)
    fake_agents = [EnvAgent(handle=row, initial_entry_point=None, current_entry_point=None, old_entry_point=None,
                            targets=set(targets)) for targets, row in rows.items()]
    distances = DistanceMap(fake_agents, distance_map.env_height, distance_map.env_width)
    distances.reset(fake_agents, rail)

    segment_lengths = []
    for a, groups in zip(agents, routes):
        # cheapest route to each entry point of the groups so far: (total length, segment lengths)
        best = {(a.waypoints[0][0].position, a.waypoints[0][0].direction): (0, [])}
        for alternatives in groups:
            reached = {}
            for candidates in alternatives:
                reached = _extend(best, candidates, distances)
                if reached:
                    break
            if not reached:
                # nothing of the group reachable: count the segment as 0 from the cheapest route so far
                total, lengths = min(best.values(), key=lambda b: b[0])
                reached = {target: (total, lengths + [0]) for target, _ in alternatives[0]}
            best = reached
        segment_lengths.append(min(best.values(), key=lambda b: b[0])[1])
    return segment_lengths


def _extend(best: dict, candidates: list, distances: DistanceMap) -> dict:
    """The cheapest extension of the routes in `best` to each of the candidates (entry point or target set, row)."""
    reached = {}
    for target, row in candidates:
        for source, (total, lengths) in best.items():
            if source == target or (isinstance(target, frozenset) and source in target):
                length = 1
            else:
                distance = distances.geometric_distance(source, row)
                if distance == math.inf:
                    continue
                length = int(math.ceil(distance + 1))
            if target not in reached or total + length < reached[target][0]:
                reached[target] = (total + length, lengths + [length])
    return reached


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

    A single agent without malfunctions, with the default acceleration_delta=1 and braking_delta=-1, can keep the
    timetable: driving the shortest route through its stops at max speed v, it enters each stop at least
    ceil(L / v) - ceil((L - 1) / v) >= 1 steps before the stop's earliest departure (L: the segment's length, see
    below), exactly 1 step at max speed 1 and n steps at 1/n. It can halt there, start moving again at the earliest
    departure and reach its target by its latest arrival. Not if its route to a stop passes its
    target, where it is done before serving the stop (#557), and not with other acceleration and braking deltas, which
    the timetable generator doesn't get (#587).

    N.B. travel times are a heuristic: each segment (between consecutive waypoints) counts the sum of its edge
    lengths plus 1, as `len(shortest_path)` (see `DistanceMap.shortest_path_length()`), rounded up to whole steps. The
    segments follow the shortest route through the stops, with each stop's entry point (position and direction) chosen
    over the whole route (see `_segment_lengths_through_stops()`). The total, divided by the agent's max speed,
    determines max_episode_steps, the departure window and latest arrival. Each segment's travel time is its length
    divided by the max speed and rounded up on its own, as the agent starts each segment from a halt. The intermediate
    earliest departures add them up from the earliest departure, counted from step 1 for an earliest departure of 0 as
    the agent enters the map in step 1 at the earliest. The intermediate latest arrivals subtract them from the latest
    arrival. Hence the step onto the map is counted per segment and scaled by the speed: the +1 stands for the halt at
    a stop, and gives slow agents some extra slack (#544). See `EnvAgent.get_travel_time_on_shortest_path()` for the
    travel time used for rewards.
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
        shortest_path_segment_lengths = _segment_lengths_through_stops(agents, distance_map)
        shortest_paths_lengths = [sum(l) for l in shortest_path_segment_lengths]
    else:
        shortest_paths_lengths = [_shortest_path_length_or_zero(distance_map, agent) for agent in distance_map.agents]
        shortest_path_segment_lengths = [[l] for l in shortest_paths_lengths]

    # Find mean_shortest_path_time
    agent_speeds = [agent.speed_counter.max_speed for agent in agents]
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
        # intermediate times: each segment's travel time divided by the speed and rounded up on its own, so the agent has
        # the whole steps it needs for each segment from a halt (rounding the sum can leave a segment one step short);
        # exact, as the speed is a Fraction and the segment lengths are whole numbers
        speed = agent_speeds[agent.handle]
        steps = [math.ceil(length / speed) for length in shortest_path_segment_lengths[agent.handle]]
        # an earliest departure of 0 lets the agent enter the map in step 1 only, as 1 does
        start = max(earliest_departure, 1)
        eds = [earliest_departure] + [start + done for done in itertools.accumulate(steps)]
        las = [latest_arrival - sum(steps[i:]) for i in range(len(steps))] + [latest_arrival]
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


def timetable_from_file(filename: Union[str, Path] = None, load_from_package=None, env_dict=None) -> timetable_generator:
    """
    Utility to load timetable generator from persisted env - uses env_dict if populated, otherwise tries to load from file / package.

    Parameters
    ----------
    filename : Union[str, Path]
        pickle file with persisted env, defaults to `None`.
    load_from_package : str
        package, defaults to `None`.
    env_dict: dict
        env_dict, defaults to `None`.

    Returns
    -------
    Tuple[List[Tuple[int,int]], List[Tuple[int,int]], List[Tuple[int,int]], List[float]]
        initial positions, directions, targets speeds
    """

    def generator(agents: List[EnvAgent], distance_map: DistanceMap, agents_hints: dict, np_random: RandomState = None) -> Timetable:
        _env_dict = env_dict
        if _env_dict is None:
            _env_dict = persistence.RailEnvPersister.load_env_dict(filename, load_from_package=load_from_package)
        agents = _env_dict["agents"]

        max_episode_steps = _env_dict.get("max_episode_steps", 0)
        if max_episode_steps == 0:
            warnings.warn("This env file has no max_episode_steps (deprecated) - setting to 100")
            max_episode_steps = 100
        earliest_departures = [a.waypoints_earliest_departure for a in agents]
        latest_arrivals = [a.waypoints_latest_arrival for a in agents]
        return Timetable(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals, max_episode_steps=max_episode_steps)

    return generator
