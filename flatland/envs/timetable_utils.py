from typing import List, NamedTuple, Dict

from flatland.envs.rail_trainrun_data_structures import Waypoint

Line = NamedTuple('Line', [
    # positions and directions with flexibility, apart from
    # - initial (which has exactly one waypoint)
    # - target (which has exactly one waypoint with no direction (=`None`))
    ('agent_waypoints', Dict[int, List[List[Waypoint]]]),
    ('agent_speeds', List[float]),
])

Timetable = NamedTuple('Timetable', [
    # earliest departures and latest arrivals including None for latest arrival at initial and None for earliest departure at target
    ('earliest_departures', List[List[int]]),
    ('latest_arrivals', List[List[int]]),
    ('max_episode_steps', int)
])
