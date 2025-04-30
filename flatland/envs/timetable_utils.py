from typing import List, NamedTuple

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import IntVector2DArray, IntVector2DArrayArray

Line = NamedTuple('Line', [
    # positions and directions without target (which has no direction)
    ('agent_positions', IntVector2DArrayArray),
    ('agent_directions', List[List[Grid4TransitionsEnum]]),
    ('agent_targets', IntVector2DArray),
    ('agent_speeds', List[float]),
])

Timetable = NamedTuple('Timetable', [
    # earliest departures and latest arrivals including None for latest arrival at initial and None for earliest departure at target
    ('earliest_departures', List[List[int]]),
    ('latest_arrivals', List[List[int]]),
    ('max_episode_steps', int)
])
