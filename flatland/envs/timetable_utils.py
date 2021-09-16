from typing import List, NamedTuple

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import IntVector2DArray

Line = NamedTuple('Line',  [('agent_positions', IntVector2DArray),
                            ('agent_directions', List[Grid4TransitionsEnum]),
                            ('agent_targets', IntVector2DArray),
                            ('agent_speeds', List[float])])

Timetable = NamedTuple('Timetable',  [('earliest_departures', List[int]),
                                    ('latest_arrivals', List[int]),
                                    ('max_episode_steps', int)])
