from typing import List, NamedTuple

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid_utils import IntVector2DArray

Schedule = NamedTuple('Schedule', [('agent_positions', IntVector2DArray),
                                   ('agent_directions', List[Grid4TransitionsEnum]),
                                   ('agent_targets', IntVector2DArray),
                                   ('agent_speeds', List[float]),
                                   ('agent_malfunction_rates', List[int]),
                                   ('max_episode_steps', int)])
