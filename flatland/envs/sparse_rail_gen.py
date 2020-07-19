"""Rail generators (infrastructure manager, "Infrastrukturbetreiber")."""
import sys
import warnings
from typing import Callable, Tuple, Optional, Dict, List

import numpy as np
from numpy.random.mtrand import RandomState

from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.grid4_utils import get_direction, mirror, direction_to_point
from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.grid.grid_utils import distance_on_rail, IntVector2DArray, IntVector2D, \
    Vec2dOperations
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail_in_grid_map, connect_straight_line_in_grid_map, \
    fix_inner_nodes, align_cell_to_city
from flatland.envs import persistence

from flatland.envs.rail_generators import RailGeneratorProduct, RailGenerator

