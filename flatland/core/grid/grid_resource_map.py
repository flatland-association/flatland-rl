from typing import Tuple, Union, Set

from flatland.core.grid.grid_utils import Vector2D
from flatland.core.resource_map import ResourceMap


class GridResourceMap(ResourceMap[Tuple[Tuple[int, int], int], Union[Tuple[Tuple[int, int], int], Tuple[int, int]]]):
    def __init__(self, level_free_positions: Set[Vector2D] = None):
        self.level_free_positions = level_free_positions
        if self.level_free_positions is None:
            self.level_free_positions = set()

    def get_resource(self, configuration):
        # TODO replace with None instead of tuple
        if configuration[0] is None:
            return None
        position, direction = configuration
        if position in self.level_free_positions:
            return position, direction % 2
        return position
