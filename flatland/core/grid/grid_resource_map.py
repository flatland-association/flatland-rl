from typing import Tuple, Union, Set, Optional

from flatland.core.grid.grid_utils import Vector2D
from flatland.core.resource_map import ResourceMap


class GridResourceMap(ResourceMap[Tuple[Tuple[int, int], int], Union[Tuple[Tuple[int, int], int], Tuple[int, int]]]):
    def __init__(self, level_free_positions: Set[Vector2D] = None):
        self.level_free_positions = level_free_positions
        if self.level_free_positions is None:
            self.level_free_positions = set()

    def get_resource(self, from_entry_point: Optional[Tuple[Tuple[int, int], int]],
                      to_entry_point: Optional[Tuple[Tuple[int, int], int]]) -> Optional[Tuple[int, int]]:
        if from_entry_point is None:
            return None
        position, direction = from_entry_point
        if position in self.level_free_positions:
            return position, direction % 2
        return position
