from collections import deque
from typing import Tuple

import numpy as np

from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


class DistanceMapWalker:
    """
    Utility class to compute distance maps from each cell in the rail network (and each possible orientation within it) to each agent's target cell.
    """

    def __init__(self, distance_map: np.ndarray, env_height: int, env_width: int):
        self.distance_map = distance_map
        self.env_height = env_height
        self.env_width = env_width

    def _distance_map_walker(self, rail: RailGridTransitionMap, position: Tuple[int, int], target_nr: int):
        """
        Utility function to compute distance maps from each cell in the rail network (and each possible
        orientation within it) to each agent's target cell.
        """
        # Returns max distance to target, from the farthest away node, while filling in distance_map
        self.distance_map[target_nr, position[0], position[1], :] = 0

        # Fill in the (up to) 4 neighboring nodes
        # direction is the direction of movement, meaning that at least a possible orientation of an agent
        # in cell (row,col) allows a movement in direction `direction'
        nodes_queue = deque(x for xs in (self._get_and_update_neighbors(rail, (position, d), target_nr, 0) for d in range(4)) for x in xs)

        # BFS from target `position' to all the reachable nodes in the grid
        # Stop the search if the target position is re-visited, in any direction
        visited = {(position[0], position[1], 0), (position[0], position[1], 1), (position[0], position[1], 2),
                   (position[0], position[1], 3)}

        max_distance = 0

        while nodes_queue:
            node = nodes_queue.popleft()

            node_id = (node[0], node[1], node[2])

            if node_id not in visited:
                visited.add(node_id)

                # From the list of possible neighbors that have at least a path to the current node, only keep those
                # whose new orientation in the current cell would allow a transition to direction node[2]
                valid_neighbors = self._get_and_update_neighbors(rail, ((node[0], node[1]), node[2]), target_nr, node[3])

                for n in valid_neighbors:
                    nodes_queue.append(n)

                if len(valid_neighbors) > 0:
                    max_distance = max(max_distance, node[3] + 1)

        return max_distance

    def _get_and_update_neighbors(self, rail: RailGridTransitionMap, configuration: Tuple[Tuple[int, int], int], target_nr: int, current_distance: int):
        """
        Utility function used by _distance_map_walker to perform a BFS walk over the rail, filling in the
        minimum distances from each target cell.
        """
        neighbors = []
        for n in rail.get_predecessor_configurations(configuration):
            new_cell, agent_orientation = n
            new_distance = min(self.distance_map[target_nr, new_cell[0], new_cell[1], agent_orientation], current_distance + 1)
            neighbors.append((new_cell[0], new_cell[1], agent_orientation, new_distance))
            self.distance_map[target_nr, new_cell[0], new_cell[1], agent_orientation] = new_distance
        return neighbors
