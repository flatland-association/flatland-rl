from collections import deque
from typing import List, Generic, TypeVar

from flatland.core.distance_map import ConfigurationDistanceMap
from flatland.core.transition_map import TransitionMap

UnderlyingDistanceMapType = TypeVar('UnderlyingDistanceMapType', bound=ConfigurationDistanceMap)
UnderlyingTransitionMapType = TypeVar('UnderlyingTransitionMapType', bound=TransitionMap)
UnderlyingConfigurationType = TypeVar('UnderlyingConfigurationType')


class DistanceMapWalker(Generic[UnderlyingDistanceMapType, UnderlyingTransitionMapType, UnderlyingConfigurationType]):
    """
    "All-to-any-one-in-cluster": utility class to compute distance maps from each configuration in the rail network (cell and each possible orientation within it in grid case)
     to any one in the set of target configurations using backwards BFS. Agnostic of any agent/target_nr - operates
     purely in terms of configurations.
    """

    def __init__(self, distance_map: ConfigurationDistanceMap):
        self.distance_map = distance_map

    def _distance_map_walker(self,
                             rail: UnderlyingTransitionMapType,
                             target_configurations: List[UnderlyingConfigurationType]):
        """
        Utility function to compute distance maps from each cell in the rail network (and each possible
        orientation within it) to each of the target configurations.

        Parameters
        ----------
        target_configurations
        """
        # Returns max distance to target, from the farthest away node, while filling in distance_map
        for target_configuration in target_configurations:
            self.distance_map._set_distance(target_configuration, target_configuration, 0)

        # Fill in the (up to) 4 neighboring nodes
        # direction is the direction of movement, meaning that at least one possible orientation of an agent
        # in cell (row,col) allows a movement in direction `direction'
        nodes_queue = deque(
            x
            for target_configuration in target_configurations
            for x in self._get_and_update_neighbors(rail, target_configuration, 0, target_configuration)
        )

        # BFS from target `position' to all the reachable nodes in the grid
        # Stop the search if the target position is re-visited, in any direction
        visited = set(target_configurations)

        max_distance = 0

        while nodes_queue:
            configuration, distance, target_configuration = nodes_queue.popleft()

            if configuration not in visited:
                visited.add(configuration)

                # From the list of possible neighbors that have at least a path to the current node, only keep those
                # whose new orientation in the current cell would allow a transition to the configuration
                valid_neighbors = self._get_and_update_neighbors(rail, configuration, distance, target_configuration)

                for n in valid_neighbors:
                    nodes_queue.append(n)

                if len(valid_neighbors) > 0:
                    max_distance = max(max_distance, distance + 1)

        return max_distance

    def _get_and_update_neighbors(self, rail: UnderlyingTransitionMapType, configuration: UnderlyingConfigurationType,
                                  current_distance: int, target_configuration: UnderlyingConfigurationType):
        """
        Utility function used by _distance_map_walker to perform a BFS walk over the rail, filling in the
        minimum distances from each target cell.
        """
        neighbors = []
        for predecessor_configuration in rail.get_predecessor_configurations(configuration):
            new_distance = min(
                self.distance_map._get_distance(predecessor_configuration, target_configuration),
                current_distance + 1
            )
            neighbors.append((predecessor_configuration, new_distance, target_configuration))
            self.distance_map._set_distance(predecessor_configuration, target_configuration, new_distance)
        return neighbors
