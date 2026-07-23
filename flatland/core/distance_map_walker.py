from collections import deque
from typing import List, Generic, Set, TypeVar

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
                             target_configurations: List[UnderlyingConfigurationType]
                             ) -> Set[UnderlyingConfigurationType]:
        """
        Utility function to compute distance maps from each cell in the rail network (and each possible
        orientation within it) to each of the target configurations. Each target configuration is walked
        independently (its own BFS, its own visited set) - a shared visited set across multiple target
        configurations would incorrectly cut off exploration wherever their searches cross (e.g. on a
        cyclic/looped rail layout).

        N.B. this makes the walk cost O(K * V) instead of O(V) for K target configurations (e.g. up to 4
        headings for one physical target) - an accepted correctness-over-performance tradeoff.

        Parameters
        ----------
        target_configurations

        Returns
        -------
        Set[UnderlyingConfigurationType]
            the set of all configurations backwards-reachable from any of the target configurations (i.e.
            those a distance was filled in for).
        """
        reachable_configurations = set()
        for target_configuration in target_configurations:
            reachable_configurations |= self._walk_to_target(rail, target_configuration)
        return reachable_configurations

    def _walk_to_target(self, rail: UnderlyingTransitionMapType, target_configuration: UnderlyingConfigurationType
                        ) -> Set[UnderlyingConfigurationType]:
        """
        Backward BFS from a single target configuration to every configuration that can reach it, filling in
        the minimum distances.
        """
        self.distance_map._set_distance(target_configuration, target_configuration, 0)

        # Fill in the (up to) 4 neighboring nodes
        # direction is the direction of movement, meaning that at least one possible orientation of an agent
        # in cell (row,col) allows a movement in direction `direction'
        nodes_queue = deque(self._get_and_update_neighbors(rail, target_configuration, 0, target_configuration))

        # BFS from target `position' to all the reachable nodes in the grid
        # Stop the search if the target position is re-visited, in any direction
        visited = {target_configuration}

        while nodes_queue:
            configuration, distance = nodes_queue.popleft()

            if configuration not in visited:
                visited.add(configuration)

                # From the list of possible neighbors that have at least a path to the current node, only keep those
                # whose new orientation in the current cell would allow a transition to the configuration
                valid_neighbors = self._get_and_update_neighbors(rail, configuration, distance, target_configuration)

                for n in valid_neighbors:
                    nodes_queue.append(n)

        return visited

    def _get_and_update_neighbors(self, rail: UnderlyingTransitionMapType, configuration: UnderlyingConfigurationType,
                                  current_distance: int, target_configuration: UnderlyingConfigurationType):
        """
        Utility function used by _walk_to_target to perform a BFS walk over the rail, filling in the
        minimum distances to a single target configuration.
        """
        neighbors = []
        for predecessor_configuration in rail.get_predecessor_configurations(configuration):
            new_distance = min(
                self.distance_map._get_distance(predecessor_configuration, target_configuration),
                current_distance + 1
            )
            neighbors.append((predecessor_configuration, new_distance))
            self.distance_map._set_distance(predecessor_configuration, target_configuration, new_distance)
        return neighbors
