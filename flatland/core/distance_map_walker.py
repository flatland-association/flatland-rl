import math
from collections import deque
from typing import Dict, List, Generic, Set, TypeVar

from flatland.core.entry_point_distance_map import EntryPointDistanceMap
from flatland.core.transition_map import TransitionMap

EntryPointDistanceMapT = TypeVar('EntryPointDistanceMapT', bound=EntryPointDistanceMap)
TransitionMapT = TypeVar('TransitionMapT', bound=TransitionMap)
EntryPointT = TypeVar('EntryPointT')


class DistanceMapWalker(Generic[EntryPointDistanceMapT, TransitionMapT, EntryPointT]):
    """
    "All-to-any-one-in-cluster": utility class to compute distance maps from each entry point in the rail network (cell and each possible orientation within it in grid case)
     to any one in the set of target entry points using backwards BFS, re-visiting an entry point whenever a shorter
     distance to it is found (label-correcting, as edges can have different lengths, see `EntryPointDistanceMap.edge_length`).
     Agnostic of any agent/target_nr - operates purely in terms of entry points.
    """

    def __init__(self, distance_map: EntryPointDistanceMap):
        self.distance_map = distance_map

    def _distance_map_walker(self,
                             rail: TransitionMapT,
                             target_entry_points: List[EntryPointT]
                             ) -> Set[EntryPointT]:
        """
        Utility function to compute distance maps from each cell in the rail network (and each possible
        orientation within it) to each of the target entry points. Each target entry point is walked
        independently (its own BFS, its own visited set) - a shared visited set across multiple target
        entry points would incorrectly cut off exploration wherever their searches cross (e.g. on a
        cyclic/looped rail layout).

        N.B. this makes the walk cost O(K * V) instead of O(V) for K target entry points (e.g. up to 4
        headings for one physical target) - an accepted correctness-over-performance tradeoff.

        Parameters
        ----------
        target_entry_points

        Returns
        -------
        Set[EntryPointT]
            the set of all entry points backwards-reachable from any of the target entry points (i.e.
            those a distance was filled in for).
        """
        reachable_entry_points = set()
        for target_entry_point in target_entry_points:
            reachable_entry_points |= self._walk_to_target(rail, target_entry_point)
        return reachable_entry_points

    def _walk_to_target(self, rail: TransitionMapT, target_entry_point: EntryPointT
                        ) -> Set[EntryPointT]:
        """
        Backward BFS from a single target entry point to every entry point that can reach it, filling in
        the minimum distances. An entry point is re-queued whenever a shorter distance to it is found - with unit
        edge lengths, the first distance found is already the minimum, so this is a plain BFS.
        """
        self.distance_map._set_distance(target_entry_point, target_entry_point, 0)

        # entry point -> minimum distance found so far by this walk (not the distance map's own distances, which may
        # still hold a previous walk's, e.g. when re-computed, and would then never be improved upon)
        visited = {target_entry_point: 0}

        # Fill in the (up to) 4 neighboring nodes
        # direction is the direction of movement, meaning that at least one possible orientation of an agent
        # in cell (row,col) allows a movement in direction `direction'
        nodes_queue = deque(self._get_and_update_neighbors(rail, target_entry_point, 0, target_entry_point, visited))

        # BFS from target `position' to all the reachable nodes in the grid
        while nodes_queue:
            entry_point, distance = nodes_queue.popleft()

            # skip if a shorter distance has been found since queued (and the entry point re-queued with it)
            if distance == visited[entry_point]:
                # From the list of possible neighbors that have at least a path to the current node, only keep those
                # whose new orientation in the current cell would allow a transition to the entry point
                valid_neighbors = self._get_and_update_neighbors(rail, entry_point, distance, target_entry_point, visited)

                for n in valid_neighbors:
                    nodes_queue.append(n)

        return set(visited)

    def _get_and_update_neighbors(self, rail: TransitionMapT, entry_point: EntryPointT,
                                  current_distance: int, target_entry_point: EntryPointT, visited: Dict[EntryPointT, int]):
        """
        Utility function used by _walk_to_target to perform a BFS walk over the rail, filling in the
        minimum distances to a single target entry point. Returns the predecessors to which a shorter distance was found.
        """
        neighbors = []
        for predecessor_entry_point in rail.get_predecessor_entry_points(entry_point):
            new_distance = current_distance + self.distance_map.edge_length(predecessor_entry_point, entry_point)
            if new_distance < visited.get(predecessor_entry_point, math.inf):
                visited[predecessor_entry_point] = new_distance
                neighbors.append((predecessor_entry_point, new_distance))
                self.distance_map._set_distance(predecessor_entry_point, target_entry_point, min(
                    self.distance_map._get_distance(predecessor_entry_point, target_entry_point),
                    new_distance
                ))
        return neighbors
