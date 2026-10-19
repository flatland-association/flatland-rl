import heapq
import math
from collections import deque
from itertools import count
from typing import List, Generic, Set, TypeVar

from flatland.core.entry_point_distance_map import EntryPointDistanceMap
from flatland.core.transition_map import TransitionMap

EntryPointDistanceMapT = TypeVar('EntryPointDistanceMapT', bound=EntryPointDistanceMap)
TransitionMapT = TypeVar('TransitionMapT', bound=TransitionMap)
EntryPointT = TypeVar('EntryPointT')


class DistanceMapWalker(Generic[EntryPointDistanceMapT, TransitionMapT, EntryPointT]):
    """
    "All-to-any-one-in-cluster": utility class to compute distance maps from each entry point in the rail network (cell and each possible orientation within it in grid case)
     to any one in the set of target entry points using backwards BFS (backwards Dijkstra if the distance map has
     an `edge_length`). Agnostic of any agent/target_nr - operates purely in terms of entry points.
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
        the minimum distances - backward Dijkstra instead if the distance map has an `edge_length`.
        """
        if self.distance_map.edge_length is not None:
            return self._weighted_walk_to_target(rail, target_entry_point)
        self.distance_map._set_distance(target_entry_point, target_entry_point, 0)

        # Fill in the (up to) 4 neighboring nodes
        # direction is the direction of movement, meaning that at least one possible orientation of an agent
        # in cell (row,col) allows a movement in direction `direction'
        nodes_queue = deque(self._get_and_update_neighbors(rail, target_entry_point, 0, target_entry_point))

        # BFS from target `position' to all the reachable nodes in the grid
        # Stop the search if the target position is re-visited, in any direction
        visited = {target_entry_point}

        while nodes_queue:
            entry_point, distance = nodes_queue.popleft()

            if entry_point not in visited:
                visited.add(entry_point)

                # From the list of possible neighbors that have at least a path to the current node, only keep those
                # whose new orientation in the current cell would allow a transition to the entry point
                valid_neighbors = self._get_and_update_neighbors(rail, entry_point, distance, target_entry_point)

                for n in valid_neighbors:
                    nodes_queue.append(n)

        return visited

    def _weighted_walk_to_target(self, rail: TransitionMapT, target_entry_point: EntryPointT
                                 ) -> Set[EntryPointT]:
        """
        Backward Dijkstra from a single target entry point to every entry point that can reach it, filling in
        the minimum sum of edge lengths `edge_length(u, v)` to reach the target.
        """
        edge_length = self.distance_map.edge_length
        # tentative distances of this walk - not the distance map's own, which may still hold a previous walk's
        # distances (e.g. when re-computed), which would prevent the walk from expanding at all.
        tentative = {target_entry_point: 0}
        # tie-breaker so entry points themselves never need to be comparable
        tie_breaker = count()
        queue = [(0, next(tie_breaker), target_entry_point)]
        visited = set()
        while queue:
            distance, _, entry_point = heapq.heappop(queue)
            if entry_point in visited:
                continue
            visited.add(entry_point)
            self.distance_map._set_distance(
                entry_point, target_entry_point, min(self.distance_map._get_distance(entry_point, target_entry_point), distance))
            for predecessor_entry_point in rail.get_predecessor_entry_points(entry_point):
                if predecessor_entry_point in visited:
                    continue
                new_distance = distance + edge_length(predecessor_entry_point, entry_point)
                if new_distance < tentative.get(predecessor_entry_point, math.inf):
                    tentative[predecessor_entry_point] = new_distance
                    heapq.heappush(queue, (new_distance, next(tie_breaker), predecessor_entry_point))
        return visited

    def _get_and_update_neighbors(self, rail: TransitionMapT, entry_point: EntryPointT,
                                  current_distance: int, target_entry_point: EntryPointT):
        """
        Utility function used by _walk_to_target to perform a BFS walk over the rail, filling in the
        minimum distances to a single target entry point.
        """
        neighbors = []
        for predecessor_entry_point in rail.get_predecessor_entry_points(entry_point):
            new_distance = min(
                self.distance_map._get_distance(predecessor_entry_point, target_entry_point),
                current_distance + 1
            )
            neighbors.append((predecessor_entry_point, new_distance))
            self.distance_map._set_distance(predecessor_entry_point, target_entry_point, new_distance)
        return neighbors
