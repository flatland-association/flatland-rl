from collections import defaultdict, deque
from typing import List, Set, Tuple

import cython
import matplotlib.pyplot as plt
import numpy as np

from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid.distance_map import DistanceMap
from flatland.envs.rail_trainrun_data_structures import Waypoint

# mirrors flatland.core.grid.grid4_utils.MOVEMENT_ARRAY ([(-1,0),(0,1),(1,0),(0,-1)] for N/E/S/W), split into two
# flat int tuples so _k_shortest_paths_search can index row/col deltas directly instead of unpacking a tuple-of-tuples
MOVEMENT_ROW = (-1, 0, 1, 0)
MOVEMENT_COL = (0, 1, 0, -1)


def _k_shortest_paths_search(rail_grid, height, width, k, debug,
                              target_row, target_col, target_direction, cutoff,
                              forbidden_mask, shortest_paths, count, heap):
    """
    Modified-Dijkstra search loop extracted from `get_k_shortest_paths` so it can be cythonized in isolation
    (see the accompanying rail_env_shortest_paths.pxd) - mutates `shortest_paths`, `count` and `heap` in place.
    """
    cost: cython.int
    row: cython.int
    col: cython.int
    direction: cython.int
    idx: cython.int
    cell_transition: cython.int
    nesw: cython.int
    new_direction: cython.int
    new_row: cython.int
    new_col: cython.int

    # while B is not empty and countt < K:
    while heap and len(shortest_paths) < k:
        if debug:
            print("iteration heap={}, shortest_paths={}".format(heap, shortest_paths))
        # – let Pu be the shortest cost path in B with cost C
        cost = min(heap)
        pu = heap[cost].popleft()
        if not heap[cost]:
            del heap[cost]
        u: Waypoint = pu[-1]
        if debug:
            print("  looking at pu={}".format(pu))

        row, col = u.position
        direction = u.direction
        #     – countu = countu + 1
        idx = ((row * width) + col) * 4 + direction
        count[idx] += 1

        # – if u = t then P = P U {Pu}
        if row == target_row and col == target_col:
            if target_direction == -1 or target_direction == direction:
                if debug:
                    print(" found of length {} {}".format(len(pu), pu))
                shortest_paths.append(pu)

        # – if countu ≤ K then
        # CAVEAT: do not allow for loopy paths
        elif count[idx] <= k:
            cell_transition = rail_grid[row, col]
            nesw = (cell_transition >> ((3 - direction) * 4)) & 0xF
            if debug:
                print("  looking at neighbors of u={}, nesw={:04b}".format(u, nesw))
            #     for each vertex v adjacent to u:
            for new_direction in range(4):
                if debug:
                    print("        looking at new_direction={}".format(new_direction))
                if (nesw >> (3 - new_direction)) & 1:
                    new_row = row + MOVEMENT_ROW[new_direction]
                    new_col = col + MOVEMENT_COL[new_direction]
                    if debug:
                        print("        looking at neighbor v={}".format((new_row, new_col, new_direction)))

                    v = Waypoint(position=(new_row, new_col), direction=new_direction)
                    # CAVEAT: do not allow for loopy paths
                    if v in pu:
                        continue

                    # – let Pv be a new path with cost C + w(u, v) formed by concatenating edge (u, v) to path Pu
                    pv = pu + (v,)

                    # ignore if cutoff reached
                    if cutoff != -1 and len(pv) > cutoff:
                        if debug:
                            print(f"        ignoring v={v} as out cutoff {cutoff} reached.")
                        continue
                    # ignore if out of bounds
                    if new_row >= height or new_row < 0 or new_col >= width or new_col < 0:
                        if debug:
                            print(f"        ignoring v={v} as out out bounds ({height, width}).")
                        continue
                    # ignore if in forbidden_cells
                    if forbidden_mask[new_row, new_col]:
                        if debug:
                            print(f"        ignoring v={v} as in forbidden_cells.")
                        continue
                    #     – insert Pv into B
                    heap[len(pv)].append(pv)


def get_k_shortest_paths(env: "RailEnv",
                         source_position: Tuple[int, int],
                         source_direction: int,
                         target_position=Tuple[int, int],
                         k: int = 1, debug=False,
                         target_direction: int = None,
                         rail: GridTransitionMap = None,
                         cutoff: int = None,
                         forbidden_cells: Set[Tuple[int, int]] = None,
                         ) -> List[Tuple[Waypoint]]:
    """
    Computes the k shortest paths using modified Dijkstra
    following pseudo-code https://en.wikipedia.org/wiki/K_shortest_path_routing
    In contrast to the pseudo-code in wikipedia, we do not a allow for loopy paths.

    A `cutoff` can be defined optionally to limit search.
    If the grid is not closed under transitions, then paths going out of the grid are ignored.

    Parameters
    ----------
    env :             RailEnv
    source_position:  Tuple[int,int]
    source_direction: int
    target_position:  Tuple[int,int]
    k :               int
        max number of shortest paths
    debug:            bool
        print debug statements
    target_direction: Optional[Tuple[int,int]]
    cutoff :          Optional[int]
        do not consider paths longer than cutoff
    forbidden_cells : Optional[Set[Tuple[int, int]]]
        cells to exclude from the search - paths are never expanded into one of these cells
    Returns
    -------
    List[Tuple[WalkingElement]]
        We use tuples since we need the path elements to be hashable.
        We use a list of paths in order to keep the order of length.
    """
    if env is not None:
        rail = env.rail
    else:
        assert rail is not None
    height = rail.height
    width = rail.width

    # P: set of shortest paths from s to t
    # P =empty,
    shortest_paths: List[Tuple[Waypoint]] = []

    # countu: number of shortest paths found to node u, for all u in V - as a flat C-int array indexed by
    # (row, col, direction) instead of a dict keyed by tuple
    count = np.zeros(height * width * 4, dtype=np.intc)

    # forbidden_cells as a fixed (height, width) mask, built once - never Optional inside the hot loop
    forbidden_mask = np.zeros((height, width), dtype=np.uint8)
    if forbidden_cells is not None:
        for (fr, fc) in forbidden_cells:
            forbidden_mask[fr, fc] = 1

    # B is a heap data structure containing paths, bucketed by path length: Dict[int, deque[Tuple[Waypoint]]]
    # N.B. use deque per bucket to make result deterministic (insertion order == retrieval order); OrderedSet's
    # de-dup isn't needed here since a given path can never be enqueued into the same bucket twice (each pu is
    # removed from its bucket before being expanded, and a single expansion's 4 new_directions always produce
    # distinct Waypoints since direction alone differs)
    # NOT annotated `: Dict[...]` - in compiled mode Cython enforces that as an exact-`dict` type check, which
    # rejects `defaultdict` (a dict subclass) at assignment ("Expected dict, got collections.defaultdict")
    heap = defaultdict(deque)

    # insert path Ps = {s} into B with cost 0
    heap[1].append((Waypoint(source_position, source_direction),))

    target_row, target_col = target_position
    _k_shortest_paths_search(
        rail.grid, height, width, k, debug,
        target_row, target_col,
        -1 if target_direction is None else target_direction,
        -1 if cutoff is None else cutoff,
        forbidden_mask, shortest_paths, count, heap,
    )

    # return P
    return shortest_paths


def visualize_distance_map(distance_map: DistanceMap, agent_handle: int = 0):
    if agent_handle >= distance_map.get().shape[0]:
        print("Error: agent_handle cannot be larger than actual number of agents")
        return
    # take min value of all 4 directions
    min_distance_map = np.min(distance_map.get(), axis=3)
    plt.imshow(min_distance_map[agent_handle][:][:])
    plt.show()
