from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.utils.ordered_set import OrderedSet


def get_k_shortest_paths(env,
                         source_position: Tuple[int, int],
                         source_direction: int,
                         target_position=Tuple[int, int],
                         k: int = 1, debug=False) -> List[Tuple[Waypoint]]:
    """
    Computes the k shortest paths using modified Dijkstra
    following pseudo-code https://en.wikipedia.org/wiki/K_shortest_path_routing
    In contrast to the pseudo-code in wikipedia, we do not a allow for loopy paths.

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

    Returns
    -------
    List[Tuple[WalkingElement]]
        We use tuples since we need the path elements to be hashable.
        We use a list of paths in order to keep the order of length.
    """

    # P: set of shortest paths from s to t
    # P =empty,
    shortest_paths: List[Tuple[Waypoint]] = []

    # countu: number of shortest paths found to node u
    # countu = 0, for all u in V
    count = {(r, c, d): 0 for r in range(env.height) for c in range(env.width) for d in range(4)}

    # B is a heap data structure containing paths
    # N.B. use OrderedSet to make result deterministic!
    heap: OrderedSet[Tuple[Waypoint]] = OrderedSet()

    # insert path Ps = {s} into B with cost 0
    heap.add((Waypoint(source_position, source_direction),))

    # while B is not empty and countt < K:
    while len(heap) > 0 and len(shortest_paths) < k:
        if debug:
            print("iteration heap={}, shortest_paths={}".format(heap, shortest_paths))
        # – let Pu be the shortest cost path in B with cost C
        cost = np.inf
        pu = None
        for path in heap:
            if len(path) < cost:
                pu = path
                cost = len(path)
        u: Waypoint = pu[-1]
        if debug:
            print("  looking at pu={}".format(pu))

        #     – B = B − {Pu }
        heap.remove(pu)
        #     – countu = countu + 1

        urcd = (*u.position, u.direction)
        count[urcd] += 1

        # – if u = t then P = P U {Pu}
        if u.position == target_position:
            if debug:
                print(" found of length {} {}".format(len(pu), pu))
            shortest_paths.append(pu)

        # – if countu ≤ K then
        # CAVEAT: do not allow for loopy paths
        elif count[urcd] <= k:
            possible_transitions = env.rail.get_transitions(*urcd)
            if debug:
                print("  looking at neighbors of u={}, transitions are {}".format(u, possible_transitions))
            #     for each vertex v adjacent to u:
            for new_direction in range(4):
                if debug:
                    print("        looking at new_direction={}".format(new_direction))
                if possible_transitions[new_direction]:
                    new_position = get_new_position(u.position, new_direction)
                    if debug:
                        print("        looking at neighbor v={}".format((*new_position, new_direction)))

                    v = Waypoint(position=new_position, direction=new_direction)
                    # CAVEAT: do not allow for loopy paths
                    if v in pu:
                        continue

                    # – let Pv be a new path with cost C + w(u, v) formed by concatenating edge (u, v) to path Pu
                    pv = pu + (v,)
                    #     – insert Pv into B
                    heap.add(pv)

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
