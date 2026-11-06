from typing import List, Tuple

import networkx as nx

from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap


def greedy_graph_route_through_waypoints(rail: GraphTransitionMap, source: str, waypoint_groups: List[List[str]]) -> Tuple[
    List[str], List[int]]:
    """
    Routes from `source` through `waypoint_groups` in order, one leg per group: per group, the alternative
    nearest to the end of the previous leg (ties broken by the order in the group), then the shortest path
    to it. Every edge counts as length 1.

    Greedy: each leg is fixed once chosen and never revised (no backtracking). So the route may not be the
    shortest one through all groups, and routing may fail at a later group although another choice for an
    earlier group would have reached it.

    Waypoints are exact entry points (graph nodes, i.e. cell and direction), as Flatland's rewards match
    intermediate stops on exact entry points.

    TODO https://github.com/flatland-association/flatland-rl/issues/553 replace the `networkx` routing by `get_k_shortest_paths()` once it is
     generalized to graphs, so grid and graph share one routing.

    Parameters
    ----------
    rail : GraphTransitionMap
        the graph
    source : str
        the entry point to route from
    waypoint_groups : List[List[str]]
        the groups to route through, each a list of alternative entry points

    Returns
    -------
    Tuple[List[str], List[int]]
        The route (starting at `source`) and, per group routed, the index in the route of the alternative
        picked. If a group cannot be reached, routing stops there: the stop indices then hold fewer entries
        than there are groups.
    """
    g = rail.g
    route = [source]
    stop_indices = []
    for group in waypoint_groups:
        distances = nx.single_source_shortest_path_length(g, route[-1])
        reachable = [entry_point for entry_point in group if entry_point in distances]
        if len(reachable) == 0:
            break
        nearest = min(reachable, key=lambda entry_point: distances[entry_point])
        route += nx.shortest_path(g, route[-1], nearest)[1:]
        stop_indices.append(len(route) - 1)
    return route, stop_indices
