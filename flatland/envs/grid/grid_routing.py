from typing import List, Tuple

from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import _sanitize_entry_point
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths

GridEntryPoint = Tuple[Tuple[int, int], int]


def greedy_grid_route_through_waypoints(rail: GridTransitionMap, source: GridEntryPoint, waypoint_groups: List[List[GridEntryPoint]],
                                        nearest_alternative: bool = False) -> Tuple[List[GridEntryPoint], List[int]]:
    """
    Routes from `source` through `waypoint_groups` in order, one leg per group: per group, the first reachable
    alternative (default, the previous behaviour of `ShortestPathPolicy`) or, with `nearest_alternative`, the
    alternative nearest to the end of the previous leg (ties broken by the order in the group), as
    `greedy_graph_route_through_waypoints()` does; then the shortest path to it (`get_k_shortest_paths()`).

    Greedy: each leg is fixed once chosen and never revised (no backtracking). So the route may not be the
    shortest one through all groups, and routing may fail at a later group although another choice for an
    earlier group would have reached it.

    Parameters
    ----------
    rail : GridTransitionMap
        the grid
    source : GridEntryPoint
        the entry point `((row, column), direction)` to route from
    waypoint_groups : List[List[GridEntryPoint]]
        the groups to route through, each a list of alternative entry points
    nearest_alternative : bool
        pick the nearest alternative per group instead of the first reachable one - `ShortestPathPolicy`'s default
        router sets it with `serve_intermediate_stops`

    Returns
    -------
    Tuple[List[GridEntryPoint], List[int]]
        The route (starting at `source`) and, per group routed, the index in the route of the alternative
        picked. If a group cannot be reached, routing stops there: the stop indices then hold fewer entries
        than there are groups.
    """
    route = [source]
    stop_indices = []
    for group in waypoint_groups:
        (row, column), direction = route[-1]
        legs = []
        for (target_row, target_column), target_direction in group:
            paths = get_k_shortest_paths(None, (row, column), direction, (target_row, target_column), target_direction=target_direction, rail=rail)
            if len(paths) > 0:
                legs.append(paths[0])
                if not nearest_alternative:
                    break
        if len(legs) == 0:
            break
        leg = min(legs, key=len)
        route += [_sanitize_entry_point(waypoint._to_tuple()) for waypoint in leg[1:]]
        stop_indices.append(len(route) - 1)
    return route, stop_indices
