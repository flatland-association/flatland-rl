from flatland.envs.grid.grid_routing import greedy_grid_route_through_waypoints
from tests.test_flatland_envs_rail_env_policies import A, B, C, _corridor_rail


def test_greedy_grid_route_through_waypoints():
    route, stop_indices = greedy_grid_route_through_waypoints(_corridor_rail(), A, [[B], [C]])
    assert route == [A, ((0, 2), 1), ((0, 3), 1), B, ((0, 5), 1), ((0, 6), 1), C]
    assert stop_indices == [3, 6]


def test_greedy_grid_route_through_waypoints_picks_first_reachable_alternative_by_default():
    unreachable = ((0, 4), 0)  # northbound, no such entry point on the corridor
    route, stop_indices = greedy_grid_route_through_waypoints(_corridor_rail(), A, [[unreachable, C, B]])
    assert route == [A, ((0, 2), 1), ((0, 3), 1), B, ((0, 5), 1), ((0, 6), 1), C]
    assert stop_indices == [6]


def test_greedy_grid_route_through_waypoints_picks_nearest_reachable_alternative():
    unreachable = ((0, 4), 0)  # northbound, no such entry point on the corridor
    route, stop_indices = greedy_grid_route_through_waypoints(_corridor_rail(), A, [[unreachable, C, B]], nearest_alternative=True)
    assert route == [A, ((0, 2), 1), ((0, 3), 1), B]
    assert stop_indices == [3]


def test_greedy_grid_route_through_waypoints_stops_at_unreachable_group():
    unreachable = ((0, 4), 0)  # northbound, no such entry point on the corridor
    route, stop_indices = greedy_grid_route_through_waypoints(_corridor_rail(), A, [[B], [unreachable], [C]])
    assert route == [A, ((0, 2), 1), ((0, 3), 1), B]
    assert stop_indices == [3]
