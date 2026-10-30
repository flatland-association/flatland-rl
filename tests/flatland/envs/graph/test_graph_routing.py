import networkx as nx

from flatland.envs.graph.graph_routing import greedy_graph_route_through_waypoints
from flatland.envs.graph.rail_graph_transition_map import GraphTransitionMap


def _path_graph() -> GraphTransitionMap:
    """ a -> b -> c -> d, plus an isolated node z. """
    g = nx.DiGraph()
    nx.add_path(g, ["a", "b", "c", "d"])
    g.add_node("z")
    return GraphTransitionMap(g)


def test_greedy_graph_route_through_waypoints():
    assert greedy_graph_route_through_waypoints(_path_graph(), "a", [["c"], ["d"]]) == (["a", "b", "c", "d"], [2, 3])


def test_greedy_graph_route_through_waypoints_source_in_group():
    assert greedy_graph_route_through_waypoints(_path_graph(), "a", [["a"], ["c"]]) == (["a", "b", "c"], [0, 2])


def test_greedy_graph_route_through_waypoints_picks_nearest_alternative():
    assert greedy_graph_route_through_waypoints(_path_graph(), "a", [["d", "b"]]) == (["a", "b"], [1])


def test_greedy_graph_route_through_waypoints_breaks_ties_by_order_in_group():
    g = nx.DiGraph()
    nx.add_path(g, ["a", "x"])
    nx.add_path(g, ["a", "y"])
    assert greedy_graph_route_through_waypoints(GraphTransitionMap(g), "a", [["y", "x"]]) == (["a", "y"], [1])
    assert greedy_graph_route_through_waypoints(GraphTransitionMap(g), "a", [["x", "y"]]) == (["a", "x"], [1])


def test_greedy_graph_route_through_waypoints_stops_at_unreachable_group():
    assert greedy_graph_route_through_waypoints(_path_graph(), "a", [["c"], ["z"], ["d"]]) == (["a", "b", "c"], [2])
