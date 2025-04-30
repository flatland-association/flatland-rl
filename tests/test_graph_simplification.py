import pytest

from flatland.core.graph.graph_simplification import DecisionPointGraph
from flatland.core.graph.grid_to_graph import GraphTransitionMap
from flatland.utils.simple_rail import make_simple_rail, make_disconnected_simple_rail, make_simple_rail2, make_simple_rail_with_alternatives, make_oval_rail


@pytest.mark.parametrize("env_generator,expected_num_vertices,expected_num_edges", [
    (make_simple_rail, 2, 4),
    (make_disconnected_simple_rail, 2, 4),
    (make_simple_rail2, 2, 4),
    (make_simple_rail_with_alternatives, 2, 4),
    (make_oval_rail, 2, 2),
])
def test_simple_rail_simplification(env_generator, expected_num_vertices, expected_num_edges):
    rail, _, _ = env_generator()

    gtm = GraphTransitionMap(GraphTransitionMap.grid_to_digraph(rail))
    dpg = DecisionPointGraph.fromGraphTransitionMap(gtm)

    assert len(dpg.g) == expected_num_vertices
    assert len(dpg.g.edges) == expected_num_edges
