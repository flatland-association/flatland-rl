import pytest

from flatland.core.graph.graph_simplification import DecisionPointGraph
from flatland.core.graph.grid_to_graph import GraphTransitionMap
from flatland.utils.simple_rail import make_simple_rail, make_disconnected_simple_rail, make_simple_rail2, make_simple_rail_with_alternatives, make_oval_rail


@pytest.mark.parametrize("env_creator,expected_num_decision_points", [
    (make_simple_rail, 2),
    (make_disconnected_simple_rail, 2),
    (make_simple_rail2, 2),
    (make_simple_rail_with_alternatives, 2),
    (make_oval_rail, 0),  # TODO what do we want here?
])
def test_simple_rail_simplification(env_creator, expected_num_decision_points):
    rail, _, _ = env_creator()

    gtm = GraphTransitionMap(GraphTransitionMap.grid_to_digraph(rail))
    dpg = DecisionPointGraph.fromGraphTransitionMap(gtm)

    assert len(dpg.g) == expected_num_decision_points
    assert len(dpg.g.edges) == 2 * expected_num_decision_points
