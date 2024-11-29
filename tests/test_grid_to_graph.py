import numpy as np
import pytest

from flatland.core.graph.grid_to_graph import GraphTransitionMap
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.core.transition_map import GridTransitionMap

# rows are top to bottom!
testdata = [
    # Case 0 - empty cell (1)
    (RailEnvTransitionsEnum.empty, []),
    # Case 1 - straight (2)
    (RailEnvTransitionsEnum.horizontal_straight, [
        ((1, 1, Grid4TransitionsEnum.EAST), (1, 2, Grid4TransitionsEnum.EAST)),
        ((1, 1, Grid4TransitionsEnum.WEST), (1, 0, Grid4TransitionsEnum.WEST)),
    ]),
    (RailEnvTransitionsEnum.vertical_straight, [
        ((1, 1, Grid4TransitionsEnum.NORTH), (0, 1, Grid4TransitionsEnum.NORTH)),
        ((1, 1, Grid4TransitionsEnum.SOUTH), (2, 1, Grid4TransitionsEnum.SOUTH)),
    ]),

    # TODO test further cases....

    # Case 6 - symmetrical (4)
    (RailEnvTransitionsEnum.symmetric_switch_south, [
        # E<->S
        ((1, 1, Grid4TransitionsEnum.WEST), (2, 1, Grid4TransitionsEnum.SOUTH)),
        ((1, 1, Grid4TransitionsEnum.NORTH), (1, 2, Grid4TransitionsEnum.EAST)),

        # W<->S
        ((1, 1, Grid4TransitionsEnum.EAST), (2, 1, Grid4TransitionsEnum.SOUTH)),
        ((1, 1, Grid4TransitionsEnum.NORTH), (1, 0, Grid4TransitionsEnum.WEST)),
    ]),

    # Case 7 - dead end (4)
    (RailEnvTransitionsEnum.dead_end_from_north, [((1, 1, Grid4TransitionsEnum.SOUTH), (0, 1, Grid4TransitionsEnum.NORTH))]),
    (RailEnvTransitionsEnum.dead_end_from_south, [((1, 1, Grid4TransitionsEnum.NORTH), (2, 1, Grid4TransitionsEnum.SOUTH))]),
]


@pytest.mark.parametrize("transition,expected_edges", testdata)
def test_grid_to_digraph(transition, expected_edges):
    RailEnvTransitions().print(transition)
    # use 3 x 3 not to go -1
    rail_map = np.array(
        [[RailEnvTransitionsEnum.empty] * 3] +
        [[RailEnvTransitionsEnum.empty, transition, RailEnvTransitionsEnum.empty]] +
        [[RailEnvTransitionsEnum.empty] * 3], dtype=np.uint16)

    gtm = GridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=RailEnvTransitions())
    gtm.grid = rail_map

    g = GraphTransitionMap.grid_to_digraph(gtm)
    print(list(g.edges))
    print(expected_edges)
    print(f"surplus edges: {set(g.edges) - set(expected_edges)}")
    print(f"missing edges: {set(expected_edges) - set(g.edges)}")
    assert len(g.edges) == len(expected_edges)
    assert set(g.edges) == set(expected_edges)
