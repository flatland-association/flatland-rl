from enum import IntEnum

import numpy as np
import pytest

from flatland.core.graph.grid_to_graph import GraphTransitionMap
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap


# #30 cases in total
class RailEnvTransitionsEnum(IntEnum):
    # Case 0 - empty cell (1)
    empty = RailEnvTransitions().transition_list[0]

    # Case 1 - straight (2)
    vertical_straight = RailEnvTransitions().transition_list[1]
    horizontal_straight = RailEnvTransitions().rotate_transition(RailEnvTransitions().transition_list[1], 90)

    # Case 2 - simple switch left (4)
    simple_switch_north_left = RailEnvTransitions().transition_list[2]

    # Case 3 - diamond crossing (1)
    # Case 4 - single slip (4)
    # Case 5 - double slip (2)

    # Case 6 - symmetrical (4)
    #   NESW
    # N 0101
    # E 0010
    # S 0000
    # W 0010
    symmetric_switch_south = RailEnvTransitions().transition_list[6]
    symmetric_switch_north = RailEnvTransitions().rotate_transition(symmetric_switch_south, 180)

    # Case 7 - dead end (4)
    dead_end_from_south = RailEnvTransitions().transition_list[7]
    dead_end_from_west = RailEnvTransitions().rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = RailEnvTransitions().rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = RailEnvTransitions().rotate_transition(dead_end_from_south, 270)

    # Case 1b/1c (8)/(9)  - simple turn  (4)
    right_turn_from_south = RailEnvTransitions().transition_list[8]

    right_turn_from_west = RailEnvTransitions().rotate_transition(right_turn_from_south, 90)
    right_turn_from_north = RailEnvTransitions().rotate_transition(right_turn_from_south, 180)

    # Case 2b (10) - simple switch right (4)
    simple_switch_north_right = RailEnvTransitions().transition_list[10]
    simple_switch_left_east = RailEnvTransitions().rotate_transition(simple_switch_north_left, 90)





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
