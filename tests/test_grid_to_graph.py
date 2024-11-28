import numpy as np
import pytest

from flatland.core.graph.grid_to_graph import GraphTransitionMap
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap

# TODO make enum
transitions = RailEnvTransitions()
cells = transitions.transition_list

# #30 cases in total

# Case 0 - empty cell (1)
empty = cells[0]

# Case 1 - straight (2)
vertical_straight = cells[1]
horizontal_straight = transitions.rotate_transition(vertical_straight, 90)

# Case 2 - simple switch left (4)
simple_switch_north_left = cells[2]

# Case 3 - diamond crossing (1)
# Case 4 - single slip (4)
# Case 5 - double slip (2)

# Case 6 - symmetrical (4)
double_switch_south_horizontal_straight = horizontal_straight + cells[6]
double_switch_north_horizontal_straight = transitions.rotate_transition(double_switch_south_horizontal_straight, 180)

# Case 7 - dead end (4)
dead_end_from_south = cells[7]
dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)

# Case 1b/1c (8)/(9)  - simple turn  (4)
right_turn_from_south = cells[8]

right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)

# Case 2b (10) - simple switch right (4)
simple_switch_north_right = cells[10]
simple_switch_left_east = transitions.rotate_transition(simple_switch_north_left, 90)

# rows are top to bottom!
testdata = [
    # Case 0 - empty cell (1)
    (empty, []),
    # Case 1 - straight (2)
    (horizontal_straight, [
        ((1, 1, Grid4TransitionsEnum.EAST), (1, 2, Grid4TransitionsEnum.EAST)),
        ((1, 1, Grid4TransitionsEnum.WEST), (1, 0, Grid4TransitionsEnum.WEST))
    ]),
    (vertical_straight, [
        ((1, 1, Grid4TransitionsEnum.NORTH), (0, 1, Grid4TransitionsEnum.NORTH)),
        ((1, 1, Grid4TransitionsEnum.SOUTH), (2, 1, Grid4TransitionsEnum.SOUTH))
    ]),

    # TODO test further cases....

    # Case 7 - dead end (4)
    (dead_end_from_north, [((1, 1, Grid4TransitionsEnum.SOUTH), (0, 1, Grid4TransitionsEnum.NORTH))]),
    (dead_end_from_south, [((1, 1, Grid4TransitionsEnum.NORTH), (2, 1, Grid4TransitionsEnum.SOUTH))]),
]


@pytest.mark.parametrize("transition,expected_edges", testdata)
def test_grid_to_digraph(transition, expected_edges):
    transitions.print(transition)
    # use 3 x 3 not to go -1
    rail_map = np.array(
        [[empty] * 3] +
        [[empty, transition, empty]] +
        [[empty] * 3], dtype=np.uint16)

    gtm = GridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
    gtm.grid = rail_map

    g = GraphTransitionMap.grid_to_digraph(gtm)
    print(list(g.edges))
    print(expected_edges)
    assert set(g.edges) == set(expected_edges)

# TODO unit tests for  simplification
