import numpy as np
import pytest

from flatland.core.graph.grid_to_graph import GraphTransitionMap
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.core.grid.rail_env_grid import RailEnvTransitions, RailEnvTransitionsEnum
from flatland.core.transition_map import GridTransitionMap

# rows are top to bottom!
_EAST_OWN = (1, 1, Grid4TransitionsEnum.EAST)  # at "W" side, heading "E"
_EAST_NEIGHBOR = (1, 2, Grid4TransitionsEnum.EAST)  # to the "E", heading "E"
_WEST_OWN = (1, 1, Grid4TransitionsEnum.WEST)  # at "E" side, heading "W"
_WEST_NEIGHBOR = (1, 0, Grid4TransitionsEnum.WEST)  # to the "W", heading "W"
_NORTH_OWN = (1, 1, Grid4TransitionsEnum.NORTH)  # at "S" side, heading "N"
_NORTH_NEIGHBOR = (0, 1, Grid4TransitionsEnum.NORTH)  # at "N" side, heading "N"
_SOUTH_OWN = (1, 1, Grid4TransitionsEnum.SOUTH)  # at "N" side, heading "S"
_SOUTH_NEIGHBOR = (2, 1, Grid4TransitionsEnum.SOUTH)  # to the "S", heading "S"


@pytest.mark.parametrize("transition,expected_edges", [
    # Case 0 - empty cell (1)
    (RailEnvTransitionsEnum.empty, []),

    # Case 1 - straight (2)
    (RailEnvTransitionsEnum.horizontal_straight, [
        (_EAST_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _WEST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.vertical_straight, [
        (_NORTH_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _SOUTH_NEIGHBOR),
    ]),

    # Case 2 - simple switch left (4)
    (RailEnvTransitionsEnum.simple_switch_west_left, [
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_WEST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _EAST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_north_left, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_NORTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _SOUTH_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_east_left, [
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_EAST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _WEST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_south_left, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_SOUTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _NORTH_NEIGHBOR),
    ]),

    # Case 3 - diamond crossing (1)
    (RailEnvTransitionsEnum.diamond_crossing, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
    ]),

    # Case 4 - single slip (4)
    (RailEnvTransitionsEnum.single_slip_SW, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.single_slip_NW, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_SOUTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _NORTH_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.single_slip_NE, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_WEST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _EAST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.single_slip_SE, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_NORTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _SOUTH_NEIGHBOR),
    ]),

    # Case 5 - double slip (2)
    (RailEnvTransitionsEnum.double_slip_NW_SE, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_SOUTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _NORTH_NEIGHBOR),
        (_NORTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _SOUTH_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.double_slip_NE_SW, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_WEST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _EAST_NEIGHBOR),
        (_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR),
    ]),

    # Case 6 - symmetrical switch (4)
    (RailEnvTransitionsEnum.symmetric_switch_from_north, [
        (_SOUTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _NORTH_NEIGHBOR),
        (_SOUTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _NORTH_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.symmetric_switch_from_east, [
        (_WEST_OWN, _NORTH_NEIGHBOR), (_NORTH_OWN, _EAST_NEIGHBOR),
        (_WEST_OWN, _SOUTH_NEIGHBOR), (_SOUTH_OWN, _EAST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.symmetric_switch_from_south, [
        (_WEST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _EAST_NEIGHBOR),
        (_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.symmetric_switch_from_west, [
        (_EAST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _WEST_NEIGHBOR),
        (_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR),
    ]),

    # Case 7 - dead end (4)
    (RailEnvTransitionsEnum.dead_end_from_north, [(_SOUTH_OWN, _NORTH_NEIGHBOR)]),
    (RailEnvTransitionsEnum.dead_end_from_south, [(_NORTH_OWN, _SOUTH_NEIGHBOR)]),
    (RailEnvTransitionsEnum.dead_end_from_west, [(_EAST_OWN, _WEST_NEIGHBOR)]),
    (RailEnvTransitionsEnum.dead_end_from_east, [(_WEST_OWN, _EAST_NEIGHBOR)]),

    # Case 1b (8)  - simple turn right (4)
    # Case 1c (9)  - simple turn left (- same as Case 1b)
    (RailEnvTransitionsEnum.right_turn_from_north, [(_SOUTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _NORTH_NEIGHBOR)]),
    (RailEnvTransitionsEnum.right_turn_from_east, [(_WEST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _EAST_NEIGHBOR)]),
    (RailEnvTransitionsEnum.right_turn_from_south, [(_NORTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _SOUTH_NEIGHBOR)]),
    (RailEnvTransitionsEnum.right_turn_from_west, [(_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR)]),

    # Case 2b (10) - simple switch right (4)
    (RailEnvTransitionsEnum.simple_switch_west_right, [
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_WEST_OWN, _NORTH_NEIGHBOR), (_SOUTH_OWN, _EAST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_north_right, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_NORTH_OWN, _EAST_NEIGHBOR), (_WEST_OWN, _SOUTH_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_east_right, [
        (_WEST_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _EAST_NEIGHBOR),
        (_EAST_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _WEST_NEIGHBOR),
    ]),
    (RailEnvTransitionsEnum.simple_switch_south_right, [
        (_SOUTH_OWN, _SOUTH_NEIGHBOR), (_NORTH_OWN, _NORTH_NEIGHBOR),
        (_SOUTH_OWN, _WEST_NEIGHBOR), (_EAST_OWN, _NORTH_NEIGHBOR),
    ]),
])
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
