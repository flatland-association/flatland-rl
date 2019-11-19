import numpy as np

from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail_in_grid_map, connect_straight_line_in_grid_map, \
    fix_inner_nodes


def test_build_railway_infrastructure():
    rail_trans = RailEnvTransitions()
    grid_map = GridTransitionMap(width=20, height=20, transitions=rail_trans)
    grid_map.grid.fill(0)

    # Make connection with dead-ends on both sides
    start_point = (2, 2)
    end_point = (8, 8)
    connection_001 = connect_rail_in_grid_map(grid_map, start_point, end_point, rail_trans, flip_start_node_trans=True,
                                              flip_end_node_trans=True, respect_transition_validity=True,
                                              forbidden_cells=None)
    connection_001_expected = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8),
                               (7, 8), (8, 8)]

    # Make connection with open ends on both sides
    start_point = (1, 3)
    end_point = (1, 7)
    connection_002 = connect_rail_in_grid_map(grid_map, start_point, end_point, rail_trans, flip_start_node_trans=False,
                                              flip_end_node_trans=False, respect_transition_validity=True,
                                              forbidden_cells=None)
    connection_002_expected = [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]

    # Make connection with open end at beginning and dead end on end
    start_point = (6, 2)
    end_point = (6, 5)
    connection_003 = connect_rail_in_grid_map(grid_map, start_point, end_point, rail_trans, flip_start_node_trans=False,
                                              flip_end_node_trans=True, respect_transition_validity=True,
                                              forbidden_cells=None)
    connection_003_expected = [(6, 2), (6, 3), (6, 4), (6, 5)]

    # Make connection with dead end on start and opend end
    start_point = (7, 5)
    end_point = (8, 9)
    connection_004 = connect_rail_in_grid_map(grid_map, start_point, end_point, rail_trans, flip_start_node_trans=True,
                                              flip_end_node_trans=False, respect_transition_validity=True,
                                              forbidden_cells=None)
    connection_004_expected = [(7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (8, 9)]

    assert connection_001 == connection_001_expected, \
        "actual={}, expected={}".format(connection_001, connection_001_expected)
    assert connection_002 == connection_002_expected, \
        "actual={}, expected={}".format(connection_002, connection_002_expected)
    assert connection_003 == connection_003_expected, \
        "actual={}, expected={}".format(connection_003, connection_003_expected)
    assert connection_004 == connection_004_expected, \
        "actual={}, expected={}".format(connection_004, connection_004_expected)

    grid_map_grid_expected = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1025, 1025, 1025, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 1025, 1025, 1025, 1025, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1025, 1025, 256, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 1025, 1025, 33825, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    for i in range(len(grid_map_grid_expected)):
        assert np.all(grid_map.grid[i] == grid_map_grid_expected[i])


def test_fix_inner_nodes():
    rail_trans = RailEnvTransitions()
    grid_map = GridTransitionMap(width=6, height=10, transitions=rail_trans)
    grid_map.grid.fill(0)

    start = (2, 2)
    target = (8, 2)
    parallel_start = (3, 3)
    parallel_target = (7, 3)
    parallel_start_1 = (4, 4)
    parallel_target_1 = (6, 4)

    inner_nodes = [start, target, parallel_start, parallel_target, parallel_start_1, parallel_target_1]
    track_0 = connect_straight_line_in_grid_map(grid_map, start, target, rail_trans)
    track_1 = connect_straight_line_in_grid_map(grid_map, parallel_start, parallel_target, rail_trans)
    track_2 = connect_straight_line_in_grid_map(grid_map, parallel_start_1, parallel_target_1, rail_trans)

    # Fix the ends of the inner node
    # This is not a fix in transition type but rather makes the necessary connections to the parallel tracks

    for node in inner_nodes:
        fix_inner_nodes(grid_map, node, rail_trans)

    def orienation(pos):
        if pos[0] < grid_map.grid.shape[0] / 2:
            return 2
        else:
            return 0

    # Fix all the different transitions to legal elements
    for c in range(grid_map.grid.shape[1]):
        for r in range(grid_map.grid.shape[0]):
            grid_map.fix_transitions((r, c), orienation((r, c)))
            # Print for assertion tests
            # print("assert grid_map.grid[{}] == {}".format((r,c),grid_map.grid[(r,c)]))

    assert grid_map.grid[(1, 0)] == 0
    assert grid_map.grid[(2, 0)] == 0
    assert grid_map.grid[(3, 0)] == 0
    assert grid_map.grid[(4, 0)] == 0
    assert grid_map.grid[(5, 0)] == 0
    assert grid_map.grid[(6, 0)] == 0
    assert grid_map.grid[(7, 0)] == 0
    assert grid_map.grid[(8, 0)] == 0
    assert grid_map.grid[(9, 0)] == 0
    assert grid_map.grid[(0, 1)] == 0
    assert grid_map.grid[(1, 1)] == 0
    assert grid_map.grid[(2, 1)] == 0
    assert grid_map.grid[(3, 1)] == 0
    assert grid_map.grid[(4, 1)] == 0
    assert grid_map.grid[(5, 1)] == 0
    assert grid_map.grid[(6, 1)] == 0
    assert grid_map.grid[(7, 1)] == 0
    assert grid_map.grid[(8, 1)] == 0
    assert grid_map.grid[(9, 1)] == 0
    assert grid_map.grid[(0, 2)] == 0
    assert grid_map.grid[(1, 2)] == 0
    assert grid_map.grid[(2, 2)] == 8192
    assert grid_map.grid[(3, 2)] == 49186
    assert grid_map.grid[(4, 2)] == 32800
    assert grid_map.grid[(5, 2)] == 32800
    assert grid_map.grid[(6, 2)] == 32800
    assert grid_map.grid[(7, 2)] == 32872
    assert grid_map.grid[(8, 2)] == 128
    assert grid_map.grid[(9, 2)] == 0
    assert grid_map.grid[(0, 3)] == 0
    assert grid_map.grid[(1, 3)] == 0
    assert grid_map.grid[(2, 3)] == 0
    assert grid_map.grid[(3, 3)] == 4608
    assert grid_map.grid[(4, 3)] == 49186
    assert grid_map.grid[(5, 3)] == 32800
    assert grid_map.grid[(6, 3)] == 32872
    assert grid_map.grid[(7, 3)] == 2064
    assert grid_map.grid[(8, 3)] == 0
    assert grid_map.grid[(9, 3)] == 0
    assert grid_map.grid[(0, 4)] == 0
    assert grid_map.grid[(1, 4)] == 0
    assert grid_map.grid[(2, 4)] == 0
    assert grid_map.grid[(3, 4)] == 0
    assert grid_map.grid[(4, 4)] == 4608
    assert grid_map.grid[(5, 4)] == 32800
    assert grid_map.grid[(6, 4)] == 2064
    assert grid_map.grid[(7, 4)] == 0
    assert grid_map.grid[(8, 4)] == 0
    assert grid_map.grid[(9, 4)] == 0
    assert grid_map.grid[(0, 5)] == 0
    assert grid_map.grid[(1, 5)] == 0
    assert grid_map.grid[(2, 5)] == 0
    assert grid_map.grid[(3, 5)] == 0
    assert grid_map.grid[(4, 5)] == 0
    assert grid_map.grid[(5, 5)] == 0
    assert grid_map.grid[(6, 5)] == 0
    assert grid_map.grid[(7, 5)] == 0
    assert grid_map.grid[(8, 5)] == 0
    assert grid_map.grid[(9, 5)] == 0
