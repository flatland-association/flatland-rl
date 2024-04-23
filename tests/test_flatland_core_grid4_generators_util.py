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
    connection_001_start = (2,2)
    connection_001_end = (8, 8)                               
    connection_001_expected = (13, connection_001_start,connection_001_end) 
    print(connection_001)
    # Make connection with dead-ends on both sides
    start_point = (1, 3)
    end_point = (1, 7)
    connection_002 = connect_rail_in_grid_map(grid_map, start_point, end_point, rail_trans, flip_start_node_trans=True,
                                              flip_end_node_trans=True, respect_transition_validity=True,
                                              forbidden_cells=None)
    connection_002_start = (1,3)
    connection_002_end = (1, 7)
    connection_002_expected = (5, connection_002_start,connection_002_end) 

    # Make connection with dead-ends on both sides
    start_point = (6, 1)
    end_point = (6, 9)
    connection_003 = connect_rail_in_grid_map(grid_map, start_point, end_point, rail_trans, flip_start_node_trans=True,
                                              flip_end_node_trans=True, respect_transition_validity=True,
                                            forbidden_cells=None)
    connection_003_start = (6,1)
    connection_003_end = (6, 9)
    connection_003_expected = (9, connection_003_start, connection_003_end)


    # Make connection with dead-ends on both sides
    start_point = (4, 0)
    end_point = (8, 10)
    connection_004 = connect_rail_in_grid_map(grid_map, start_point, end_point, rail_trans, flip_start_node_trans=True,
                                              flip_end_node_trans=True, respect_transition_validity=True,
                                              forbidden_cells=None)
    connection_004_start = (4,0)
    connection_004_end = (8, 10)
    connection_004_expected = (15, connection_004_start, connection_004_end)


    for (k, (connection, connection_expected)) in enumerate(zip(
        [connection_001, connection_002, connection_003, connection_004],
        [connection_001_expected, connection_002_expected, connection_003_expected, connection_004_expected],
    )):
        assert len(connection) == connection_expected[0], \
            "map {}. actual length={}, expected length={}".format(k+1, len(connection), connection_expected[0])
        assert connection[0] == connection_expected[1], \
            "map {}. actual start={}, expected start={}".format(k+1, connection[0], connection_expected[1])
        assert connection[len(connection)-1] == connection_expected[2], \
            "map {}. actual end={}, expected end={}".format(k+1, connection[len(connection)-1], connection_expected[2])

    #Testing the number of occuppied cells in the Grid
    
    #number of shared cell on the path
    s1= set(connection_001)
    s2 = set(connection_002)
    s3= set(connection_003)
    s4= set(connection_004)

    s = s1.intersection(s2)
    s.update(s1.intersection(s3))
    s.update(s1.intersection(s4))
    s.update(s2.intersection(s3))
    s.update(s2.intersection(s4))
    s.update(s3.intersection(s4))
    intersection = len(s)
    #number of occupied cell on the grid
    occupied_cells_expected = connection_001_expected[0] + connection_002_expected[0] + connection_003_expected[0]+ connection_004_expected[0]-intersection

    count = 0

    #testing grid
    for i in range(20):
        row = grid_map.grid[i]
        occupied = row[row>0]
        count += len(occupied)

    assert count == occupied_cells_expected, \
        "number of occuppied cell={}, expected occupation={}".format(count, occupied_cells_expected)

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
