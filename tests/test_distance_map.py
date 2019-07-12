import numpy as np

from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.generators import rail_from_grid_transition_map
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv


def test_walker():
    # _ _ _

    cells = [int('0000000000000000', 2),  # empty cell - Case 0
             int('1000000000100000', 2),  # Case 1 - straight
             int('1001001000100000', 2),  # Case 2 - simple switch
             int('1000010000100001', 2),  # Case 3 - diamond drossing
             int('1001011000100001', 2),  # Case 4 - single slip switch
             int('1100110000110011', 2),  # Case 5 - double slip switch
             int('0101001000000010', 2),  # Case 6 - symmetrical switch
             int('0010000000000000', 2)]  # Case 7 - dead end
    transitions = Grid4Transitions([])
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)

    rail_map = np.array(
        [[dead_end_from_east] + [horizontal_straight] + [dead_end_from_west]], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2,
                                                       predictor=ShortestPathPredictorForRailEnv(max_depth=10)),
                  )
    # reset to initialize agents_static
    env.reset()

    # set initial position and direction for testing...
    env.agents_static[0].position = (0, 1)
    env.agents_static[0].direction = 1
    env.agents_static[0].target = (0, 0)

    # reset to set agents from agents_static
    env.reset(False, False)
    obs_builder: TreeObsForRailEnv = env.obs_builder

    print(obs_builder.distance_map[(0, *[0, 1], 1)])
    assert obs_builder.distance_map[(0, *[0, 1], 1)] == 3
    print(obs_builder.distance_map[(0, *[0, 2], 3)])
    assert obs_builder.distance_map[(0, *[0, 2], 1)] == 2
