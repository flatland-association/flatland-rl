import numpy as np

from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_grid_transition_map import RailGridTransitionMap


def test_walker():
    # _ _ _

    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)

    rail_map = np.array(
        [[dead_end_from_east] + [horizontal_straight] + [dead_end_from_west]], dtype=np.uint16)
    rail = RailGridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map

    city_positions = [(0, 2), (0, 1)]
    train_stations = [
        [((0, 1), 0)],
        [((0, 2), 0)],
    ]
    city_orientations = [1, 0]
    agents_hints = {
        'num_agents': 1,
        'city_positions': city_positions,
        'train_stations': train_stations,
        'city_orientations': city_orientations
    }
    optionals = {'agents_hints': agents_hints}

    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2,
                                                       predictor=ShortestPathPredictorForRailEnv(max_depth=10)),
                  )
    env.reset()

    # set initial position and direction for testing...
    env.agents[0].position = (0, 1)
    env.agents[0].direction = 1
    env.agents[0].target = (0, 0)
    # reset to set agents from agents_static
    # env.reset(False, False)
    env.distance_map._compute(env.agents, env.rail)

    print(env.distance_map.get()[(0, *[0, 1], 1)])
    assert env.distance_map.get()[(0, *[0, 1], 1)] == 3
    print(env.distance_map.get()[(0, *[0, 2], 3)])
    assert env.distance_map.get()[(0, *[0, 2], 1)] == 2
