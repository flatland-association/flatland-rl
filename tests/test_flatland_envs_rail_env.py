#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import complex_rail_generator, rail_from_file
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator, complex_schedule_generator, schedule_from_file
from flatland.utils.simple_rail import make_simple_rail

"""Tests for `flatland` package."""


def test_load_env():
    env = RailEnv(10, 10)
    env.reset()
    env.load_resource('env_data.tests', 'test-10x10.mpk')

    agent_static = EnvAgent((0, 0), 2, (5, 5), False)
    env.add_agent(agent_static)
    assert env.get_num_agents() == 1


def test_save_load():
    env = RailEnv(width=10, height=10,
                  rail_generator=complex_rail_generator(nr_start_goal=2, nr_extra=5, min_dist=6, seed=1),
                  schedule_generator=complex_schedule_generator(), number_of_agents=2)
    env.reset()
    agent_1_pos = env.agents[0].position
    agent_1_dir = env.agents[0].direction
    agent_1_tar = env.agents[0].target
    agent_2_pos = env.agents[1].position
    agent_2_dir = env.agents[1].direction
    agent_2_tar = env.agents[1].target
    env.save("test_save.dat")
    env.load("test_save.dat")
    assert (env.width == 10)
    assert (env.height == 10)
    assert (len(env.agents) == 2)
    assert (agent_1_pos == env.agents[0].position)
    assert (agent_1_dir == env.agents[0].direction)
    assert (agent_1_tar == env.agents[0].target)
    assert (agent_2_pos == env.agents[1].position)
    assert (agent_2_dir == env.agents[1].direction)
    assert (agent_2_tar == env.agents[1].target)


def test_rail_environment_single_agent():
    # We instantiate the following map on a 3x3 grid
    #  _  _
    # / \/ \
    # | |  |
    # \_/\_/

    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    vertical_line = cells[1]
    south_symmetrical_switch = cells[6]
    north_symmetrical_switch = transitions.rotate_transition(south_symmetrical_switch, 180)
    south_east_turn = int('0100000000000010', 2)
    south_west_turn = transitions.rotate_transition(south_east_turn, 90)
    north_east_turn = transitions.rotate_transition(south_east_turn, 270)
    north_west_turn = transitions.rotate_transition(south_east_turn, 180)

    rail_map = np.array([[south_east_turn, south_symmetrical_switch,
                          south_west_turn],
                         [vertical_line, vertical_line, vertical_line],
                         [north_east_turn, north_symmetrical_switch,
                          north_west_turn]],
                        dtype=np.uint16)

    rail = GridTransitionMap(width=3, height=3, transitions=transitions)
    rail.grid = rail_map
    rail_env = RailEnv(width=3, height=3, rail_generator=rail_from_grid_transition_map(rail),
                       schedule_generator=random_schedule_generator(), number_of_agents=1,
                       obs_builder_object=GlobalObsForRailEnv())

    for _ in range(50):
        _ = rail_env.reset(False, False, True)

        # We do not care about target for the moment
        agent = rail_env.agents[0]
        agent.target = [-1, -1]

        # Check that trains are always initialized at a consistent position
        # or direction.
        # They should always be able to go somewhere.
        assert (transitions.get_transitions(
            rail_map[agent.position],
            agent.direction) != (0, 0, 0, 0))

        initial_pos = agent.position

        valid_active_actions_done = 0
        pos = initial_pos
        while valid_active_actions_done < 6:
            # We randomly select an action
            action = np.random.randint(4)

            _, _, _, _ = rail_env.step({0: action})

            prev_pos = pos
            pos = agent.position  # rail_env.agents_position[0]
            if prev_pos != pos:
                valid_active_actions_done += 1

        # After 6 movements on this railway network, the train should be back
        # to its original height on the map.
        assert (initial_pos[0] == agent.position[0])

        # We check that the train always attains its target after some time
        for _ in range(10):
            _ = rail_env.reset()

            done = False
            while not done:
                # We randomly select an action
                action = np.random.randint(4)

                _, _, dones, _ = rail_env.step({0: action})
                done = dones['__all__']


def test_dead_end():
    transitions = RailEnvTransitions()

    straight_vertical = int('1000000000100000', 2)  # Case 1 - straight
    straight_horizontal = transitions.rotate_transition(straight_vertical,
                                                        90)

    dead_end_from_south = int('0010000000000000', 2)  # Case 7 - dead end

    # We instantiate the following railway
    # O->-- where > is the train and O the target. After 6 steps,
    # the train should be done.

    rail_map = np.array(
        [[transitions.rotate_transition(dead_end_from_south, 270)] +
         [straight_horizontal] * 3 +
         [transitions.rotate_transition(dead_end_from_south, 90)]],
        dtype=np.uint16)

    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0],
                             transitions=transitions)

    rail.grid = rail_map
    rail_env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                       rail_generator=rail_from_grid_transition_map(rail),
                       schedule_generator=random_schedule_generator(), number_of_agents=1,
                       obs_builder_object=GlobalObsForRailEnv())

    # We try the configuration in the 4 directions:
    rail_env.reset()
    rail_env.agents = [EnvAgent(initial_position=(0, 2), initial_direction=1, direction=1, target=(0, 0), moving=False)]

    rail_env.reset()
    rail_env.agents = [EnvAgent(initial_position=(0, 2), initial_direction=3, direction=3, target=(0, 4), moving=False)]

    # In the vertical configuration:
    rail_map = np.array(
        [[dead_end_from_south]] + [[straight_vertical]] * 3 +
        [[transitions.rotate_transition(dead_end_from_south, 180)]],
        dtype=np.uint16)

    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0],
                             transitions=transitions)

    rail.grid = rail_map
    rail_env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                       rail_generator=rail_from_grid_transition_map(rail),
                       schedule_generator=random_schedule_generator(), number_of_agents=1,
                       obs_builder_object=GlobalObsForRailEnv())

    rail_env.reset()
    rail_env.agents = [EnvAgent(initial_position=(2, 0), initial_direction=2, direction=2, target=(0, 0), moving=False)]

    rail_env.reset()
    rail_env.agents = [EnvAgent(initial_position=(2, 0), initial_direction=0, direction=0, target=(4, 0), moving=False)]

    # TODO make assertions


def test_get_entry_directions():
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()

    def _assert(position, expected):
        actual = env.get_valid_directions_on_grid(*position)
        assert actual == expected, "[{},{}] actual={}, expected={}".format(*position, actual, expected)

    # north dead end
    _assert((0, 3), [True, False, False, False])

    # west dead end
    _assert((3, 0), [False, False, False, True])

    # switch
    _assert((3, 3), [False, True, True, True])

    # horizontal
    _assert((3, 2), [False, True, False, True])

    # vertical
    _assert((2, 3), [True, False, True, False])

    # nowhere
    _assert((0, 0), [False, False, False, False])


def test_rail_env_reset():
    file_name = "test_rail_env_reset.pkl"

    # Test to save and load file.

    rail, rail_map = make_simple_rail()

    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(), number_of_agents=3,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()
    env.save(file_name)
    dist_map_shape = np.shape(env.distance_map.get())
    rails_initial = env.rail.grid
    agents_initial = env.agents

    env2 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
                   schedule_generator=schedule_from_file(file_name), number_of_agents=1,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env2.reset(False, False, False)
    rails_loaded = env2.rail.grid
    agents_loaded = env2.agents

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded

    env3 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
                   schedule_generator=schedule_from_file(file_name), number_of_agents=1,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env3.reset(False, True, False)
    rails_loaded = env3.rail.grid
    agents_loaded = env3.agents

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded

    env4 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
                   schedule_generator=schedule_from_file(file_name), number_of_agents=1,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env4.reset(True, False, False)
    rails_loaded = env4.rail.grid
    agents_loaded = env4.agents

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded
