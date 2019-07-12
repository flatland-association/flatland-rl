#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from flatland.core.grid.grid4 import Grid4Transitions
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.agent_utils import EnvAgentStatic
from flatland.envs.generators import complex_rail_generator
from flatland.envs.generators import rail_from_grid_transition_map
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv

"""Tests for `flatland` package."""


def test_load_env():
    env = RailEnv(10, 10)
    env.load_resource('env_data.tests', 'test-10x10.mpk')

    agent_static = EnvAgentStatic((0, 0), 2, (5, 5), False)
    env.add_agent_static(agent_static)
    assert env.get_num_agents() == 1


def test_save_load():
    env = RailEnv(width=10, height=10,
                  rail_generator=complex_rail_generator(nr_start_goal=2, nr_extra=5, min_dist=6, seed=0),
                  number_of_agents=2)
    env.reset()
    agent_1_pos = env.agents_static[0].position
    agent_1_dir = env.agents_static[0].direction
    agent_1_tar = env.agents_static[0].target
    agent_2_pos = env.agents_static[1].position
    agent_2_dir = env.agents_static[1].direction
    agent_2_tar = env.agents_static[1].target
    env.save("test_save.dat")
    env.load("test_save.dat")
    assert (env.width == 10)
    assert (env.height == 10)
    assert (len(env.agents) == 2)
    assert (agent_1_pos == env.agents_static[0].position)
    assert (agent_1_dir == env.agents_static[0].direction)
    assert (agent_1_tar == env.agents_static[0].target)
    assert (agent_2_pos == env.agents_static[1].position)
    assert (agent_2_dir == env.agents_static[1].direction)
    assert (agent_2_tar == env.agents_static[1].target)


def test_rail_environment_single_agent():
    cells = [int('0000000000000000', 2),  # empty cell - Case 0
             int('1000000000100000', 2),  # Case 1 - straight
             int('1001001000100000', 2),  # Case 2 - simple switch
             int('1000010000100001', 2),  # Case 3 - diamond drossing
             int('1001011000100001', 2),  # Case 4 - single slip switch
             int('1100110000110011', 2),  # Case 5 - double slip switch
             int('0101001000000010', 2),  # Case 6 - symmetrical switch
             int('0010000000000000', 2)]  # Case 7 - dead end

    # We instantiate the following map on a 3x3 grid
    #  _  _
    # / \/ \
    # | |  |
    # \_/\_/

    transitions = RailEnvTransitions()
    vertical_line = cells[1]
    south_symmetrical_switch = cells[6]
    north_symmetrical_switch = transitions.rotate_transition(south_symmetrical_switch, 180)
    # Simple turn not in the base transitions ?
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
    rail_env = RailEnv(width=3,
                       height=3,
                       rail_generator=rail_from_grid_transition_map(rail),
                       number_of_agents=1,
                       obs_builder_object=GlobalObsForRailEnv())

    for _ in range(200):
        _ = rail_env.reset()

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


test_rail_environment_single_agent()


def test_dead_end():
    transitions = Grid4Transitions([])

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
    rail_env = RailEnv(width=rail_map.shape[1],
                       height=rail_map.shape[0],
                       rail_generator=rail_from_grid_transition_map(rail),
                       number_of_agents=1,
                       obs_builder_object=GlobalObsForRailEnv())

    def check_consistency(rail_env):
        # We run step to check that trains do not move anymore
        # after being done.
        # TODO: GIACOMO: this is deprecated and should be updated; thenew behavior is that agents keep moving
        # until they are manually stopped.
        for i in range(7):
            prev_pos = rail_env.agents[0].position

            # The train cannot turn, so we check that when it tries,
            # it stays where it is.
            _ = rail_env.step({0: 1})
            _ = rail_env.step({0: 3})
            assert (rail_env.agents[0].position == prev_pos)
            _, _, dones, _ = rail_env.step({0: 2})

            if i < 5:
                assert (not dones[0] and not dones['__all__'])
            else:
                assert (dones[0] and dones['__all__'])

    # We try the configuration in the 4 directions:
    rail_env.reset()
    rail_env.agents = [EnvAgent(position=(0, 2), direction=1, target=(0, 0), moving=False)]

    rail_env.reset()
    rail_env.agents = [EnvAgent(position=(0, 2), direction=3, target=(0, 4), moving=False)]

    # In the vertical configuration:
    rail_map = np.array(
        [[dead_end_from_south]] + [[straight_vertical]] * 3 +
        [[transitions.rotate_transition(dead_end_from_south, 180)]],
        dtype=np.uint16)

    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0],
                             transitions=transitions)

    rail.grid = rail_map
    rail_env = RailEnv(width=rail_map.shape[1],
                       height=rail_map.shape[0],
                       rail_generator=rail_from_grid_transition_map(rail),
                       number_of_agents=1,
                       obs_builder_object=GlobalObsForRailEnv())

    rail_env.reset()
    rail_env.agents = [EnvAgent(position=(2, 0), direction=2, target=(0, 0), moving=False)]

    rail_env.reset()
    rail_env.agents = [EnvAgent(position=(2, 0), direction=0, target=(4, 0), moving=False)]
