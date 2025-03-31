#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import pytest

from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.agent_utils import EnvAgent
from flatland.envs.line_generators import sparse_line_generator, line_from_file
from flatland.envs.observations import GlobalObsForRailEnv, TreeObsForRailEnv
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.rail_generators import sparse_rail_generator, rail_from_file
from flatland.utils.rendertools import RenderTool
from flatland.utils.simple_rail import make_simple_rail

"""Tests for `flatland` package."""


def test_save_load():
    env = RailEnv(width=30, height=30,
                  rail_generator=sparse_rail_generator(seed=1),
                  line_generator=sparse_line_generator(), number_of_agents=2)
    env.reset()

    agent_1_pos = env.agents[0].position
    agent_1_dir = env.agents[0].direction
    agent_1_tar = env.agents[0].target
    agent_2_pos = env.agents[1].position
    agent_2_dir = env.agents[1].direction
    agent_2_tar = env.agents[1].target

    os.makedirs("tmp", exist_ok=True)

    RailEnvPersister.save(env, "tmp/test_save.pkl")
    env.save("tmp/test_save_2.pkl")

    # env.load("test_save.dat")
    env, env_dict = RailEnvPersister.load_new("tmp/test_save.pkl")
    assert (env.width == 30)
    assert (env.height == 30)
    assert (len(env.agents) == 2)
    assert (agent_1_pos == env.agents[0].position)
    assert (agent_1_dir == env.agents[0].direction)
    assert (agent_1_tar == env.agents[0].target)
    assert (agent_2_pos == env.agents[1].position)
    assert (agent_2_dir == env.agents[1].direction)
    assert (agent_2_tar == env.agents[1].target)


@pytest.mark.skip("Msgpack serializing not supported")
def test_save_load_mpk():
    env = RailEnv(width=30, height=30,
                  rail_generator=sparse_rail_generator(seed=1),
                  line_generator=sparse_line_generator(), number_of_agents=2)
    env.reset()

    os.makedirs("tmp", exist_ok=True)

    RailEnvPersister.save(env, "tmp/test_save.mpk")

    # env.load("test_save.dat")
    env2, env_dict = RailEnvPersister.load_new("tmp/test_save.mpk")
    assert (env.width == env2.width)
    assert (env.height == env2.height)
    assert (len(env2.agents) == len(env.agents))

    for agent1, agent2 in zip(env.agents, env2.agents):
        assert (agent1.position == agent2.position)
        assert (agent1.direction == agent2.direction)
        assert (agent1.target == agent2.target)


@pytest.mark.skip(reason="Old file used to create env, not sure how to regenerate")
def test_rail_environment_single_agent(show=False):
    # We instantiate the following map on a 3x3 grid
    #  _  _
    # / \/ \
    # | |  |
    # \_/\_/

    transitions = RailEnvTransitions()

    if False:
        # This env creation doesn't quite work right.
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
                           line_generator=sparse_line_generator(), number_of_agents=1,
                           obs_builder_object=GlobalObsForRailEnv())
    else:
        rail_env, env_dict = RailEnvPersister.load_new("test_env_loop.pkl", "env_data.tests")
        rail_map = rail_env.rail.grid

    rail_env._max_episode_steps = 1000

    _ = rail_env.reset(False, False, True)

    liActions = [int(a) for a in RailEnvActions]

    env_renderer = RenderTool(rail_env)

    # RailEnvPersister.save(rail_env, "test_env_figure8.pkl")

    for _ in range(5):

        # rail_env.agents[0].initial_position = (1,2)
        _ = rail_env.reset(False, False, True)

        # We do not care about target for the moment
        agent = rail_env.agents[0]
        agent.target = [-1, -1]

        # Check that trains are always initialized at a consistent position
        # or direction.
        # They should always be able to go somewhere.
        if show:
            print("After reset - agent pos:", agent.position, "dir: ", agent.direction)
            print(transitions.get_transitions(rail_map[agent.position], agent.direction))

        # assert (transitions.get_transitions(
        #    rail_map[agent.position],
        #    agent.direction) != (0, 0, 0, 0))

        # HACK - force the direction to one we know is good.
        # agent.initial_position = agent.position = (2,3)
        agent.initial_direction = agent.direction = 0

        if show:
            print("handle:", agent.handle)
        # agent.initial_position = initial_pos = agent.position

        valid_active_actions_done = 0
        pos = agent.position

        if show:
            env_renderer.render_env(show=show, show_agents=True)
            time.sleep(0.01)

        iStep = 0
        while valid_active_actions_done < 6:
            # We randomly select an action
            action = np.random.choice(liActions)
            # action = RailEnvActions.MOVE_FORWARD

            _, _, dict_done, _ = rail_env.step({0: action})

            prev_pos = pos
            pos = agent.position  # rail_env.agents_position[0]

            print("action:", action, "pos:", agent.position, "prev:", prev_pos, agent.direction)
            print(dict_done)
            if prev_pos != pos:
                valid_active_actions_done += 1
            iStep += 1

            if show:
                env_renderer.render_env(show=show, show_agents=True, step=iStep)
                time.sleep(0.01)
            assert iStep < 100, "valid actions should have been performed by now - hung agent"

        # After 6 movements on this railway network, the train should be back
        # to its original height on the map.
        # assert (initial_pos[0] == agent.position[0])

        # We check that the train always attains its target after some time
        for _ in range(10):
            _ = rail_env.reset()

            rail_env.agents[0].direction = 0

            # JW - to avoid problem with sparse_line_generator.
            # rail_env.agents[0].position = (1,2)

            iStep = 0
            while iStep < 100:
                # We randomly select an action
                action = np.random.choice(liActions)

                _, _, dones, _ = rail_env.step({0: action})
                done = dones['__all__']
                if done:
                    break
                iStep += 1
                assert iStep < 100, "agent should have finished by now"
                env_renderer.render_env(show=show)


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

    city_positions = [(0, 0), (0, 3)]
    train_stations = [
        [((0, 0), 0)],
        [((0, 0), 0)],
    ]
    city_orientations = [0, 2]
    agents_hints = {'num_agents': 2,
                    'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                    }
    optionals = {'agents_hints': agents_hints}

    rail_env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                       rail_generator=rail_from_grid_transition_map(rail, optionals),
                       line_generator=sparse_line_generator(), number_of_agents=1,
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

    city_positions = [(0, 0), (0, 3)]
    train_stations = [
        [((0, 0), 0)],
        [((0, 0), 0)],
    ]
    city_orientations = [0, 2]
    agents_hints = {'num_agents': 2,
                    'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                    }
    optionals = {'agents_hints': agents_hints}

    rail.grid = rail_map
    rail_env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                       rail_generator=rail_from_grid_transition_map(rail, optionals),
                       line_generator=sparse_line_generator(), number_of_agents=1,
                       obs_builder_object=GlobalObsForRailEnv())

    rail_env.reset()
    rail_env.agents = [EnvAgent(initial_position=(2, 0), initial_direction=2, direction=2, target=(0, 0), moving=False)]

    rail_env.reset()
    rail_env.agents = [EnvAgent(initial_position=(2, 0), initial_direction=0, direction=0, target=(4, 0), moving=False)]

    # TODO make assertions


def test_get_entry_directions():
    rail, rail_map, optionals = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=1,
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

    rail, rail_map, optionals = make_simple_rail()

    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=3,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()

    # env.save(file_name)
    RailEnvPersister.save(env, file_name)

    dist_map_shape = np.shape(env.distance_map.get())
    rails_initial = env.rail.grid
    agents_initial = env.agents

    # env2 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
    #               line_generator=line_from_file(file_name), number_of_agents=1,
    #               obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    # env2.reset(False, False, False)
    env2, env2_dict = RailEnvPersister.load_new(file_name)

    rails_loaded = env2.rail.grid
    agents_loaded = env2.agents

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded

    env3 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
                   line_generator=line_from_file(file_name), number_of_agents=1,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env3.reset(False, True)
    rails_loaded = env3.rail.grid
    agents_loaded = env3.agents
    # override `earliest_departure` & `latest_arrival` since they aren't expected to be the same
    for agent_initial, agent_loaded in zip(agents_initial, agents_loaded):
        agent_loaded.earliest_departure = agent_initial.earliest_departure
        agent_loaded.latest_arrival = agent_initial.latest_arrival
        agent_loaded.waypoints_earliest_departure = [agent_initial.earliest_departure, None]
        agent_loaded.waypoints_latest_arrival = [None, agent_initial.latest_arrival]

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded

    env4 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
                   line_generator=line_from_file(file_name), number_of_agents=1,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env4.reset(True, False)
    rails_loaded = env4.rail.grid
    agents_loaded = env4.agents
    # override `earliest_departure` & `latest_arrival` since they aren't expected to be the same
    for agent_initial, agent_loaded in zip(agents_initial, agents_loaded):
        agent_loaded.earliest_departure = agent_initial.earliest_departure
        agent_loaded.latest_arrival = agent_initial.latest_arrival
        agent_loaded.waypoints_earliest_departure = [agent_initial.earliest_departure, None]
        agent_loaded.waypoints_latest_arrival = [None, agent_initial.latest_arrival]

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded


def test_process_illegal_action():
    assert RailEnv._process_illegal_action(None) == RailEnvActions.DO_NOTHING
    assert RailEnv._process_illegal_action(0) == RailEnvActions.DO_NOTHING
    assert RailEnv._process_illegal_action(RailEnvActions.DO_NOTHING) == RailEnvActions.DO_NOTHING
    assert RailEnv._process_illegal_action("Alice") == RailEnvActions.DO_NOTHING
    assert RailEnv._process_illegal_action("MOVE_LEFT") == RailEnvActions.DO_NOTHING
    assert RailEnv._process_illegal_action(1) == RailEnvActions.MOVE_LEFT
