#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map, rail_from_file, empty_rail_generator
from flatland.envs.line_generators import sparse_line_generator, line_from_file
from flatland.utils.simple_rail import make_simple_rail
from flatland.envs.persistence import RailEnvPersister


def test_empty_rail_generator():
    n_agents = 2
    x_dim = 5
    y_dim = 10

    # Check that a random level at with correct parameters is generated
    env = RailEnv(width=x_dim, height=y_dim, rail_generator=empty_rail_generator(), number_of_agents=n_agents)
    env.reset()
    # Check the dimensions
    assert env.rail.grid.shape == (y_dim, x_dim)
    # Check that no grid was generated
    assert np.count_nonzero(env.rail.grid) == 0
    # Check that no agents where placed
    assert env.get_num_agents() == 0


def test_rail_from_grid_transition_map():
    rail, rail_map, optionals = make_simple_rail()
    n_agents = 4
    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=n_agents)
    env.reset(False, False, True)
    nr_rail_elements = np.count_nonzero(env.rail.grid)

    # Check if the number of non-empty rail cells is ok
    assert nr_rail_elements == 16

    # Check that agents are placed on a rail
    for a in env.agents:
        assert env.rail.grid[a.position] != 0

    assert env.get_num_agents() == n_agents


def tests_rail_from_file():
    file_name = "test_with_distance_map.pkl"

    # Test to save and load file with distance map.

    rail, rail_map, optionals = make_simple_rail()

    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail, optionals),
                  line_generator=sparse_line_generator(), number_of_agents=3,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()
    #env.save(file_name)
    RailEnvPersister.save(env, file_name)
    dist_map_shape = np.shape(env.distance_map.get())
    rails_initial = env.rail.grid
    agents_initial = env.agents

    env = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
                  line_generator=line_from_file(file_name), number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()))
    env.reset()
    rails_loaded = env.rail.grid
    agents_loaded = env.agents

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded

    # Check that distance map was not recomputed
    assert np.shape(env.distance_map.get()) == dist_map_shape
    assert env.distance_map.get() is not None

    # Test to save and load file without distance map.

    file_name_2 = "test_without_distance_map.pkl"

    env2 = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0],
                   rail_generator=rail_from_grid_transition_map(rail), line_generator=sparse_line_generator(),
                   number_of_agents=3, obs_builder_object=GlobalObsForRailEnv())
    env2.reset()
    #env2.save(file_name_2)
    RailEnvPersister.save(env2, file_name_2)

    rails_initial_2 = env2.rail.grid
    agents_initial_2 = env2.agents

    env2 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name_2),
                   line_generator=line_from_file(file_name_2), number_of_agents=1,
                   obs_builder_object=GlobalObsForRailEnv())
    env2.reset()
    rails_loaded_2 = env2.rail.grid
    agents_loaded_2 = env2.agents

    assert np.all(np.array_equal(rails_initial_2, rails_loaded_2))
    assert agents_initial_2 == agents_loaded_2
    assert not hasattr(env2.obs_builder, "distance_map")

    # Test to save with distance map and load without

    env3 = RailEnv(width=1, height=1, rail_generator=rail_from_file(file_name),
                   line_generator=line_from_file(file_name), number_of_agents=1,
                   obs_builder_object=GlobalObsForRailEnv())
    env3.reset()
    rails_loaded_3 = env3.rail.grid
    agents_loaded_3 = env3.agents

    assert np.all(np.array_equal(rails_initial, rails_loaded_3))
    assert agents_initial == agents_loaded_3
    assert not hasattr(env2.obs_builder, "distance_map")

    # Test to save without distance map and load with generating distance map

    env4 = RailEnv(width=1,
                   height=1,
                   rail_generator=rail_from_file(file_name_2),
                   line_generator=line_from_file(file_name_2),
                   number_of_agents=1,
                   obs_builder_object=TreeObsForRailEnv(max_depth=2),
                   )
    env4.reset()
    rails_loaded_4 = env4.rail.grid
    agents_loaded_4 = env4.agents

    # Check that no distance map was saved
    assert not hasattr(env2.obs_builder, "distance_map")
    assert np.all(np.array_equal(rails_initial_2, rails_loaded_4))
    assert agents_initial_2 == agents_loaded_4

    # Check that distance map was generated with correct shape
    assert env4.distance_map.get() is not None
    assert np.shape(env4.distance_map.get()) == dist_map_shape
