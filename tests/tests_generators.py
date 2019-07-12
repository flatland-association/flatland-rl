#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from flatland.envs.generators import rail_from_grid_transition_map, rail_from_file, complex_rail_generator, \
    random_rail_generator, empty_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from tests.simple_rail import make_simple_rail


def test_empty_rail_generator():
    np.random.seed(0)
    n_agents = 1
    x_dim = 5
    y_dim = 10

    # Check that a random level at with correct parameters is generated
    env = RailEnv(width=x_dim,
                  height=y_dim,
                  number_of_agents=n_agents,
                  rail_generator=empty_rail_generator()
                  )
    # Check the dimensions
    assert env.rail.grid.shape == (y_dim, x_dim)
    # Check that no grid was generated
    assert np.count_nonzero(env.rail.grid) == 0
    # Check that no agents where placed
    assert env.get_num_agents() == 0


def test_random_rail_generator():
    np.random.seed(0)
    n_agents = 1
    x_dim = 5
    y_dim = 10

    # Check that a random level at with correct parameters is generated
    env = RailEnv(width=x_dim,
                  height=y_dim,
                  number_of_agents=n_agents,
                  rail_generator=random_rail_generator()
                  )
    assert env.rail.grid.shape == (y_dim, x_dim)
    assert env.get_num_agents() == n_agents


def test_complex_rail_generator():
    n_agents = 10
    n_start = 2
    x_dim = 10
    y_dim = 10
    min_dist = 4

    # Check that agent number is changed to fit generated level
    env = RailEnv(width=x_dim,
                  height=y_dim,
                  number_of_agents=n_agents,
                  rail_generator=complex_rail_generator(nr_start_goal=n_start, nr_extra=0, min_dist=min_dist)
                  )
    assert env.get_num_agents() == 2
    assert env.rail.grid.shape == (y_dim, x_dim)

    min_dist = 2 * x_dim

    # Check that no agents are generated when level cannot be generated
    env = RailEnv(width=x_dim,
                  height=y_dim,
                  number_of_agents=n_agents,
                  rail_generator=complex_rail_generator(nr_start_goal=n_start, nr_extra=0, min_dist=min_dist)
                  )
    assert env.get_num_agents() == 0
    assert env.rail.grid.shape == (y_dim, x_dim)

    # Check that everything stays the same when correct parameters are given
    min_dist = 2
    n_start = 5
    n_agents = 5

    env = RailEnv(width=x_dim,
                  height=y_dim,
                  number_of_agents=n_agents,
                  rail_generator=complex_rail_generator(nr_start_goal=n_start, nr_extra=0, min_dist=min_dist)
                  )
    assert env.get_num_agents() == n_agents
    assert env.rail.grid.shape == (y_dim, x_dim)


def test_rail_from_grid_transition_map():
    rail, rail_map = make_simple_rail()
    n_agents = 3
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  number_of_agents=n_agents
                  )
    nr_rail_elements = np.count_nonzero(env.rail.grid)

    # Check if the number of non-empty rail cells is ok
    assert nr_rail_elements == 16

    # Check that agents are placed on a rail
    for a in env.agents:
        assert env.rail.grid[a.position] != 0

    assert env.get_num_agents() == n_agents


def tests_rail_from_file():
    file_name = "test_pkl.pkl"
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_grid_transition_map(rail),
                  number_of_agents=3,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )
    env.save(file_name)
    # initialize agents_static
    rails_initial = env.rail.grid
    agents_initial = env.agents

    env = RailEnv(width=1,
                  height=1,
                  rail_generator=rail_from_file(file_name),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )
    rails_loaded = env.rail.grid
    agents_loaded = env.agents

    assert np.all(np.array_equal(rails_initial, rails_loaded))
    assert agents_initial == agents_loaded

