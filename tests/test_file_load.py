#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from flatland.envs.generators import rail_from_GridTransitionMap_generator, rail_from_file
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from tests.simple_rail import make_simple_rail


def test_load_pkl():
    file_name = "test_pkl.pkl"
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_GridTransitionMap_generator(rail),
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

    return
