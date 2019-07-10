#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flatland.envs.generators import rail_from_GridTransitionMap_generator, empty_rail_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from tests.simple_rail import make_simple_rail


def test_load_pkl():
    rail, rail_map = make_simple_rail()
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_GridTransitionMap_generator(rail),
                  number_of_agents=3,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  )
    env.save("test_pkl.pkl")
    # initialize agents_static
    obs_0 = env.reset(False, False)
    file_name = "test_pkl.pkl"

    env = RailEnv(width=1,
                  height=1,
                  rail_generator=empty_rail_generator(),
                  number_of_agents=1,
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv()),
                  file_name=file_name
                  )
    obs_1 = env.reset(False, False)
    assert obs_0 == obs_1
    return
