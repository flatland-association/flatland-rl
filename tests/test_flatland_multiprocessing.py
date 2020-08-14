#!/usr/bin/env python
# -*- coding: utf-8 -*-
from multiprocessing.pool import Pool

from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import rail_from_grid_transition_map
from flatland.envs.schedule_generators import random_schedule_generator
from flatland.utils.simple_rail import make_simple_rail

"""Tests for `flatland` package."""


def test_multiprocessing_tree_obs():
    number_of_agents = 5
    rail, rail_map = make_simple_rail()

    obs_builder = TreeObsForRailEnv(max_depth=2, predictor=ShortestPathPredictorForRailEnv())

    env = RailEnv(width=rail_map.shape[1], height=rail_map.shape[0], rail_generator=rail_from_grid_transition_map(rail),
                  schedule_generator=random_schedule_generator(), number_of_agents=number_of_agents,
                  obs_builder_object=obs_builder)
    env.reset(True, True)

    pool = Pool()
    pool.map(obs_builder.get, range(number_of_agents))


def main():
    test_multiprocessing_tree_obs()


if __name__ == "__main__":
    main()
