#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from flatland.core.transition_map import GridTransitionMap, Grid4Transitions
from flatland.envs.generators import rail_from_GridTransitionMap_generator
from flatland.envs.observations import GlobalObsForRailEnv
from flatland.envs.rail_env import RailEnv

"""Tests for `flatland` package."""


def test_global_obs():
    # We instantiate a very simple rail network on a 7x10 grid:
    #        |
    #        |
    #        |
    # _ _ _ /_\ _ _  _  _ _ _
    #               \ /
    #                |
    #                |
    #                |

    cells = [int('0000000000000000', 2),  # empty cell - Case 0
             int('1000000000100000', 2),  # Case 1 - straight
             int('1001001000100000', 2),  # Case 2 - simple switch
             int('1000010000100001', 2),  # Case 3 - diamond drossing
             int('1001011000100001', 2),  # Case 4 - single slip switch
             int('1100110000110011', 2),  # Case 5 - double slip switch
             int('0101001000000010', 2),  # Case 6 - symmetrical switch
             int('0010000000000000', 2)]  # Case 7 - dead end

    transitions = Grid4Transitions([])
    empty = cells[0]

    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)

    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)

    double_switch_south_horizontal_straight = horizontal_straight + cells[6]
    double_switch_north_horizontal_straight = transitions.rotate_transition(
        double_switch_south_horizontal_straight, 180)

    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 +
         [double_switch_north_horizontal_straight] +
         [horizontal_straight] * 2 + [double_switch_south_horizontal_straight] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)

    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    env = RailEnv(width=rail_map.shape[1],
                  height=rail_map.shape[0],
                  rail_generator=rail_from_GridTransitionMap_generator(rail),
                  number_of_agents=1,
                  obs_builder_object=GlobalObsForRailEnv())

    global_obs = env.reset()

    assert (global_obs[0][0].shape == rail_map.shape + (16,))

    rail_map_recons = np.zeros_like(rail_map)
    for i in range(global_obs[0][0].shape[0]):
        for j in range(global_obs[0][0].shape[1]):
            rail_map_recons[i, j] = int(
                ''.join(global_obs[0][0][i, j].astype(int).astype(str)), 2)

    assert (rail_map_recons.all() == rail_map.all())

    # If this assertion is wrong, it means that the observation returned
    # places the agent on an empty cell
    assert (np.sum(rail_map * global_obs[0][1][:, :, :4].sum(2)) > 0)
