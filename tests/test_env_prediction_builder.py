#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from flatland.core.transition_map import GridTransitionMap, Grid4Transitions
from flatland.envs.generators import rail_from_GridTransitionMap_generator
from flatland.envs.observations import TreeObsForRailEnv
from flatland.envs.predictions import DummyPredictorForRailEnv
from flatland.envs.rail_env import RailEnv

"""Test predictions for `flatland` package."""


def test_predictions():
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
                  obs_builder_object=TreeObsForRailEnv(max_depth=2, predictor=DummyPredictorForRailEnv(max_depth=20)),
                  )

    env.reset()

    # set initial position and direction for testing...
    env.agents[0].position = (5, 6)
    env.agents[0].direction = 0
    env.agents[0].target = (3., 0.)

    predictions = env.obs_builder.predictor.get()
    positions = np.array(list(map(lambda prediction: [prediction[1], prediction[2]], predictions[0])))
    directions = np.array(list(map(lambda prediction: [prediction[3]], predictions[0])))
    time_offsets = np.array(list(map(lambda prediction: [prediction[0]], predictions[0])))
    actions = np.array(list(map(lambda prediction: [prediction[4]], predictions[0])))

    # compare against expected values
    expected_positions = np.array([[5., 6.],
                                   [4., 6.],
                                   [3., 6.],
                                   [3., 5.],
                                   [3., 4.],
                                   [3., 3.],
                                   [3., 2.],
                                   [3., 1.],
                                   # at target (3,0): stay in this position from here on
                                   [3., 0.],
                                   [3., 0.],
                                   [3., 0.],
                                   ])
    expected_directions = np.array([[0.],
                                    [0.],
                                    [0.],
                                    [3.],
                                    [3.],
                                    [3.],
                                    [3.],
                                    [3.],
                                    # at target (3,0): stay in this position from here on
                                    [3.],
                                    [3.],
                                    [3.]
                                    ])
    expected_time_offsets = np.array([[0.],
                                      [1.],
                                      [2.],
                                      [3.],
                                      [4.],
                                      [5.],
                                      [6.],
                                      [7.],
                                      [8.],
                                      [9.],
                                      [10.],
                                      ])
    expected_actions = np.array([[0.],
                                 [2.],
                                 [2.],
                                 [1.],
                                 [2.],
                                 [2.],
                                 [2.],
                                 [2.],
                                 # reaching target by straight
                                 [2.],
                                 # at target: stopped moving
                                 [4.],
                                 [4.],
                                 ])
    assert np.array_equal(positions, expected_positions)
    assert np.array_equal(directions, expected_directions)
    assert np.array_equal(time_offsets, expected_time_offsets)
    assert np.array_equal(actions, expected_actions)


def main():
    test_predictions()


if __name__ == "__main__":
    main()
